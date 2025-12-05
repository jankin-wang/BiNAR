#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import open3d as o3d
import cv2
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

@torch.no_grad()                                        
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def save_tensor_image(tensor, filename):
    array = (tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)  
    cv2.imwrite(filename, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))  

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)                       
    scene = Scene(dataset, gaussians,shuffle=False)                     
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras_color = scene.getTrainCameras_color().copy()
    testCameras_color = scene.getTestCameras_color().copy()
    trainCameras_red = scene.getTrainCameras_red().copy()
    testCameras_red = scene.getTestCameras_red().copy()
    
    highresolution_index_color = []
    for index, camera in enumerate(trainCameras_color):
        if camera.image_width >= 800:
            highresolution_index_color.append(index)
    highresolution_index_red = []
    for index, camera in enumerate(trainCameras_red):
        if camera.image_width >= 800:
            highresolution_index_red.append(index)

    gaussians.compute_3D_filter('color', cameras=trainCameras_color)
    gaussians.compute_3D_filter('red', cameras=trainCameras_red)

    viewpoint_stack_color = None
    viewpoint_stack_red = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # Start Training
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack_color:
            viewpoint_stack_color = scene.getTrainCameras_color().copy()
        randint_index = randint(0, len(viewpoint_stack_color)-1)     
        viewpoint_cam_color = viewpoint_stack_color.pop(randint_index)
        if not viewpoint_stack_red:
            viewpoint_stack_red = scene.getTrainCameras_red().copy()
        viewpoint_cam_red = viewpoint_stack_red.pop(randint_index)
        
        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            randint_index_high = randint(0, len(highresolution_index_color)-1)
            viewpoint_cam_color = trainCameras_color[highresolution_index_color[randint_index_high]]
            viewpoint_cam_red = trainCameras_red[highresolution_index_red[randint_index_high]]
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset_color = torch.rand((int(viewpoint_cam_color.image_height), int(viewpoint_cam_color.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            subpixel_offset_red = torch.rand((int(viewpoint_cam_red.image_height), int(viewpoint_cam_red.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset_color = None
            subpixel_offset_red = None

        # RGB Render
        render_pkg_color = render('color', viewpoint_cam_color, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset_color)
        image_color, viewspace_point_tensor_color, visibility_filter_color, radii_color = render_pkg_color["render"], render_pkg_color["viewspace_points"], render_pkg_color["visibility_filter"], render_pkg_color["radii"]
        
        # RGB Loss
        gt_image_color = viewpoint_cam_color.original_image.cuda()
        # sample gt_image with subpixel offset
        if dataset.resample_gt_image:
            gt_image_color = create_offset_gt(gt_image_color, subpixel_offset_color)

        Ll1_color = l1_loss(image_color, gt_image_color)
        # Calculate total RGB loss as a weighted sum of L1 loss and DSSIM loss
        loss_color = (1.0 - opt.lambda_dssim) * Ll1_color + opt.lambda_dssim * (1.0 - ssim(image_color, gt_image_color))

        # Backpropagate RGB loss
        loss_color.backward()

        with torch.no_grad():
            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter_color] = torch.max(gaussians.max_radii2D[visibility_filter_color], radii_color[visibility_filter_color])
                gaussians.add_densification_stats(viewspace_point_tensor_color, visibility_filter_color)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent_color, size_threshold)
                    gaussians.compute_3D_filter('color', cameras=trainCameras_color)
                    gaussians.compute_3D_filter('red', cameras=trainCameras_red)


                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity_color()
                    gaussians.reset_opacity_red()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter('color', cameras=trainCameras_color)
                    gaussians.compute_3D_filter('red', cameras=trainCameras_red)
        
            # RGB Optimization
            if iteration < opt.iterations:
                gaussians.optimizer_common.step()
                gaussians.optimizer_common.zero_grad(set_to_none = True)
                gaussians.optimizer_color.step()
                gaussians.optimizer_color.zero_grad(set_to_none = True)

        # IR Render
        render_pkg_red = render('red', viewpoint_cam_red, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset_red)
        image_red, viewspace_point_tensor_red, visibility_filter_red, radii_red = render_pkg_red["render"], render_pkg_red["viewspace_points"], render_pkg_red["visibility_filter"], render_pkg_red["radii"]
        
        gt_image_red = viewpoint_cam_red.original_image.cuda()
        # sample gt_image with subpixel offset
        if dataset.resample_gt_image:
            gt_image_red = create_offset_gt(gt_image_red, subpixel_offset_red)

        Ll1_red = l1_loss(image_red, gt_image_red)
        # Calculate total IR loss as a weighted sum of L1 loss and DSSIM loss
        loss_red = (1.0 - opt.lambda_dssim) * Ll1_red + opt.lambda_dssim * (1.0 - ssim(image_red, gt_image_red))
        # Backpropagate IR loss
        loss_red.backward()

        with torch.no_grad():
            ema_loss_for_log_color = 0.4 * loss_color.item() + 0.6 * ema_loss_for_log
            ema_loss_for_log_red = 0.4 * loss_red.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss_color": f"{ema_loss_for_log_color:.7f}", "Loss_red": f"{ema_loss_for_log_red:.7f}"})

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration < opt.iterations:
                gaussians.optimizer_common.step()
                gaussians.optimizer_common.zero_grad(set_to_none = True)
                gaussians.optimizer_red.step()
                gaussians.optimizer_red.zero_grad(set_to_none = True)
                

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        iter_end.record()

def prepare_output_and_logger(args):    
    if not args.model_path:
        args.model_path = os.path.join("./output_scene/", args.scene_name)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")                               
    lp = ModelParams(parser)                                                                         
    op = OptimizationParams(parser)                                                                  
    pp = PipelineParams(parser)                                                                     
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6008)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG) and fix random seed
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)        
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)   

    # All done
    print("\nTraining complete.")
