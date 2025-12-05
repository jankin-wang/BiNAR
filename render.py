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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views_color, views_red, gaussians, pipeline, background, kernel_size, scale_factor_color, scale_factor_red):
    render_path_color = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_color_{scale_factor_color}")
    gts_path_color = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_color_{scale_factor_color}")
    render_path_red = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_red_{scale_factor_red}")
    gts_path_red = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_red_{scale_factor_red}")

    makedirs(render_path_color, exist_ok=True)
    makedirs(gts_path_color, exist_ok=True)
    makedirs(render_path_red, exist_ok=True)
    makedirs(gts_path_red, exist_ok=True)

    for idx, view in enumerate(tqdm(views_color, desc="Rendering progress")):
        rendering_color = render('color', view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
        gt_color = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering_color, os.path.join(render_path_color, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_color, os.path.join(gts_path_color, '{0:05d}'.format(idx) + ".png"))


    for idx, view in enumerate(tqdm(views_red, desc="Rendering progress")):
        rendering_red = render('red', view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
        gt_red = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering_red, os.path.join(render_path_red, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_red, os.path.join(gts_path_red, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scale_factor_color = dataset.resolution_color
        scale_factor_red = dataset.resolution_red
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor_color=scale_factor_color, scale_factor_red= scale_factor_red)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras_color(), scene.getTestCameras_red(), gaussians, pipeline, background, kernel_size, scale_factor_color=scale_factor_color, scale_factor_red= scale_factor_red)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true", default=True)
    parser.add_argument("--skip_test", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    args = get_combined_args(parser)
    
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)