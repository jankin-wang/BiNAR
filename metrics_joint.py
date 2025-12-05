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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim

import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)

        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, scale_color, scale_red):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir_color = method_dir/ f"gt_color_{scale_color}"
            gt_dir_red = method_dir / f"gt_red_{scale_red}"
            renders_dir_color = method_dir / f"test_preds_color_{scale_color}"
            renders_dir_red = method_dir / f"test_preds_red_{scale_red}"
            renders_color, gts_color, image_names_color = readImages(renders_dir_color, gt_dir_color)
            renders_red, gts_red, image_names_red = readImages(renders_dir_red, gt_dir_red)

            ssims_color = []
            psnrs_color = []
            lpipss_color = []

            ssims_red = []
            psnrs_red = []
            lpipss_red = []

            for idx in tqdm(range(len(renders_color)), desc="Metric evaluation progress"):
                ssims_color.append(ssim(renders_color[idx], gts_color[idx]))
                psnrs_color.append(psnr(renders_color[idx], gts_color[idx]))
                lpipss_color.append(lpips_fn(renders_color[idx], gts_color[idx]).detach())
            
            for idx in tqdm(range(len(renders_red)), desc="Metric evaluation progress"):
                ssims_red.append(ssim(renders_red[idx], gts_red[idx]))
                psnrs_red.append(psnr(renders_red[idx], gts_red[idx]))
                lpipss_red.append(lpips_fn(renders_red[idx], gts_red[idx]).detach())

            print("----------color metrics----------")
            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims_color).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs_color).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss_color).mean(), ".5"))
            print("")
            print("----------red metrics----------")
            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims_red).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs_red).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss_red).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM_color": torch.tensor(ssims_color).mean().item(),
                                                    "PSNR_color": torch.tensor(psnrs_color).mean().item(),
                                                    "LPIPS_color": torch.tensor(lpipss_color).mean().item(),
                                                    "SSIM_red": torch.tensor(ssims_red).mean().item(),
                                                    "PSNR_red": torch.tensor(psnrs_red).mean().item(),
                                                    "LPIPS_red": torch.tensor(lpipss_red).mean().item()})
            
            per_view_dict[scene_dir][method].update({"SSIM_color": {name: ssim for ssim, name in zip(torch.tensor(ssims_color).tolist(), image_names_color)},
                                                        "PSNR_color": {name: psnr for psnr, name in zip(torch.tensor(psnrs_color).tolist(), image_names_color)},
                                                        "LPIPS_color": {name: lp for lp, name in zip(torch.tensor(lpipss_color).tolist(), image_names_color)},
                                                        "SSIM_red": {name: ssim for ssim, name in zip(torch.tensor(ssims_red).tolist(), image_names_red)},
                                                        "PSNR_red": {name: psnr for psnr, name in zip(torch.tensor(psnrs_red).tolist(), image_names_red)},
                                                        "LPIPS_red": {name: lp for lp, name in zip(torch.tensor(lpipss_red).tolist(), image_names_red)}})

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--resolution_color', type=int, default=4)
    parser.add_argument('--resolution_red', type=int, default=1)
    
    args = parser.parse_args()
    evaluate(args.model_paths, args.resolution_color, args.resolution_red)