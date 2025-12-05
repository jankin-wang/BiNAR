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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0                     
        self.max_sh_degree = sh_degree                
        self._xyz = torch.empty(0)                    
        self._features_dc_color = torch.empty(0)      
        self._features_dc_red = torch.empty(0)        
        self._features_rest_color = torch.empty(0)    
        self._features_rest_red = torch.empty(0)      
        self._scaling = torch.empty(0)               
        self._rotation = torch.empty(0)               
        self._opacity_color = torch.empty(0)          
        self._opacity_red = torch.empty(0)            
        self.max_radii2D = torch.empty(0)             
        self.xyz_gradient_accum = torch.empty(0)      
        self.denom = torch.empty(0)
        self.optimizer_common = None
        self.optimizer_color = None
        self.optimizer_red =  None
        self.percent_dense = 0                        
        self.spatial_lr_scale = 0                     
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc_color,
            self._features_dc_red,
            self._features_rest_color,
            self._features_rest_red,
            self._scaling,
            self._rotation,
            self._opacity_color,
            self._opacity_red,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer_common.state_dict(),
            self.optimizer_color.state_dict(),
            self.optimizer_red.state_dict(),
            self.spatial_lr_scale,
            self.filter_3D_color,
            self.filter_3D_red
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc_color, 
        self._features_dc_red,
        self._features_rest_color,
        self._features_rest_red,
        self._scaling, 
        self._rotation, 
        self._opacity_color,
        self._opacity_red,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict_common,
        opt_dict_color,
        opt_dict_red, 
        self.spatial_lr_scale,
        self.filter_3D_color,
        self.filter_3D_red) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer_common.load_state_dict(opt_dict_common)
        self.optimizer_color.load_state_dict(opt_dict_color)
        self.optimizer_red.load_state_dict(opt_dict_red)

    def restore_for_render(self, model_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc_color, 
        self._features_dc_red,
        self._features_rest_color,
        self._features_rest_red,
        self._scaling, 
        self._rotation, 
        self._opacity_color,
        self._opacity_red,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict_common,
        opt_dict_color,
        opt_dict_red, 
        self.spatial_lr_scale,
        self.filter_3D_color,
        self.filter_3D_red) = model_args
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_color_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D_color)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_scaling_red_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D_red)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features_color(self):
        features_dc_color = self._features_dc_color
        features_rest_color = self._features_rest_color
        return torch.cat((features_dc_color, features_rest_color), dim=1)
    
    @property
    def get_features_red(self):
        features_dc_red = self._features_dc_red
        features_rest_red = self._features_rest_red
        return torch.cat((features_dc_red, features_rest_red), dim=1)
    
    @property
    def get_opacity_color(self):
        return self.opacity_activation(self._opacity_color)
    
    @property
    def get_opacity_red(self):
        return self.opacity_activation(self._opacity_red)
    
    @property
    def get_opacity_color_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity_color)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D_color) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]
    
    @property
    def get_opacity_red_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity_red)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D_red) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @torch.no_grad()
    def compute_3D_filter(self, mode, cameras):
        print(f"Computing 3D filter", mode)
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        if mode == 'color':
            self.filter_3D_color = filter_3D[..., None]
        elif mode == 'red':
            self.filter_3D_red = filter_3D[..., None]
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale_color : float, spatial_lr_scale_red : float):
        self.spatial_lr_scale_color = spatial_lr_scale_color
        self.spatial_lr_scale_red = spatial_lr_scale_red
        self.spatial_lr_scale = (self.spatial_lr_scale_color + self.spatial_lr_scale_red)/2
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        fused_color_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features_color = torch.zeros((fused_color_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features_color[:, :3, 0 ] = fused_color_color
        features_color[:, 3:, 1:] = 0.0

        # fused_color_red = RGB2SH(torch.zeros_like(torch.tensor(np.asarray(pcd.colors)).float().cuda()))
        fused_color_red = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features_red = torch.zeros((fused_color_red.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features_red[:, :3, 0 ] = fused_color_red
        features_red[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities_color = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        opacities_red = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc_color = nn.Parameter(features_color[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc_red = nn.Parameter(features_red[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_color = nn.Parameter(features_color[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_red = nn.Parameter(features_red[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity_color = nn.Parameter(opacities_color.requires_grad_(True))
        self._opacity_red = nn.Parameter(opacities_red.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        l_color = [
            {'params': [self._features_dc_color], 'lr': training_args.feature_lr_color, "name": "f_dc_color"},
            {'params': [self._features_rest_color], 'lr': training_args.feature_lr_color / 20.0, "name": "f_rest_color"},
            {'params': [self._opacity_color], 'lr': training_args.opacity_lr_color, "name": "opacity_color"},
        ]

        l_red = [
            {'params': [self._features_dc_red], 'lr': training_args.feature_lr_red, "name": "f_dc_red"},
            {'params': [self._features_rest_red], 'lr': training_args.feature_lr_red / 20.0, "name": "f_rest_red"},
            {'params': [self._opacity_red], 'lr': training_args.opacity_lr_red, "name": "opacity_red"},
        ]
        self.optimizer_common = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.optimizer_color = torch.optim.Adam(l_color, lr=0.0, eps=1e-15)
        self.optimizer_red = torch.optim.Adam(l_red, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer_color.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
        for param_group in self.optimizer_red.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
        for param_group in self.optimizer_common.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, exclude_filter=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc_color.shape[1]*self._features_dc_color.shape[2]):
            l.append('f_dc_color_{}'.format(i))
        for i in range(self._features_dc_red.shape[1]*self._features_dc_red.shape[2]):
            l.append('f_dc_red_{}'.format(i))
        for i in range(self._features_rest_color.shape[1]*self._features_rest_color.shape[2]):
            l.append('f_rest_color_{}'.format(i))
        for i in range(self._features_rest_red.shape[1]*self._features_rest_red.shape[2]):
            l.append('f_rest_red_{}'.format(i))
        l.append('opacity_color')
        l.append('opacity_red')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if not exclude_filter:
            l.append('filter_3D_color')
            l.append('filter_3D_red')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc_color = self._features_dc_color.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc_red = self._features_dc_red.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_color = self._features_rest_color.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_red = self._features_rest_red.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities_color = self._opacity_color.detach().cpu().numpy()
        opacities_red = self._opacity_red.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        filter_3D_color = self.filter_3D_color.detach().cpu().numpy()
        filter_3D_red = self.filter_3D_red.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc_color, f_dc_red, f_rest_color, f_rest_red, opacities_color,  opacities_red, scale, rotation, filter_3D_color, filter_3D_red), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_fused_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc_color = self._features_dc_color.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc_red = self._features_dc_red.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_color = self._features_rest_color.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_red = self._features_rest_red.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # fuse opacity and scale
        current_opacity_color_with_filter = self.get_opacity_color_with_3D_filter
        current_opacity_red_with_filter = self.get_opacity_red_with_3D_filter
        opacities_color = inverse_sigmoid(current_opacity_color_with_filter).detach().cpu().numpy()
        opacities_red = inverse_sigmoid(current_opacity_red_with_filter).detach().cpu().numpy()
        scale = self.scaling_inverse_activation(self.get_scaling_with_3D_filter).detach().cpu().numpy()
        
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc_color, f_dc_red, f_rest_color, f_rest_red, opacities_color,  opacities_red, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity_color(self):
        # reset opacity to by considering 3D filter
        current_opacity_color_with_filter = self.get_opacity_color_with_3D_filter
        opacities_color_new = torch.min(current_opacity_color_with_filter, torch.ones_like(current_opacity_color_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D_color) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_color_new = opacities_color_new / coef[..., None]
        opacities_color_new = inverse_sigmoid(opacities_color_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_color_new, "opacity_color")
        self._opacity_color = optimizable_tensors["opacity_color"]

    def reset_opacity_red(self):
        # reset opacity to by considering 3D filter
        current_opacity_red_with_filter = self.get_opacity_red_with_3D_filter
        opacities_red_new = torch.min(current_opacity_red_with_filter, torch.ones_like(current_opacity_red_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D_red) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_red_new = opacities_red_new / coef[..., None]
        opacities_red_new = inverse_sigmoid(opacities_red_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_red_new, "opacity_red")
        self._opacity_red = optimizable_tensors["opacity_red"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities_color = np.asarray(plydata.elements[0]["opacity_color"])[..., np.newaxis]
        opacities_red = np.asarray(plydata.elements[0]["opacity_red"])[..., np.newaxis]

        filter_3D_color = np.asarray(plydata.elements[0]["filter_3D_color"])[..., np.newaxis]
        filter_3D_red = np.asarray(plydata.elements[0]["filter_3D_red"])[..., np.newaxis]

        features_dc_color = np.zeros((xyz.shape[0], 3, 1))
        features_dc_color[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_color_0"])
        features_dc_color[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_color_1"])
        features_dc_color[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_color_2"])

        features_dc_red = np.zeros((xyz.shape[0], 3, 1))
        features_dc_red[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_red_0"])
        features_dc_red[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_red_1"])
        features_dc_red[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_red_2"])

        extra_f_names_color = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_color_")]
        extra_f_names_color = sorted(extra_f_names_color, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names_color)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra_color = np.zeros((xyz.shape[0], len(extra_f_names_color)))
        for idx, attr_name in enumerate(extra_f_names_color):
            features_extra_color[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra_color = features_extra_color.reshape((features_extra_color.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        extra_f_names_red = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_red_")]
        extra_f_names_red = sorted(extra_f_names_red, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names_red)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra_red = np.zeros((xyz.shape[0], len(extra_f_names_red)))
        for idx, attr_name in enumerate(extra_f_names_red):
            features_extra_red[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra_red = features_extra_red.reshape((features_extra_red.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc_color = nn.Parameter(torch.tensor(features_dc_color, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc_red = nn.Parameter(torch.tensor(features_dc_red, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_color = nn.Parameter(torch.tensor(features_extra_color, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_red = nn.Parameter(torch.tensor(features_extra_red, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity_color = nn.Parameter(torch.tensor(opacities_color, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_red = nn.Parameter(torch.tensor(opacities_red, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D_color = torch.tensor(filter_3D_color, dtype=torch.float, device="cuda")
        self.filter_3D_red = torch.tensor(filter_3D_red, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for myoptimizer in [self.optimizer_common, self.optimizer_color, self.optimizer_red]:
            for group in myoptimizer.param_groups:
                if group["name"] == name:
                    stored_state = myoptimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del myoptimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    myoptimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for myoptimizer in [self.optimizer_common, self.optimizer_color, self.optimizer_red]:
            for group in myoptimizer.param_groups:
                stored_state = myoptimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del myoptimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    myoptimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc_color = optimizable_tensors["f_dc_color"]
        self._features_dc_red = optimizable_tensors["f_dc_red"]
        self._features_rest_color = optimizable_tensors["f_rest_color"]
        self._features_rest_red = optimizable_tensors["f_rest_red"]
        self._opacity_color = optimizable_tensors["opacity_color"]
        self._opacity_red = optimizable_tensors["opacity_red"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for myoptimizer in [self.optimizer_common, self.optimizer_color, self.optimizer_red]:
            for group in myoptimizer.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = myoptimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del myoptimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    myoptimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc_color, new_features_rest_color, new_opacities_color, new_features_dc_red, new_features_rest_red, new_opacities_red, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc_color": new_features_dc_color,
        "f_dc_red": new_features_dc_red,
        "f_rest_color": new_features_rest_color,
        "f_rest_red": new_features_rest_red,
        "opacity_color": new_opacities_color,
        "opacity_red": new_opacities_red,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc_color = optimizable_tensors["f_dc_color"]
        self._features_rest_color = optimizable_tensors["f_rest_color"]
        self._opacity_color = optimizable_tensors["opacity_color"]
        self._features_dc_red = optimizable_tensors["f_dc_red"]
        self._features_rest_red = optimizable_tensors["f_rest_red"]
        self._opacity_red = optimizable_tensors["opacity_red"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        #TODO Maybe we don't need to reset the value, it's better to use moving average instead of reset the value
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc_color = self._features_dc_color[selected_pts_mask].repeat(N,1,1)
        new_features_rest_color = self._features_rest_color[selected_pts_mask].repeat(N,1,1)
        new_opacity_color = self._opacity_color[selected_pts_mask].repeat(N,1)
        new_features_dc_red = self._features_dc_red[selected_pts_mask].repeat(N,1,1)
        new_features_rest_red = self._features_rest_red[selected_pts_mask].repeat(N,1,1)
        new_opacity_red = self._opacity_red[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc_color, new_features_rest_color, new_opacity_color, new_features_dc_red, new_features_rest_red, new_opacity_red, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc_color = self._features_dc_color[selected_pts_mask]
        new_features_rest_color = self._features_rest_color[selected_pts_mask]
        new_opacities_color = self._opacity_color[selected_pts_mask]
        new_features_dc_red = self._features_dc_red[selected_pts_mask]
        new_features_rest_red = self._features_rest_red[selected_pts_mask]
        new_opacities_red = self._opacity_red[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc_color, new_features_rest_color, new_opacities_color ,new_features_dc_red, new_features_rest_red, new_opacities_red, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        
        before = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]
        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]

        prune_mask = ((self.get_opacity_color < min_opacity) & (self.get_opacity_red < min_opacity)).squeeze()
        # prune_mask = (self.get_opacity_color < min_opacity).squeeze()
        # prune_mask = ((self.get_opacity_color < min_opacity) | (self.get_opacity_red < min_opacity)).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        prune = self._xyz.shape[0]
        # torch.cuda.empty_cache()
        return clone - before, split - clone, split - prune

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #TODO maybe use max instead of average
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1
