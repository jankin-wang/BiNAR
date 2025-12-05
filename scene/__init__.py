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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos_color,cameraList_from_camInfos_red, camera_to_JSON
from scipy.spatial.transform import Rotation as R
import numpy as np

class  Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales_color=[1.0],resolution_scales_red=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras_color = {}
        self.test_cameras_color = {}
        self.train_cameras_red = {}
        self.test_cameras_red = {}

        # Process RGB 
        if os.path.exists(os.path.join(args.source_path_color, "sparse")):                                       
            scene_info_color = sceneLoadTypeCallbacks["Colmap"](args.source_path_color, args.model_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path_color, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info_color = sceneLoadTypeCallbacks["Blender"](args.source_path_color, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path_color, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info_color = sceneLoadTypeCallbacks["Multi-scale"](args.source_path_color, args.white_background, args.eval, args.load_allres)
        else:
            assert False, "Could not recognize color scene type!"

        # Process IR 
        if os.path.exists(os.path.join(args.source_path_color, "sparse")):                                          
            scene_info_red = sceneLoadTypeCallbacks["Colmap_red"](args.source_path_red, args.source_path_color, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path_color, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info_red = sceneLoadTypeCallbacks["Blender"](args.source_path_red, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path_color, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info_red = sceneLoadTypeCallbacks["Multi-scale"](args.source_path_red, args.white_background, args.eval, args.load_allres)
        else:
            assert False, "Could not recognize red scene type!"


        if not self.loaded_iter:
            dest_file_path = os.path.join(self.model_path, "input.ply")
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
    
            with open(scene_info_color.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info_color.test_cameras:
                camlist.extend(scene_info_color.test_cameras)
            if scene_info_color.train_cameras:
                camlist.extend(scene_info_color.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras_color.json"), 'w') as file:
                json.dump(json_cams, file)

            with open(scene_info_red.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info_red.test_cameras:
                camlist.extend(scene_info_red.test_cameras)
            if scene_info_red.train_cameras:
                camlist.extend(scene_info_red.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras_red.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info_color.train_cameras)  # Multi-res consistent random shuffling
            random.seed(0) 
            random.shuffle(scene_info_red.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent_color = scene_info_color.nerf_normalization["radius"]
        self.cameras_extent_red = scene_info_red.nerf_normalization["radius"]

        # ------------------Adjust IR camera poses-------------------------
        red_rt_path = args.rt_path + '/RT.json'
        if not os.path.exists(red_rt_path):
            print(f"[Error] File not found: {red_rt_path}")
        with open(red_rt_path, 'r') as f:
            red_json = json.load(f)
        
        red_train_cameras = []
        red_test_cameras = []
        len_red_train = len(scene_info_red.train_cameras)
        len_red_test = len(scene_info_red.test_cameras)

        RT = self.rotation_angles_and_translation_to_rt(red_json["RT"])

        for index in range(len_red_test):
            C2W_color = np.eye(4)
            C2W_color[:3, :3] = scene_info_red.test_cameras[index].R.transpose()
            C2W_color[:3, 3] =  scene_info_red.test_cameras[index].T
            W2C_color = np.linalg.inv(C2W_color)
            W2C_red = np.dot(W2C_color, RT)
            C2W_red = np.linalg.inv(W2C_red)
            cam_infos_temp = scene_info_red.test_cameras[index]._replace(R = C2W_red[:3, :3].transpose(), T = C2W_red[:3, 3])
            red_test_cameras.append(cam_infos_temp)
        
        for index in range(len_red_train):
            C2W_color = np.eye(4)
            C2W_color[:3, :3] = scene_info_red.train_cameras[index].R.transpose()
            C2W_color[:3, 3] =  scene_info_red.train_cameras[index].T
            W2C_color = np.linalg.inv(C2W_color)
            W2C_red = np.dot(W2C_color, RT)
            C2W_red = np.linalg.inv(W2C_red)    
            cam_infos_temp = scene_info_red.train_cameras[index]._replace(R = C2W_red[:3, :3].transpose(), T = C2W_red[:3, 3])
            red_train_cameras.append(cam_infos_temp)

        scene_info_red_new = scene_info_red._replace(train_cameras = red_train_cameras, test_cameras = red_test_cameras)
        # ---------------------------------------------------------

        for resolution_scale in resolution_scales_color:
            print("Loading Training Cameras Color")
            self.train_cameras_color[resolution_scale] = cameraList_from_camInfos_color(scene_info_color.train_cameras, resolution_scale, args)
            print("Loading Test Cameras Color")
            self.test_cameras_color[resolution_scale] = cameraList_from_camInfos_color(scene_info_color.test_cameras, resolution_scale, args)

        for resolution_scale in resolution_scales_red:
            print("Loading Training Cameras Red")
            self.train_cameras_red[resolution_scale] = cameraList_from_camInfos_red(scene_info_red_new.train_cameras, resolution_scale, args)
            print("Loading Test Cameras Red")
            self.test_cameras_red[resolution_scale] = cameraList_from_camInfos_red(scene_info_red_new.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info_color.point_cloud, self.cameras_extent_color, self.cameras_extent_red)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras_color(self, scale=1.0):
        return self.train_cameras_color[scale]
    
    def getTrainCameras_red(self, scale=1.0):
        return self.train_cameras_red[scale]

    def getTestCameras_color(self, scale=1.0):
        return self.test_cameras_color[scale]
    
    def getTestCameras_red(self, scale=1.0):
        return self.test_cameras_red[scale]
    
    def rotation_angles_and_translation_to_rt(self, params):
        roll, pitch, yaw = params[:3]
        tx, ty, tz = params[3:]

        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        rotation_matrix = r.as_matrix()

        # Construct the RT matrix
        RT = np.eye(4)
        RT[:3, :3] = rotation_matrix
        RT[:3, 3] = [tx, ty, tz]

        return RT