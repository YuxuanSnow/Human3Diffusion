import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import re
from core.utils import get_rays
from rembg import remove 
from kiui.op import recenter

class joint_diffusion_inference_dataset(Dataset):
    def __init__(self, opt, rgb_path_list, white_bg=False):

        if white_bg == False:
            assert False, "Only white background is supported for this dataset"

        self.normalize_mean = (0.5, 0.5, 0.5)
        self.normalize_std = (0.5, 0.5, 0.5)

        self.opt = opt

        self.num_views = opt.num_views

        self.num_views = 32
        self.num_input_views = opt.num_input_views
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        self.prob_grid_distortion = opt.prob_grid_distortion
        self.prob_cam_jitter = opt.prob_cam_jitter
        
        self.dataset_name = 'wild'

        self.blender2opengl = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)
        
        self.items = []
        for subject in rgb_path_list:
            self.items.append(subject)

        num_subj = len(self.items)

        print(f"Infer {num_subj} images")

        self.fovy = self.opt.fovy
        self.zfar = self.opt.zfar
        self.znear = self.opt.znear
        self.cam_radius = self.opt.cam_radius
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.zfar + self.znear) / (self.zfar - self.znear)
        self.proj_matrix[3, 2] = - (self.zfar * self.znear) / (self.zfar - self.znear)
        self.proj_matrix[2, 3] = 1
    
    def __len__(self):
        return len(self.items)

    @staticmethod
    def preprocess(input_image_path):
        input_image_PIL = Image.open(input_image_path)
        input_image_rgba_detec = np.array(Image.open(input_image_path), np.uint8)
        
        if input_image_rgba_detec.shape[-1] == 4:
            print('Infer RGBA, not using rembg')
            input_image_np = np.array(input_image_PIL, np.uint8) 
            input_mask_np = input_image_np[:, :, 3]

            output_image_np_resized = recenter(input_image_np, input_mask_np, border_ratio=0.227) 
            output_image_torch = torch.from_numpy(output_image_np_resized.astype(np.float32) / 255) 

            output_image_torch = output_image_torch.permute(2, 0, 1) 
            output_mask_torch = output_image_torch[3:4]
            output_image_torch = output_image_torch[:3] * output_mask_torch + (1 - output_mask_torch) 

        elif input_image_rgba_detec.shape[-1] == 3:
            print('Infer RGB, using rembg')
            output_image_PIL = remove(input_image_PIL) 
            output_image_np = np.array(output_image_PIL, np.uint8) 

            output_mask_np = output_image_np[:, :, 3]
            output_image_np_resized = recenter(output_image_np, output_mask_np, border_ratio=0.227) 
            output_image_torch = torch.from_numpy(output_image_np_resized.astype(np.float32) / 255) 

            output_image_torch = output_image_torch.permute(2, 0, 1)
            output_mask_torch = output_image_torch[3:4]
            output_image_torch = output_image_torch[:3] * output_mask_torch + (1 - output_mask_torch)

        return output_image_torch, output_mask_torch

    def __getitem__(self, idx):

        assert self.num_input_views == 4, "Only 4 input views are supported for this dataset"
        
        subject_name = self.items[idx].split('/')[-1].split('.')[0]
        results = {}

        context_image_torch, context_mask_torch = self.preprocess(self.items[idx])

        exp_folder_structure_path = os.path.join(os.getcwd(), 'assets/cam_poses')
        rendering_path = exp_folder_structure_path
        
        imagedream_images = []
        imagedream_masks = []
        imagedream_cam_poses = []

        first_gt_image_idx = np.random.permutation(np.arange(8, 9))[:1]
        second_gt_image_idx = (first_gt_image_idx + 8) % 32
        third_gt_image_idx = (first_gt_image_idx + 16) % 32
        fourth_gt_image_idx = (first_gt_image_idx + 24) % 32
        imagedream_image_idx = np.concatenate([first_gt_image_idx+100, second_gt_image_idx+100, third_gt_image_idx+100, fourth_gt_image_idx+100])
        
        for imagedream_idx in imagedream_image_idx:

            imagedream_c2w_path = os.path.join(rendering_path, '132_{}_RT.txt'.format(imagedream_idx))
            imagedream_c2w_pose = torch.tensor(np.loadtxt(imagedream_c2w_path)).float().reshape(4, 4)
            imagedream_cam_poses.append(imagedream_c2w_pose)

        imagedream_cam_poses = torch.stack(imagedream_cam_poses, dim=0)
        imagedream_cam_poses_opengl = self.blender2opengl.unsqueeze(0)@imagedream_cam_poses 

        imagedream_translation = imagedream_cam_poses[:, :3, 3]
        imagedream_translation = imagedream_translation / imagedream_translation.norm(dim=-1, keepdim=True)
        imagedream_cam_poses[:, :3, 3] = imagedream_translation 

        results['imagedream_cam_poses_gt'] = imagedream_cam_poses 

        diffusion3d_cam_poses = []
        vids = np.arange(100, 132).tolist()
        for vid in vids: 
            diffusion3d_c2w_path = os.path.join(rendering_path, '132_{}_RT.txt'.format(vid))
            diffusion3d_c2w = torch.tensor(np.loadtxt(diffusion3d_c2w_path)).float().reshape(4, 4)
            diffusion3d_cam_poses.append(diffusion3d_c2w)

        diffusion3d_cam_poses = torch.stack(diffusion3d_cam_poses, dim=0) 
        diffusion3d_cam_poses_opengl = self.blender2opengl.unsqueeze(0)@diffusion3d_cam_poses
        diffusion3d_all_cam_poses_opengl = torch.cat([imagedream_cam_poses_opengl, diffusion3d_cam_poses_opengl], dim=0)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(diffusion3d_all_cam_poses_opengl[0])
        diffusion3d_all_cam_poses = transform.unsqueeze(0) @ diffusion3d_all_cam_poses_opengl  

        diffusion3d_cam_poses_input = diffusion3d_all_cam_poses[:self.num_input_views].clone()

        rays_embeddings = []
        for i in range(self.num_input_views):
            rays_o, rays_d = get_rays(diffusion3d_cam_poses_input[i], self.input_size, self.input_size, self.fovy) 
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) 
            rays_embeddings.append(rays_plucker)
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() 

        results['diffusion3d_cam_poses_input_embedding'] = rays_embeddings 

        print("load context image, done remove background, recentered, make white background")
        context_image = context_image_torch

        context_image_input = F.interpolate(context_image.unsqueeze(0), size=(self.input_size, self.input_size), mode='bilinear', align_corners=False) 
        context_image_input = TF.normalize(context_image_input, self.normalize_mean, self.normalize_std) 
        context_cam_pos = torch.zeros_like(diffusion3d_cam_poses[0])
        context_ray_o, context_ray_d = get_rays(context_cam_pos, self.input_size, self.input_size, self.fovy)
        context_ray_embedding = torch.cat([torch.cross(context_ray_o, context_ray_d, dim=-1), context_ray_d], dim=-1).unsqueeze(0).permute(0, 3, 1, 2).contiguous() 

        results['context_image'] = context_image_input
        results['context_ray_embedding'] = context_ray_embedding 
        
        diffusion3d_all_cam_poses[:, :3, 1:3] *= -1 

        cam_view = torch.inverse(diffusion3d_all_cam_poses).transpose(1, 2) 
        cam_view_proj = cam_view @ self.proj_matrix 
        cam_pos = - diffusion3d_all_cam_poses[:, :3, 3] 

        results['cam_view_imagedream'] = cam_view[:self.num_input_views]
        results['cam_view_proj_imagedream'] = cam_view_proj[:self.num_input_views]
        results['cam_pos_imagedream'] = cam_pos[:self.num_input_views]

        results['cam_view'] = cam_view[self.num_input_views:] 
        results['cam_view_proj'] = cam_view_proj[self.num_input_views:]
        results['cam_pos'] = cam_pos[self.num_input_views:]

        results['dataset'] = self.dataset_name
        results['subject_name'] = subject_name

        return results