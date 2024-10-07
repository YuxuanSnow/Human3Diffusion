import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import kiui
from kiui.lpips import LPIPS

from core.unet_timeImage_cond import UNet_timeimage_cond
from core.options import Options
from core.gof import GaussianRenderer

from core.utils import get_edge_aware_distortion_map, depth_to_normal

class diffusion3dgs_noise_gof(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        self.unet = UNet_timeimage_cond(
            12, 14, 
            down_channels=self.opt.down_channels,                 
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
        )

        self.conv = nn.Conv2d(14, 14, kernel_size=1) 

        self.gof = GaussianRenderer(opt)

        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5

        self.lpips_loss = LPIPS(net='vgg')
        self.lpips_loss.requires_grad_(False)

    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) 
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) 
        
        return rays_embeddings
        

    def forward_gaussians(self, images, timestep):

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images, timestep) 
        x = self.conv(x) 

        x = x.reshape(B, 5, 14, self.opt.splat_size, self.opt.splat_size)[:, :4, :, :, :] 
        
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., 0:3])
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) 
        
        return gaussians

    
    def forward(self, data, timestep, step_ratio=1):
        results = {}
        loss = 0

        images = data['input'] 

        gaussians = self.forward_gaussians(images, timestep)
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        results = self.gof.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        
        results['gaussians'] = gaussians
        pred_images = results['image']
        pred_alphas = results['mask']
        pred_normals = results['normal']
        pred_depths = results['depth']
        pred_distortions = results['distortion']


        gt_images = data['images_output'] 
        gt_masks = data['masks_output'] 
        gt_depth = data['depths_output'].to(pred_depths.device)

        gt_normal_cam = data['normals_output'].to(pred_normals.device)
        gt_normal_cam[:,:,1:3] = -gt_normal_cam[:,:,1:3] 

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
        gt_depth = gt_depth * gt_masks + torch.zeros_like(gt_depth) * (1 - gt_masks)
        gt_normal_cam = gt_normal_cam * gt_masks + torch.zeros_like(gt_normal_cam) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        results['loss_rgb'] = loss_mse
        loss = loss + loss_mse

        if self.opt.lambda_distortion > 0:
            distortion_map = pred_distortions 
            distortion_map = get_edge_aware_distortion_map(gt_images, distortion_map)
            distortion_loss = distortion_map.mean()
            results['loss_distortion'] = distortion_loss    
            loss = loss + self.opt.lambda_distortion * distortion_loss

        if self.opt.lambda_normal > 0:

            c2w = torch.linalg.inv(data['cam_view'].permute(0, 1, 3, 2).contiguous()) 
            B = c2w.shape[0]
            V = c2w.shape[1]

            depth_normal_world, _ = depth_to_normal(c2w, self.opt, pred_depths)
            depth_normal_world = depth_normal_world.permute(0, 1, 4, 2, 3) 

            gt_depth_normal_world, gt_depth_points_normal_world = depth_to_normal(c2w, self.opt, gt_depth)
            gt_depth_normal_world = gt_depth_normal_world.permute(0, 1, 4, 2, 3) 

            render_normal_cam = pred_normals 
            render_normal_cam = torch.nn.functional.normalize(render_normal_cam, p=2, dim=2)
            
            gt_normal_world = torch.einsum('bvij, bvjn -> bvin', c2w[:, :, :3, :3], gt_normal_cam.reshape(B, V, 3, -1)).reshape(B, V, 3, self.opt.output_size, self.opt.output_size)
            gt_normal_world = torch.nn.functional.normalize(gt_normal_world, p=2, dim=2)
            
            results['depth_normal_world'] = depth_normal_world
            results['gt_normal_world'] = gt_normal_world
            results['gt_depth_normal_world'] = gt_depth_normal_world
            results['gt_depth_points_normal_world'] = gt_depth_points_normal_world
            
            normal_error_gt_world = gt_masks * (1 - (depth_normal_world * gt_normal_world).sum(dim=2).unsqueeze(dim=2)) 
            depth_normal_loss = normal_error_gt_world.mean() 

            results['normal_per_pixel_error_world'] = normal_error_gt_world
            
            results['loss_normal_world'] = normal_error_gt_world.mean()
            loss = loss + self.opt.lambda_normal * depth_normal_loss

        if self.opt.lambda_depth > 0:
            pred_depths = pred_depths * gt_masks
            loss_depth = F.l1_loss(pred_depths, gt_depth)
            results['loss_depth'] = loss_depth
            loss = loss + self.opt.lambda_depth * loss_depth

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results
    