import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from kiui.op import safe_normalize

import matplotlib.pyplot as plt
from matplotlib import cm

def get_rays(pose, h, w, fovy, opengl=True):

    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  
    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) 

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d


def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(torch.abs(gt_image[:, :, :, 1:-1, 1:-1] - gt_image[:, :, :, 1:-1, :-2]), 2) 
    grad_img_right = torch.mean(torch.abs(gt_image[:, :, :, 1:-1, 1:-1] - gt_image[:, :, :, 1:-1, 2:]), 2)
    grad_img_top = torch.mean(torch.abs(gt_image[:, :, :, 1:-1, 1:-1] - gt_image[:, :, :, :-2, 1:-1]), 2)
    grad_img_bottom = torch.mean(torch.abs(gt_image[:, :, :, 1:-1, 1:-1] - gt_image[:, :, :, 2:, 1:-1]), 2)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0] 

    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    max_grad = max_grad.unsqueeze(2) 
    return distortion_map * max_grad

def depths_to_points(c2w, opt, depthmap):
    B = c2w.shape[0]
    V = c2w.shape[1]

    W, H = opt.output_size, opt.output_size
    fx = W / (2 * math.tan(np.deg2rad(opt.fovy) / 2.))
    fy = H / (2 * math.tan(np.deg2rad(opt.fovy) / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float().cuda() 
    
    points_all = points[None, :].repeat(B, V, 1, 1)
    intrinsics_inv_all = intrins.inverse().T[None, None, :].repeat(B, V, 1, 1) 
    c2w_all = c2w[:, :, :3, :3].permute(0, 1, 3, 2).contiguous() 
    
    rays_d = torch.einsum('bvni, bvij -> bvnj', torch.einsum('bvni, bvij -> bvnj', points_all, intrinsics_inv_all), c2w_all)
    rays_o = c2w[:, :, :3, 3].unsqueeze(2).repeat(1, 1, rays_d.shape[2], 1)

    points = depthmap.reshape(B, V, -1, 1) * rays_d + rays_o
    return points.reshape(B, V, opt.output_size, opt.output_size, 3)


def depth_to_normal(c2w, opt, depth):
    B = c2w.shape[0]
    V = c2w.shape[1]
    points = depths_to_points(c2w, opt, depth) 
    output = torch.zeros_like(points)
    dx = torch.cat([points[:, :, 2:, 1:-1, :] - points[:, :, :-2, 1:-1, :]], dim=2) 
    dy = torch.cat([points[:, :, 1:-1, 2:, :] - points[:, :, 1:-1, :-2, :]], dim=3)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1) 

    output[:, :, 1:-1, 1:-1, :] = normal_map 
    return output, points


def colormap(images, cmap='jet'):
    V, W, H = images.shape[:3]
    images_processed = []
    for view in range(V):
        img = images[view]
        dpi = 300
        fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
        im = ax.imshow(img, cmap=cmap)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = torch.from_numpy(data / 255.).float().permute(2,0,1)
        plt.close()
        if img.shape[1:] != (H, W):
            img = torch.nn.functional.interpolate(img[None], (W, H), mode='bilinear', align_corners=False)[0]
        images_processed.append(img)
    return torch.stack(images_processed, dim=0)

def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device) 
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth_,
    accumulation_,
    near_plane = 0.5,
    far_plane = 2.5,
    cmap="turbo",
):
    depth_ = torch.from_numpy(depth_).float() if isinstance(depth_, np.ndarray) else depth_
    accumulation_ = torch.from_numpy(accumulation_).float() if isinstance(accumulation_, np.ndarray) else accumulation_

    vis_images = []
    for view in range(depth_.shape[0]):
        depth = depth_[view]
        near_plane = near_plane or float(torch.min(depth))
        far_plane = far_plane or float(torch.max(depth))

        depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
        depth = torch.clip(depth, 0, 1)

        colored_image = apply_colormap(depth, cmap=cmap)

        if accumulation_ is not None:
            accumulation = accumulation_[view]
            colored_image = colored_image * accumulation + (1 - accumulation)

        vis_images.append(colored_image)

    return torch.stack(vis_images, dim=0)