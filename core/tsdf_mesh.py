###############################
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

import kiui
import functools

import torchvision.transforms as transforms
from PIL import Image
import os

class GaussianRenderer:
    def __init__(self):
        
        self.output_size = 512
        self.fovy = 49.1
        self.znear = 0.5
        self.zfar = 2.5

        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.zfar + self.znear) / (self.zfar - self.znear)
        self.proj_matrix[3, 2] = - (self.zfar * self.znear) / (self.zfar - self.znear)
        self.proj_matrix[2, 3] = 1
        
    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, sub_pixel_offset=None, scale_modifier=1.0):
        device = gaussians.device
        B, V = cam_view.shape[:2]

        images = []
        normals = []
        depths = []
        masks = []
        distortions = []
      
        for b in range(B):

            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() 

            for v in range(V):
                
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.output_size,
                    image_width=self.output_size,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    kernel_size=0.0,
                    subpixel_offset= torch.Tensor([]) if sub_pixel_offset is None else None,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                rendered_image, radii = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                rendered_rgb = rendered_image[:3, :, :].clamp(0, 1)
                rendered_normal = rendered_image[3:6, :, :]
                rendered_depth = rendered_image[6:7, :, :]
                rendered_mask = rendered_image[7:8, :, :]
                rendered_distortion = rendered_image[8:9, :, :]

                rendered_rgb = rendered_rgb.clamp(0, 1)

                images.append(rendered_rgb)
                normals.append(rendered_normal)
                depths.append(rendered_depth)
                masks.append(rendered_mask)
                distortions.append(rendered_distortion)

        images = torch.stack(images, dim=0).view(B, V, 3, self.output_size, self.output_size)
        normals = torch.stack(normals, dim=0).view(B, V, 3, self.output_size, self.output_size)
        depths = torch.stack(depths, dim=0).view(B, V, 1, self.output_size, self.output_size)
        masks = torch.stack(masks, dim=0).view(B, V, 1, self.output_size, self.output_size)
        distortions = torch.stack(distortions, dim=0).view(B, V, 1, self.output_size, self.output_size)

        return {
            "image": images,
            "normal": normals,
            "depth": depths, 
            "mask": masks,
            "distortion": distortions 
        }

    def integrate(self, points3D, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, sub_pixel_offset=None, scale_modifier=1.0):

        device = gaussians.device
        B, V = cam_view.shape[:2]

        images = []
        normals = []
        depths = []
        masks = []
        distortions = []

        alpha_integrated_all = []
        color_integrated_all = []

        for b in range(B):

            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() 

            for v in range(V):
                
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.output_size,
                    image_width=self.output_size,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    kernel_size=0.0,
                    subpixel_offset= torch.Tensor([]) if sub_pixel_offset is None else None,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                rendered_image, alpha_integrated, color_integrated, radii = rasterizer.integrate(
                    points3D = points3D,
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                rendered_rgb = rendered_image[:3, :, :].clamp(0, 1)
                rendered_normal = rendered_image[3:6, :, :]
                rendered_depth = rendered_image[6:7, :, :]
                rendered_mask = rendered_image[7:8, :, :]
                rendered_distortion = rendered_image[8:9, :, :]

                rendered_rgb = rendered_rgb.clamp(0, 1)

                images.append(rendered_rgb)
                normals.append(rendered_normal)
                depths.append(rendered_depth)
                masks.append(rendered_mask)
                distortions.append(rendered_distortion)

                alpha_integrated_all.append(alpha_integrated)
                color_integrated_all.append(color_integrated)

        images = torch.stack(images, dim=0).view(B, V, 3, self.output_size, self.output_size)
        normals = torch.stack(normals, dim=0).view(B, V, 3, self.output_size, self.output_size)
        depths = torch.stack(depths, dim=0).view(B, V, 1, self.output_size, self.output_size)
        masks = torch.stack(masks, dim=0).view(B, V, 1, self.output_size, self.output_size)
        distortions = torch.stack(distortions, dim=0).view(B, V, 1, self.output_size, self.output_size)

        alpha_integrated_all = torch.stack(alpha_integrated_all, dim=0)
        color_integrated_all = torch.stack(color_integrated_all, dim=0)

        return {
            "image": images, 
            "normal": normals,
            "depth": depths, 
            "mask": masks, 
            "distortion": distortions,
            'alpha_integrated': alpha_integrated_all, 
            'color_integrated': color_integrated_all,
        }
    

    @torch.no_grad()
    def evaluate_alpha(self, points, gaussians, cam_view, cam_view_proj, cam_pos):
        final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) 

        with torch.no_grad():
            ret = self.integrate(points, gaussians, cam_view, cam_view_proj, cam_pos, bg_color)
            alpha_integrated = ret["alpha_integrated"]
            alpha_integrated = alpha_integrated.squeeze(0) 
            for viewidx in range(alpha_integrated.shape[0]):
                alpha_integrated_view = alpha_integrated[viewidx] 
                final_alpha = torch.min(final_alpha, alpha_integrated_view) 
            alpha = 1 - final_alpha
        return alpha

    @staticmethod
    def alpha_to_sdf(alpha):    
        sdf = alpha - 0.5
        sdf = sdf[None]
        return sdf

    @torch.no_grad()
    def extract_mesh(self, gaussians, cam_view, cam_view_proj, cam_pos):

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from core.gof_extract_mesh import get_tetra_points
        from core.tetmesh import marching_tetrahedra
        from tetranerf.utils.extension import cpp

        gaussians_rotation = gaussians[0, :, 7:11]
        gaussians_xyz = gaussians[0, :, 0:3]
        gaussians_scaling = gaussians[0, :, 4:7]
        
        points, points_scale = get_tetra_points(gaussians_rotation, gaussians_xyz, gaussians_scaling) 
        cells = cpp.triangulate(points)                                                          
        
        alpha = self.evaluate_alpha(points, gaussians, cam_view, cam_view_proj, cam_pos)

        vertices = points.cuda()[None] 
        tets = cells.cuda().long()

        print(vertices.shape, tets.shape, alpha.shape) 

        sdf = self.alpha_to_sdf(alpha)

        torch.cuda.empty_cache()
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
        torch.cuda.empty_cache()

        end_points, end_sdf = verts_list[0]
        end_scales = scale_list[0]
        
        faces=faces_list[0].cpu().numpy()
        points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
            
        left_points = end_points[:, 0, :]
        right_points = end_points[:, 1, :]
        left_sdf = end_sdf[:, 0, :]
        right_sdf = end_sdf[:, 1, :]
        left_scale = end_scales[:, 0, 0]
        right_scale = end_scales[:, 1, 0]
        distance = torch.norm(left_points - right_points, dim=-1)
        scale = left_scale + right_scale

        n_binary_steps = 8
        for step in range(n_binary_steps):
            print("binary search in step {}".format(step))
            mid_points = (left_points + right_points) / 2
            alpha = self.evaluate_alpha(mid_points, gaussians, cam_view, cam_view_proj, cam_pos)
            mid_sdf = self.alpha_to_sdf(alpha).squeeze().unsqueeze(-1)

            ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

            left_sdf[ind_low] = mid_sdf[ind_low]
            right_sdf[~ind_low] = mid_sdf[~ind_low]
            left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
            right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        
            points = (left_points + right_points) / 2
            if step not in [n_binary_steps-1]:
                continue

        return points, faces
    

    def save_ply(self, gaussians, path, compatible=True):

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() 

        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() 

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect', last_op=nn.Tanh()):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if last_op is not None:
            model += [last_op]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             
        
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_normal_estimator(state_dict_path):

    to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    print('Resuming from ', state_dict_path)
    ckpt = torch.load(state_dict_path, map_location='cpu')    

    netG = GlobalGenerator(input_nc=3, output_nc=3, ngf=64, n_downsampling=4, n_blocks=9, last_op=nn.Tanh())  
    netG = netG.cuda()

    state_dict = netG.state_dict()

    for k, v in ckpt['model_state_dict'].items():
        if k[10:] in state_dict: 
            if k[:10] == 'netG.netF.':
                if state_dict[k[10:]].shape == v.shape: 
                    state_dict[k[10:]].copy_(v)
                else:
                    pass
        else:
            pass

    return netG, to_tensor

from scipy.sparse import spdiags, csr_matrix, vstack
from scipy.sparse.linalg import cg
import numpy as np
from tqdm.auto import tqdm
import time
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def move_left(mask): return np.pad(mask,((0,0),(0,1)),'constant',constant_values=0)[:,1:]  
def move_right(mask): return np.pad(mask,((0,0),(1,0)),'constant',constant_values=0)[:,:-1]  
def move_top(mask): return np.pad(mask,((0,1),(0,0)),'constant',constant_values=0)[1:,:] 
def move_bottom(mask): return np.pad(mask,((1,0),(0,0)),'constant',constant_values=0)[:-1,:] 
def move_top_left(mask): return np.pad(mask,((0,1),(0,1)),'constant',constant_values=0)[1:,1:]  
def move_top_right(mask): return np.pad(mask,((0,1),(1,0)),'constant',constant_values=0)[1:,:-1]  
def move_bottom_left(mask): return np.pad(mask,((1,0),(0,1)),'constant',constant_values=0)[:-1,1:]  
def move_bottom_right(mask): return np.pad(mask,((1,0),(1,0)),'constant',constant_values=0)[:-1,:-1]  

def generate_dx_dy(mask, nz_horizontal, nz_vertical, step_size=1):
    num_pixel = np.sum(mask)

    pixel_idx = np.zeros_like(mask, dtype=int)
    pixel_idx[mask] = np.arange(num_pixel)

    has_left_mask = np.logical_and(move_right(mask), mask)
    has_right_mask = np.logical_and(move_left(mask), mask)
    has_bottom_mask = np.logical_and(move_top(mask), mask)
    has_top_mask = np.logical_and(move_bottom(mask), mask)

    nz_left = nz_horizontal[has_left_mask[mask]]
    nz_right = nz_horizontal[has_right_mask[mask]]
    nz_top = nz_vertical[has_top_mask[mask]]
    nz_bottom = nz_vertical[has_bottom_mask[mask]]

    data = np.stack([-nz_left/step_size, nz_left/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_right/step_size, nz_right/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_top/step_size, nz_top/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_bottom/step_size, nz_bottom/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg

def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-k * x))

def bilateral_normal_integration_depth(normal_map,
                                        depth_map,
                                        mask,
                                        lambda1=1e4,
                                        k=2,
                                        step_size=1,
                                        max_iter=150,
                                        tol=1e-4,
                                        cg_max_iter=5000,
                                        cg_tol=1e-3):
    
    K = np.array([
            [5.604441528320312500e+02, 0.000000000000000000e+00, 2.560000000000000000e+02],
            [0.000000000000000000e+00, 5.604441528320312500e+02, 2.560000000000000000e+02],
            [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
    normal_mask = mask
    depth_mask = mask

    num_normals = np.sum(normal_mask)

    nx = normal_map[normal_mask, 1]
    ny = normal_map[normal_mask, 0]
    nz = - normal_map[normal_mask, 2]

    img_height, img_width = normal_mask.shape[:2]

    yy, xx = np.meshgrid(range(img_width), range(img_height))
    xx = np.flip(xx, axis=0)

    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    uu = xx[normal_mask] - cx
    vv = yy[normal_mask] - cy

    nz_u = uu * nx + vv * ny + fx * nz
    nz_v = uu * nx + vv * ny + fy * nz
    del xx, yy, uu, vv

    A3, A4, A1, A2 = generate_dx_dy(normal_mask, nz_horizontal=nz_v, nz_vertical=nz_u, step_size=step_size)

    A = vstack((A1, A2, A3, A4))
    b = np.concatenate((-nx, -nx, -ny, -ny))

    W = spdiags(0.5 * np.ones(4*num_normals), 0, 4*num_normals, 4*num_normals, format="csr")
    z = np.zeros(np.sum(normal_mask))
    energy = (A @ z - b).T @ W @ (A @ z - b)

    tic = time.time()

    energy_list = []

    m = depth_mask[normal_mask].astype(int)
    M = spdiags(m, 0, num_normals, num_normals, format="csr")
    z_prior = np.log(depth_map)[normal_mask] if K is not None else depth_map[normal_mask]

    for i in range(max_iter):
        A_mat = A.T @ W @ A
        b_vec = A.T @ W @ b
        if depth_map is not None:
            depth_diff = M @ (z_prior - z)
            depth_diff[depth_diff==0] = np.nan
            offset = np.nanmean(depth_diff)
            z = z + offset
            A_mat += lambda1 * M
            b_vec += lambda1 * M @ z_prior

        D = spdiags(1/np.clip(A_mat.diagonal(), 1e-5, None), 0, num_normals, num_normals, format="csr")  

        z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=cg_max_iter, tol=cg_tol)

        wu = sigmoid((A2 @ z) ** 2 - (A1 @ z) ** 2, k)
        wv = sigmoid((A4 @ z) ** 2 - (A3 @ z) ** 2, k)
        W = spdiags(np.concatenate((wu, 1-wu, wv, 1-wv)), 0, 4*num_normals, 4*num_normals, format="csr")

        energy_old = energy
        energy = (A @ z - b).T @ W @ (A @ z - b)
        energy_list.append(energy)
        relative_energy = np.abs(energy - energy_old) / energy_old
        if relative_energy < tol:
            break
    
    depth_map_refined = np.ones_like(normal_mask, float) * np.nan
    depth_map_refined[normal_mask] = z

    return np.exp(depth_map_refined)


from glob import glob

from numba import njit, prange
from skimage import measure

import pycuda.driver as cuda
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule
FUSION_GPU_MODE = 1

class TSDFVolume:
  def __init__(self, vol_bnds, voxel_size, use_gpu=True):
    vol_bnds = np.asarray(vol_bnds)
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    self._vol_bnds = vol_bnds
    self._voxel_size = float(voxel_size)
    self._trunc_margin = 5 * self._voxel_size  
    self._color_const = 256 * 256

    self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
    self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
    self._vol_origin = self._vol_bnds[:,0].copy(order='C').astype(np.float32)

    print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
      self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
      self._vol_dim[0]*self._vol_dim[1]*self._vol_dim[2])
    )

    self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
    self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

    self.gpu_mode = use_gpu and FUSION_GPU_MODE

    if self.gpu_mode:
      self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
      cuda.memcpy_htod(self._tsdf_vol_gpu,self._tsdf_vol_cpu)
      self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
      cuda.memcpy_htod(self._weight_vol_gpu,self._weight_vol_cpu)
      self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
      cuda.memcpy_htod(self._color_vol_gpu,self._color_vol_cpu)

      self._cuda_src_mod = SourceModule("""
        __global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_intr,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im) {
          // Get voxel index
          int gpu_loop_idx = (int) other_params[0];
          int max_threads_per_block = blockDim.x;
          int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
          int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
          int vol_dim_x = (int) vol_dim[0];
          int vol_dim_y = (int) vol_dim[1];
          int vol_dim_z = (int) vol_dim[2];
          if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
              return;
          // Get voxel grid coordinates (note: be careful when casting)
          float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
          float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
          float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
          // Voxel grid coordinates to world coordinates
          float voxel_size = other_params[1];
          float pt_x = vol_origin[0]+voxel_x*voxel_size;
          float pt_y = vol_origin[1]+voxel_y*voxel_size;
          float pt_z = vol_origin[2]+voxel_z*voxel_size;
          // World coordinates to camera coordinates
          float tmp_pt_x = pt_x-cam_pose[0*4+3];
          float tmp_pt_y = pt_y-cam_pose[1*4+3];
          float tmp_pt_z = pt_z-cam_pose[2*4+3];
          float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
          float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
          float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
          // Camera coordinates to image pixels
          int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
          int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
          // Skip if outside view frustum
          int im_h = (int) other_params[2];
          int im_w = (int) other_params[3];
          if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
              return;
          // Skip invalid depth
          float depth_value = depth_im[pixel_y*im_w+pixel_x];
          if (depth_value == 0)
              return;
          // Integrate TSDF
          float trunc_margin = other_params[4];
          float depth_diff = depth_value-cam_pt_z;
          if (depth_diff < -trunc_margin)
              return;
          float dist = fmin(1.0f,depth_diff/trunc_margin);
          float w_old = weight_vol[voxel_idx];
          float obs_weight = other_params[5];
          float w_new = w_old + obs_weight;
          weight_vol[voxel_idx] = w_new;
          tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
          // Integrate color
          float old_color = color_vol[voxel_idx];
          float old_b = floorf(old_color/(256*256));
          float old_g = floorf((old_color-old_b*256*256)/256);
          float old_r = old_color-old_b*256*256-old_g*256;
          float new_color = color_im[pixel_y*im_w+pixel_x];
          float new_b = floorf(new_color/(256*256));
          float new_g = floorf((new_color-new_b*256*256)/256);
          float new_r = new_color-new_b*256*256-new_g*256;
          new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
          new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
          new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
          color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
        }""")

      self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

      gpu_dev = cuda.Device(0)
      self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
      n_blocks = int(np.ceil(float(np.prod(self._vol_dim))/float(self._max_gpu_threads_per_block)))
      grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,int(np.floor(np.cbrt(n_blocks))))
      grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y,int(np.floor(np.sqrt(n_blocks/grid_dim_x))))
      grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z,int(np.ceil(float(n_blocks)/float(grid_dim_x*grid_dim_y))))
      self._max_gpu_grid_dim = np.array([grid_dim_x,grid_dim_y,grid_dim_z]).astype(int)
      self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim))/float(np.prod(self._max_gpu_grid_dim)*self._max_gpu_threads_per_block)))

    else:
      xv, yv, zv = np.meshgrid(
        range(self._vol_dim[0]),
        range(self._vol_dim[1]),
        range(self._vol_dim[2]),
        indexing='ij'
      )
      self.vox_coords = np.concatenate([
        xv.reshape(1,-1),
        yv.reshape(1,-1),
        zv.reshape(1,-1)
      ], axis=0).astype(int).T

  @staticmethod
  @njit(parallel=True)
  def vox2world(vol_origin, vox_coords, vox_size):
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)
    for i in prange(vox_coords.shape[0]):
      for j in range(3):
        cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
    return cam_pts

  @staticmethod
  @njit(parallel=True)
  def cam2pix(cam_pts, intr):
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    for i in prange(cam_pts.shape[0]):
      pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
      pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
    return pix

  @staticmethod
  @njit(parallel=True)
  def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
    tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
    w_new = np.empty_like(w_old, dtype=np.float32)
    for i in prange(len(tsdf_vol)):
      w_new[i] = w_old[i] + obs_weight
      tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
    return tsdf_vol_int, w_new

  def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
    im_h, im_w = depth_im.shape

    color_im = color_im.astype(np.float32)
    color_im = np.floor(color_im[...,2] * 256 * 256 + color_im[...,1] * 256 + color_im[...,0])

    if self.gpu_mode: 
      for gpu_loop_idx in range(self._n_gpu_loops):
        self._cuda_integrate(self._tsdf_vol_gpu,
                            self._weight_vol_gpu,
                            self._color_vol_gpu,
                            cuda.InOut(self._vol_dim.astype(np.float32)),
                            cuda.InOut(self._vol_origin.astype(np.float32)),
                            cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                            cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                            cuda.InOut(np.asarray([
                              gpu_loop_idx,
                              self._voxel_size,
                              im_h,
                              im_w,
                              self._trunc_margin,
                              obs_weight
                            ], np.float32)),
                            cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                            cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                            block=(self._max_gpu_threads_per_block,1,1),
                            grid=(
                              int(self._max_gpu_grid_dim[0]),
                              int(self._max_gpu_grid_dim[1]),
                              int(self._max_gpu_grid_dim[2]),
                            )
        )
    else: 
      cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
      cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
      pix_z = cam_pts[:, 2]
      pix = self.cam2pix(cam_pts, cam_intr)
      pix_x, pix_y = pix[:, 0], pix[:, 1]

      valid_pix = np.logical_and(pix_x >= 0,
                  np.logical_and(pix_x < im_w,
                  np.logical_and(pix_y >= 0,
                  np.logical_and(pix_y < im_h,
                  pix_z > 0))))
      depth_val = np.zeros(pix_x.shape)
      depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

      depth_diff = depth_val - pix_z
      valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
      dist = np.minimum(1, depth_diff / self._trunc_margin)
      valid_vox_x = self.vox_coords[valid_pts, 0]
      valid_vox_y = self.vox_coords[valid_pts, 1]
      valid_vox_z = self.vox_coords[valid_pts, 2]
      w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      valid_dist = dist[valid_pts]
      tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
      self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
      self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

      old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      old_b = np.floor(old_color / self._color_const)
      old_g = np.floor((old_color-old_b*self._color_const)/256)
      old_r = old_color - old_b*self._color_const - old_g*256
      new_color = color_im[pix_y[valid_pts],pix_x[valid_pts]]
      new_b = np.floor(new_color / self._color_const)
      new_g = np.floor((new_color - new_b*self._color_const) /256)
      new_r = new_color - new_b*self._color_const - new_g*256
      new_b = np.minimum(255., np.round((w_old*old_b + obs_weight*new_b) / w_new))
      new_g = np.minimum(255., np.round((w_old*old_g + obs_weight*new_g) / w_new))
      new_r = np.minimum(255., np.round((w_old*old_r + obs_weight*new_r) / w_new))
      self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*self._color_const + new_g*256 + new_r

  def get_volume(self):
    if self.gpu_mode:
      cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
      cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
    return self._tsdf_vol_cpu, self._color_vol_cpu

  def get_point_cloud(self):
    """Extract a point cloud from the voxel volume.
    """
    tsdf_vol, color_vol = self.get_volume()

    verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + self._vol_origin

    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._color_const)
    colors_g = np.floor((rgb_vals - colors_b*self._color_const) / 256)
    colors_r = rgb_vals - colors_b*self._color_const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    pc = np.hstack([verts, colors])
    return pc

  def get_mesh(self):
    tsdf_vol, color_vol = self.get_volume()

    verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size+self._vol_origin  

    rgb_vals = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
    colors_b = np.floor(rgb_vals/self._color_const)
    colors_g = np.floor((rgb_vals-colors_b*self._color_const)/256)
    colors_r = rgb_vals-colors_b*self._color_const-colors_g*256
    colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
    colors = colors.astype(np.uint8)
    return verts, faces, norms, colors


def rigid_transform(xyz, transform):
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]

roty = np.array([
    [0, 0, 1, 0],
    [0, 1, 0., 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1]
])
flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
blender2opencv = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0.],
    [0, 0, -1., 0],
    [0, 0, 0, 1]
])


def create_camera_to_world_matrix(elevation, azimuth, radius=1.0):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    x = np.cos(elevation) * np.sin(azimuth) * radius
    y = np.sin(elevation) * radius
    z = np.cos(elevation) * np.cos(azimuth) * radius

    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    cam2world = np.eye(4)
    cam2world[:3, :3] = np.array([right, new_up, -forward]).T
    cam2world[:3, 3] = camera_pos
    return cam2world

def convert_opengl_to_blender(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        camera_matrix_blender = np.dot(flip_yz, camera_matrix)
    return camera_matrix_blender

def generate_tsdf_mesh(subj_base_path, normal_checkpoint_path, quality='high'):

    if not os.path.exists(subj_base_path):
        os.makedirs(subj_base_path)

    netG, to_tensor = get_normal_estimator(normal_checkpoint_path)
    if quality == 'high':
        voxel_size, obs_weight = 0.001, 1.0 
    elif quality == 'low':
        voxel_size, obs_weight = 0.003, 1.0

    vox_bnds = np.array([[-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6]])
    tsdf_vol = TSDFVolume(vox_bnds, voxel_size=voxel_size)

    gof = GaussianRenderer()
    proj_matrix = gof.proj_matrix.cuda()
    gaussians = gof.load_ply(subj_base_path + '/gs.ply').unsqueeze(0).cuda()
    bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

    blender2opengl = torch.tensor([[1, 0, 0, 0], 
                                    [0, 0, 1, 0], 
                                    [0, -1, 0, 0], 
                                    [0, 0, 0, 1]]).cuda().float()

    K = np.array([5.604441528320312500e+02, 0.000000000000000000e+00, 2.560000000000000000e+02,
                    0.000000000000000000e+00, 5.604441528320312500e+02, 2.560000000000000000e+02,
                    0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]).reshape((3, 3)).astype(np.float32)
    front_view_c2w = np.array([6.123233995736766036e-17, 0.000000000000000000e+00, 1.000000000000000000e+00, 1.500000000000000000e+00,
                                1.000000000000000000e+00, 0.000000000000000000e+00, -6.123233995736766036e-17, -9.184850993605148438e-17,
                                0.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00,
                                0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]).reshape((4, 4)).astype(np.float32)

    front_view_c2w_opengl = blender2opengl@torch.from_numpy(front_view_c2w).cuda() 
    transform_rel_front = torch.tensor([[1, 0, 0, 0], 
                                        [0, 1, 0, 0], 
                                        [0, 0, 1, 1.5], 
                                        [0, 0, 0, 1]], dtype=torch.float32).cuda() @ torch.inverse(front_view_c2w_opengl)
    
    if quality == 'high':
        view_total = 72
    elif quality == 'low':
        view_total = 24
    ELEVATION = np.arange(-45, 45, 90/view_total)
    AZIMUTH = np.arange(0*3, 360*3, 360*3/view_total)
    from tqdm import tqdm

    for view_idx in tqdm(range(view_total)):

        view_base_path = os.path.join(subj_base_path, f'{view_idx}_tsdf')
        elevation = ELEVATION[view_idx]
        azimuth = AZIMUTH[view_idx]
        
        camera_matrix = create_camera_to_world_matrix(elevation, azimuth, 1.5)
        c2w_orig = convert_opengl_to_blender(camera_matrix)

        c2w_opengl = blender2opengl@torch.from_numpy(c2w_orig).cuda().float()
        c2w_opengl = transform_rel_front@c2w_opengl
        c2w_opengl[:3, 1:3] *= -1
        cam_view_gof = torch.inverse(c2w_opengl).unsqueeze(0).transpose(1, 2).unsqueeze(0).cuda() 
        cam_view_proj_gof = (cam_view_gof @ proj_matrix).cuda()
        cam_pos_gof = c2w_opengl[:3, 3].unsqueeze(0).unsqueeze(0).cuda()

        render = gof.render(gaussians, cam_view_gof, cam_view_proj_gof, cam_pos_gof, bg_color)

        rendered_rgb = render['image'][0, 0].permute(1, 2, 0).cpu().numpy()
        rendered_mask = render['mask'][0, 0].permute(1, 2, 0).cpu().numpy()
        rendered_depth = render['depth'][0, 0].permute(1, 2, 0).cpu().numpy()

        Image.fromarray((rendered_rgb * 255).astype(np.uint8)).save(os.path.join(subj_base_path, f'rgb_{view_idx}.png'))

        rgb_img_org = rendered_rgb
        mask_img_org = rendered_mask[..., 0].astype(bool)
        depth_img_org = rendered_depth[..., 0]

        img_pifuhd = rgb_img_org[..., :3]
        mask_pifuhd = mask_img_org

        img_pifuhd = img_pifuhd * mask_img_org[..., None] 
        img_pifuhd = to_tensor(img_pifuhd).unsqueeze(0).cuda() 

        normal_estimate = netG.forward(img_pifuhd).detach()
        normal_estimate = normal_estimate[0].permute(1,2,0).detach().cpu().numpy()

        normal_estimate = ((normal_estimate + 1) / 2 * mask_pifuhd[...,None]) * 255
        normal_estimate = np.concatenate([normal_estimate, mask_pifuhd[..., None] * 255], axis=-1)

        normal_map = normal_estimate / 255 * 2 - 1
        depth_refined = bilateral_normal_integration_depth(normal_map, depth_img_org, mask_img_org)
        depth_refined = np.nan_to_num(depth_refined, nan=0)
        depth_diff = abs(depth_refined - depth_img_org)

        depth_refined_save = np.uint16((depth_refined / 10) * 65535)

        rgb_image_tsdf = cv2.cvtColor(cv2.imread(os.path.join(subj_base_path, f'rgb_{view_idx}.png')), cv2.COLOR_BGR2RGB)
        os.remove(os.path.join(subj_base_path, f'rgb_{view_idx}.png'))

        c2w_gl = np.matmul(np.linalg.inv(flip_yz), c2w_orig)
        lgm_w2c = np.linalg.inv(c2w_gl)
        lgm_w2c = np.matmul(lgm_w2c, roty)
        w2c_comb = np.matmul(blender2opencv, lgm_w2c)

        ck2can = np.linalg.inv(w2c_comb)

        tsdf_vol.integrate(rgb_image_tsdf, depth_refined, K, ck2can, obs_weight=obs_weight) 

    verts, faces, norms, colors = tsdf_vol.get_mesh()
    outfile = os.path.join(subj_base_path, f'tsdf-rgbd.ply')

    import open3d as o3d
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    if outfile.endswith('.ply'):
        colors = colors.astype(np.float32) / 255.
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    print("Clean Mesh")
    triangle_clusters, cluster_n_triangles, cluster_area = o3d_mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100000
    o3d_mesh.remove_triangles_by_mask(triangles_to_remove)
    o3d_mesh.remove_unreferenced_vertices()

    o3d_mesh_smpl = o3d_mesh.simplify_quadric_decimation(500000)
    o3d.io.write_triangle_mesh(outfile, o3d_mesh_smpl)

