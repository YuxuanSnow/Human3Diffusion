from mvdream.pipeline_imagedream import ImageDreamPipeline
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
)

from transformers import (
    CLIPTextModel,
    CLIPVisionModel,
    CLIPTokenizer,
)
from mvdream.mv_unet import MultiViewUNetModel

from safetensors.torch import load_file


from core.diffusion3d_imagedream import diffusion3dgs_noise_gof
from core.options import Options


import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import einops


from torchvision.utils import make_grid, save_image
import os
from skimage.io import imsave
import numpy as np

def get_2ddiffusion_model(unet2d_dict_path, device, org_imgdream_path="ashawkey/imagedream-ipmv-diffusers"):

    noise_scheduler = DDIMScheduler.from_pretrained(org_imgdream_path, subfolder="scheduler", revision=None) 
    image_encoder = CLIPVisionModel.from_pretrained(org_imgdream_path, subfolder="image_encoder", revision=None)
    text_encoder = CLIPTextModel.from_pretrained(org_imgdream_path, subfolder="text_encoder", revision=None)
    tokenizer = CLIPTokenizer.from_pretrained(org_imgdream_path, subfolder="tokenizer", revision=None)
    feature_extractor = None 
    vae = AutoencoderKL.from_pretrained(org_imgdream_path, subfolder="vae", revision=None)
    unet = MultiViewUNetModel.from_pretrained(org_imgdream_path, subfolder="unet", revision=None)

    # load 2d unet model
    ckpt_mvd = load_file(unet2d_dict_path, device='cpu')
    state_dict = unet.state_dict()
    for k, v in ckpt_mvd.items():
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
        else:
            print(f'[WARN] unexpected param {k}: {v.shape}')
    print("############# MVD models loaded #############")
    
    pipe = ImageDreamPipeline(
                            vae=vae,
                            unet=unet,
                            image_encoder=image_encoder,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            scheduler=noise_scheduler,
                        )
    pipe = pipe.to(device)

    pipe.unet.eval()
    pipe.unet.requires_grad_(False)
    pipe.vae.eval()
    pipe.vae.requires_grad_(False)
    pipe.image_encoder.eval()
    pipe.image_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.text_encoder.requires_grad_(False)

    return pipe



def get_3ddiffusion_model(unet3d_dict_path, device, opt):

    diffusion3dgs_model = diffusion3dgs_noise_gof(opt)
    diffusion3dgs_model = diffusion3dgs_model.to(device)

    diffusion3dgs_model.eval()
    diffusion3dgs_model.requires_grad_(False)

    ckpt = load_file(unet3d_dict_path, device='cpu')
    state_dict = diffusion3dgs_model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
        else:
            print(f'[WARN] unexpected param {k}: {v.shape}')

    print("############# MVR models loaded #############")

    return diffusion3dgs_model


def joint_2d_3d_diffusion(batch, device, diffusion_2d_pipe, diffusion_3d_pipe, weight_dtype=torch.float32):

    batch_size = 1
    num_view = 4
    batch_orthogonal_size = batch_size * num_view

    batch["context_image"] = batch["context_image"].squeeze(dim=1)

    input_image = batch["context_image"].to(device=device, dtype=weight_dtype) 
    gt_pose = batch["imagedream_cam_poses_gt"].to(device=device, dtype=weight_dtype)

    cam_view_input = batch['cam_view_imagedream'].to(device=device, dtype=weight_dtype)
    cam_view_proj_input = batch['cam_view_proj_imagedream'].to(device=device, dtype=weight_dtype)
    cam_pos_input = batch['cam_pos_imagedream'].to(device=device, dtype=weight_dtype)

    gt_pose = gt_pose.view(-1, 4, 16) 
    
    h, w = input_image.shape[2:]

    GUIDANCE_SCALE = 5.0

    diffusion_2d_pipe.scheduler.set_timesteps(50, device=device)
    timesteps = diffusion_2d_pipe.scheduler.timesteps

    image_embeds_neg, image_embeds_pos = diffusion_2d_pipe.encode_image(input_image, device, 1) 
    image_latents_neg, image_latents_pos = diffusion_2d_pipe.encode_image_latents(input_image, device, 1) 

    _prompt_embeds = diffusion_2d_pipe._encode_prompt(
            prompt=", 3d asset photorealistic human scan",
            device=device,
            num_images_per_prompt=1 * batch_size, 
            do_classifier_free_guidance=True,
            negative_prompt="uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear.",
        ) 
    prompt_embeds_neg, prompt_embeds_pos = _prompt_embeds.chunk(2) 

    actual_num_frames = 5
    latents: torch.Tensor = diffusion_2d_pipe.prepare_latents(
        5 * 1 * batch_size,
        4,
        256,
        256,
        prompt_embeds_pos.dtype,
        device,
        None,
        None,
    ) 

    camera_pose_ = gt_pose.view(batch_size, 4, 16)
    padding = [0] * (len(camera_pose_.shape) * 2)
    padding[-3] = 1
    padding_tuple = tuple(padding)
    camera = F.pad(camera_pose_, padding_tuple).to(dtype=latents.dtype, device=device)
    camera = camera.repeat_interleave(1, dim=0)
    camera = einops.rearrange(camera, 'b nv c -> (b nv) c')

    for i, t in enumerate(timesteps):

        ### 2d diffusion: get 3d inconsistent x0_tilde from 2d diffusion ###

        multiplier = 2 
        latent_model_input = torch.cat([latents] * multiplier)
        latent_model_input = diffusion_2d_pipe.scheduler.scale_model_input(latent_model_input, t) 

        current_timestep_condition = torch.tensor([t] * actual_num_frames * batch_size * multiplier, dtype=latent_model_input.dtype, device=device)

        unet_inputs = {
            'x': latent_model_input,
            'timesteps': current_timestep_condition,
            'context': torch.cat([prompt_embeds_neg] * actual_num_frames + [prompt_embeds_pos] * actual_num_frames),
            'num_frames': actual_num_frames,
            'camera': torch.cat([camera] * multiplier),
            'ip': torch.cat([image_embeds_neg] * actual_num_frames + [image_embeds_pos] * actual_num_frames),
            'ip_img': torch.cat([image_latents_neg] + [image_latents_pos]) 
        }

        noise_pred = diffusion_2d_pipe.unet.forward(**unet_inputs)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)    
        unet_noise_pred = noise_pred_uncond + 5.0 * (noise_pred_text - noise_pred_uncond)
        
        unet_noise_pred_ = einops.rearrange(unet_noise_pred, "(b nv) c h w -> b nv c h w", nv=actual_num_frames)
        unet_orthogonal_noise_pred = unet_noise_pred_[:, :-1, :, :, :] 
        unet_orthogonal_noise_pred = einops.rearrange(unet_orthogonal_noise_pred, "b nv c h w -> (b nv) c h w")

        noisy_latents = latents[:-1, :, :, :]
        pred_latent_epsilon = unet_orthogonal_noise_pred 
        alpha_prod_t = diffusion_2d_pipe.scheduler.alphas_cumprod[t] 
        beta_prod_t = 1 - alpha_prod_t 
        pred_original_latents = (noisy_latents - beta_prod_t ** (0.5) * pred_latent_epsilon) / alpha_prod_t ** (0.5) 

        ### 3d diffusion: get 3d consistent x0_hat from 3d diffusion ###    
         
        vae_decoded_x0 = diffusion_2d_pipe.vae.decode(1 / diffusion_2d_pipe.vae.config.scaling_factor * pred_original_latents).sample 
        vae_decoded_xt = diffusion_2d_pipe.vae.decode(1 / diffusion_2d_pipe.vae.config.scaling_factor * noisy_latents).sample 

        vae_decoded_x0 = einops.rearrange(vae_decoded_x0, "(b n) c h w -> b n c h w", n=num_view)
        vae_decoded_xt = einops.rearrange(vae_decoded_xt, "(b n) c h w -> b n c h w", n=num_view)
        vae_decoded_x0xt = torch.cat([vae_decoded_x0, vae_decoded_xt], dim=2)

        context_image_duplicate = torch.cat([input_image.unsqueeze(dim=1), input_image.unsqueeze(dim=1)], dim=2)
        vae_decoded_x0xt_with_clear_context = torch.cat([vae_decoded_x0xt, context_image_duplicate], dim=1)
        vae_decoded_x0xt_with_clear_context = einops.rearrange(vae_decoded_x0xt_with_clear_context, "b n c h w -> (b n) c h w") 
        vae_decoded_x0xt_with_clear_context = (vae_decoded_x0xt_with_clear_context / 2 + 0.5).clamp(0, 1)

        imagenet_mean =  (0.485, 0.456, 0.406, 0.485, 0.456, 0.406) 
        imagenet_std = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
        diffusion3d_img_input_x0xt_with_context = TF.normalize(vae_decoded_x0xt_with_clear_context, imagenet_mean, imagenet_std) 
        diffusion3d_img_input_x0xt_with_context = einops.rearrange(diffusion3d_img_input_x0xt_with_context, "(b n) c h w -> b n c h w", n=num_view+1) 

        gt_cameras_embedding = batch["diffusion3d_cam_poses_input_embedding"].to(device=diffusion3d_img_input_x0xt_with_context.device, dtype=weight_dtype) 
        input_cameras_embedding = batch['context_ray_embedding'].to(device=diffusion3d_img_input_x0xt_with_context.device, dtype=weight_dtype)

        camera_embeddings_with_context = torch.cat([gt_cameras_embedding, input_cameras_embedding], dim=1) 
        diffusion3d_input_with_context = torch.cat([diffusion3d_img_input_x0xt_with_context, camera_embeddings_with_context], dim=2) 

        t_ = einops.repeat(torch.tensor([t]), 'b -> (b n)', n=num_view) 
        t_ = einops.rearrange(t_, '(b n) -> b n', b=batch_size) 
        timestep_with_ip = t_.new_zeros((batch_size, 1), dtype=t.dtype) 
        timestep_with_ip = torch.cat([t_, timestep_with_ip], dim=1)
        timestep_with_ip = einops.rearrange(timestep_with_ip, 'b nv -> (b nv)', nv=actual_num_frames).to(device=diffusion3d_img_input_x0xt_with_context.device, dtype=weight_dtype) 

        gaussians = diffusion_3d_pipe.forward_gaussians(diffusion3d_input_with_context, timestep_with_ip) 

        mvr_rendering = diffusion_3d_pipe.gof.render(gaussians, cam_view_input, cam_view_proj_input, cam_pos_input, scale_modifier=1)
        rendered_input_image = mvr_rendering['image']
        rendered_input_image_ = einops.rearrange(rendered_input_image, 'b v c h w -> (b v) c h w')
        vae_mean =  (0.5, 0.5, 0.5)
        vae_std = (0.5, 0.5, 0.5)
        rendered_input_image_ = TF.normalize(rendered_input_image_, vae_mean, vae_std) 
        rendered_input_image_ = F.interpolate(rendered_input_image_.clone(), size=(256, 256), mode='bilinear', align_corners=False) 
        rendered_input_latent = diffusion_2d_pipe.vae.encode(rendered_input_image_).latent_dist.sample().detach() * diffusion_2d_pipe.vae.config.scaling_factor 

        ### get xt-1 from x0 hat and xt ###
        pred_epsilon_from_x0hat = (noisy_latents - alpha_prod_t ** (0.5) * rendered_input_latent) / beta_prod_t ** (0.5) 
        pred_epsilon_from_x0hat = einops.rearrange(pred_epsilon_from_x0hat, '(b n) c h w -> b n c h w', n=num_view) 
        unet_pred_noise_contexview = unet_noise_pred_[:, -1:, :, :, :] 
        pred_epsilon_from_x0hat = torch.cat([pred_epsilon_from_x0hat, unet_pred_noise_contexview], dim=1)
        pred_epsilon_from_x0hat = einops.rearrange(pred_epsilon_from_x0hat, 'b n c h w -> (b n) c h w')

        latents: torch.Tensor = diffusion_2d_pipe.scheduler.step(pred_epsilon_from_x0hat, t, latents, return_dict=False)[0]

    latent_model_input = torch.cat([latents] * multiplier)
    latent_model_input = diffusion_2d_pipe.scheduler.scale_model_input(latent_model_input, t) 

    unet_inputs = {
        'x': latent_model_input,
        'timesteps': torch.tensor([t] * actual_num_frames * batch_size * multiplier, dtype=latent_model_input.dtype, device=device),
        'context': torch.cat([prompt_embeds_neg] * actual_num_frames + [prompt_embeds_pos] * actual_num_frames),
        'num_frames': actual_num_frames,
        'camera': torch.cat([camera] * multiplier),
        'ip': torch.cat([image_embeds_neg] * actual_num_frames + [image_embeds_pos] * actual_num_frames),
        'ip_img': torch.cat([image_latents_neg] + [image_latents_pos])
    }

    noise_pred = diffusion_2d_pipe.unet.forward(**unet_inputs)

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)    
    unet_noise_pred = noise_pred_uncond + 5.0 * (noise_pred_text - noise_pred_uncond)
    
    unet_noise_pred_ = einops.rearrange(unet_noise_pred, "(b nv) c h w -> b nv c h w", nv=actual_num_frames)
    unet_orthogonal_noise_pred = unet_noise_pred_[:, :-1, :, :, :] 
    unet_orthogonal_noise_pred = einops.rearrange(unet_orthogonal_noise_pred, "b nv c h w -> (b nv) c h w") 

    noisy_latents = latents[:-1, :, :, :]
    pred_latent_epsilon = unet_orthogonal_noise_pred 
    alpha_prod_t = diffusion_2d_pipe.scheduler.alphas_cumprod[t] 
    beta_prod_t = 1 - alpha_prod_t 
    pred_original_latents = (noisy_latents - beta_prod_t ** (0.5) * pred_latent_epsilon) / alpha_prod_t ** (0.5)

    vae_decoded_x0 = diffusion_2d_pipe.vae.decode(1 / diffusion_2d_pipe.vae.config.scaling_factor * pred_original_latents).sample 
    vae_decoded_xt = diffusion_2d_pipe.vae.decode(1 / diffusion_2d_pipe.vae.config.scaling_factor * noisy_latents).sample 

    vae_decoded_x0 = einops.rearrange(vae_decoded_x0, "(b n) c h w -> b n c h w", n=num_view)
    vae_decoded_xt = einops.rearrange(vae_decoded_xt, "(b n) c h w -> b n c h w", n=num_view) 
    
    vae_decoded_x0xt = torch.cat([vae_decoded_x0, vae_decoded_xt], dim=2)

    context_image_duplicate = torch.cat([input_image.unsqueeze(dim=1), input_image.unsqueeze(dim=1)], dim=2) 
    vae_decoded_x0xt_with_clear_context = torch.cat([vae_decoded_x0xt, context_image_duplicate], dim=1) 
    vae_decoded_x0xt_with_clear_context = einops.rearrange(vae_decoded_x0xt_with_clear_context, "b n c h w -> (b n) c h w") 
    vae_decoded_x0xt_with_clear_context = (vae_decoded_x0xt_with_clear_context / 2 + 0.5).clamp(0, 1) 

    imagenet_mean =  (0.485, 0.456, 0.406, 0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
    diffusion3d_img_input_x0xt_with_context = TF.normalize(vae_decoded_x0xt_with_clear_context, imagenet_mean, imagenet_std) 
    diffusion3d_img_input_x0xt_with_context = einops.rearrange(diffusion3d_img_input_x0xt_with_context, "(b n) c h w -> b n c h w", n=num_view+1)

    gt_cameras_embedding = batch["diffusion3d_cam_poses_input_embedding"].to(device=diffusion3d_img_input_x0xt_with_context.device, dtype=weight_dtype) 
    input_cameras_embedding = batch['context_ray_embedding'].to(device=diffusion3d_img_input_x0xt_with_context.device, dtype=weight_dtype)

    camera_embeddings_with_context = torch.cat([gt_cameras_embedding, input_cameras_embedding], dim=1) 
    diffusion3d_input_with_context = torch.cat([diffusion3d_img_input_x0xt_with_context, camera_embeddings_with_context], dim=2) 

    t_ = einops.repeat(torch.tensor([t]), 'b -> (b n)', n=num_view)
    t_ = einops.rearrange(t_, '(b n) -> b n', b=batch_size) 
    timestep_with_ip = t_.new_zeros((batch_size, 1), dtype=t.dtype)
    timestep_with_ip = torch.cat([t_, timestep_with_ip], dim=1)
    timestep_with_ip = einops.rearrange(timestep_with_ip, 'b nv -> (b nv)', nv=actual_num_frames).to(device=diffusion3d_img_input_x0xt_with_context.device, dtype=weight_dtype)

    gaussians = diffusion_3d_pipe.forward_gaussians(diffusion3d_input_with_context, timestep_with_ip)

    return gaussians


def save_generation_results(subject_save_folder, batch, device, gaussians, diffusion_3d_pipe, weight_dtype):


    input_image = batch["context_image"].to(device=device, dtype=weight_dtype) 
    # save generated 3dgs model
    diffusion_3d_pipe.gof.save_ply(gaussians, os.path.join(subject_save_folder, 'gs.ply'))

    # save input image for visualization
    input_image_save = ((np.concatenate(input_image.permute(0, 2, 3, 1).cpu().numpy(), 1) + 1) / 2 * 255).astype(np.uint8)
    imsave(os.path.join(subject_save_folder, 'input.png'), input_image_save)

    # save 32 views of the 3d model
    cam_view = batch['cam_view'].to(device=device, dtype=weight_dtype)
    cam_view_proj = batch['cam_view_proj'].to(device=device, dtype=weight_dtype)
    cam_pos = batch['cam_pos'].to(device=device, dtype=weight_dtype)
    rendered_output = diffusion_3d_pipe.gof.render(gaussians, cam_view, cam_view_proj, cam_pos, scale_modifier=1)
    rendered_output_image = rendered_output['image']
    rendered_output_mask = rendered_output['mask'] 
    rendered_output_image = einops.rearrange(rendered_output_image, 'b v c h w -> (b v) c h w')

    grid_rendered = make_grid(rendered_output_image, nrow=8) 
    save_image(grid_rendered, os.path.join(subject_save_folder, 'mvrendering_3dgs.png'))

    

