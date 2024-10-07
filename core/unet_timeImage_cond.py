import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Literal
from functools import partial

from core.attention import MemEffAttention

class MVAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_frames: int = 5,
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x):
        BV, C, H, W = x.shape
        B = BV // self.num_frames

        res = x
        x = self.norm(x)

        x = x.reshape(B, self.num_frames, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, -1, C)
        x = self.attn(x)
        x = x.reshape(B, self.num_frames, H, W, C).permute(0, 1, 4, 2, 3).reshape(BV, C, H, W)

        if self.residual:
            x = (x + res) * self.skip_scale
        return x

class ResnetBlock_cond(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if time_embed_dim is not None:
            self.time_emb_proj = nn.Linear(time_embed_dim, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x, t_emb=None):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x) 

        if self.time_emb_proj is not None:
            t_emb = self.time_emb_proj(t_emb)
            x = x + t_emb[:, :, None, None] 

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x

class DownBlock_cond(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_layers: int = 1,
        downsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()
 
        nets = []
        attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock_cond(in_channels, out_channels, time_embed_dim, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_emb=None):
        xs = []

        for attn, net in zip(self.attns, self.nets):
            x = net(x, t_emb)
            if attn:
                x = attn(x)
            xs.append(x) 

        if self.downsample:
            x = self.downsample(x)
            xs.append(x) 
  
        return x, xs 

class MidBlock_cond(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        nets.append(ResnetBlock_cond(in_channels, in_channels, time_embed_dim, skip_scale=skip_scale))
        for i in range(num_layers):
            nets.append(ResnetBlock_cond(in_channels, in_channels, time_embed_dim, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(in_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        
    def forward(self, x, t_emb=None):
        x = self.nets[0](x, t_emb)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                x = attn(x)
            x = net(x, t_emb)
        return x

class UpBlock_cond(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_layers: int = 1,
        upsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels

            nets.append(ResnetBlock_cond(cin + cskip, out_channels, time_embed_dim, skip_scale=skip_scale)) 
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, xs, t_emb=None):

        for attn, net in zip(self.attns, self.nets):
            res_x = xs[-1] 
            xs = xs[:-1]  
            x = torch.cat([x, res_x], dim=1)
            x = net(x, t_emb)
            if attn:
                x = attn(x)
            
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.upsample(x)

        return x

class UNet_timeimage_cond(nn.Module): 
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True),
        mid_attention: bool = True,
        up_channels: Tuple[int, ...] = (1024, 512, 256, 128, 64),
        up_attention: Tuple[bool, ...] = (True, True, False, False, False),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
    ):
        super().__init__()

        from diffusers.models.embeddings import Timesteps, TimestepEmbedding

        flip_sin_to_cos = True
        freq_shit = 0
        self.time_proj = Timesteps(down_channels[0], flip_sin_to_cos, freq_shit)

        timestep_input_dim = down_channels[0]
        time_embed_dim = down_channels[0] * 4
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock_cond(
                cin, cout, time_embed_dim,
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), 
                attention=down_attention[i],
                skip_scale=skip_scale,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.mid_block = MidBlock_cond(down_channels[-1], time_embed_dim, attention=mid_attention, skip_scale=skip_scale)

        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))]

            up_blocks.append(UpBlock_cond(
                cin, cskip, cout, time_embed_dim,
                num_layers=layers_per_block + 1,
                upsample=(i != len(up_channels) - 1),
                attention=up_attention[i],
                skip_scale=skip_scale,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, timestep):

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        t_emb = self.time_proj(timesteps) 

        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embedding(t_emb) 

        x = self.conv_in(x)
        
        xss = [x]
        for block in self.down_blocks:
            x, xs = block(x, emb)
            xss.extend(xs)
        
        x = self.mid_block(x, emb)

        for block in self.up_blocks:
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            x = block(x, xs, emb)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x) 
        
        return x
