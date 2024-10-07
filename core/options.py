import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional

@dataclass
class Options:
    input_size: int = 256
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256, 128)
    up_attention: Tuple[bool, ...] = (True, True, True, False, False)
    splat_size: int = 128
    output_size: int = 512

    data_mode: Literal['imagedream', 'lgm'] = 'imagedream'
    fovy: float = 49.1
    znear: float = 0.5
    zfar: float = 2.5
    num_views: int = 12
    num_input_views: int = 4
    cam_radius: float = 1.5 
    num_workers: int = 8

    workspace: str = './workspace'
    resume: Optional[str] = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 30
    lambda_lpips: float = 1.0
    lambda_distortion: float = 0.0
    lambda_normal: float = 0.0
    lambda_depth: float = 0.0
    gradient_clip: float = 1.0
    mixed_precision: str = 'fp16'
    lr: float = 4e-4
    prob_grid_distortion: float = 0.5
    prob_cam_jitter: float = 0.5

    test_path: Optional[str] = None

    force_cuda_rast: bool = False
    fancy_video: bool = False

    tracker_project_name: str = None
    

config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['lrm'] = 'the default settings for LGM'
config_defaults['lrm'] = Options()

config_doc['small'] = 'small model with lower resolution Gaussians'
config_defaults['small'] = Options(
    input_size=256,
    splat_size=64,
    output_size=256,
    batch_size=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['big'] = 'big model with higher resolution Gaussians'
config_defaults['big'] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128),
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size=512, 
    batch_size=8,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['tiny'] = 'tiny model for ablation'
config_defaults['tiny'] = Options(
    input_size=256, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    splat_size=64,
    output_size=256,
    batch_size=16,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
