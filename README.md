# Human 3Diffusion: Realistic Avatar Creation via Explicit 3D Consistent Diffusion Models
#### [Project Page](https://yuxuan-xue.com/human-3diffusion) | [Paper](https://yuxuan-xue.com/human-3diffusion/paper/human-3diffusion.pdf)

NeurIPS, 2024

[Yuxuan Xue](https://yuxuan-xue.com/)<sup>1 </sup>, [Xianghui Xie](https://virtualhumans.mpi-inf.mpg.de/people/Xie.html)<sup>1, 2</sup>, [Riccardo Marin](https://ricma.netlify.app/)<sup>1</sup>, [Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html)<sup>1, 2</sup>


<sup>1</sup>Real Virtual Human Group @ University of Tübingen & Tübingen AI Center \
<sup>2</sup>Max Planck Institute for Informatics, Saarland Informatics Campus

![](https://github.com/YuxuanSnow/Human3Diffusion/blob/main/assets/teaser_vid_short.gif)

## News :triangular_flag_on_post:
- [2024/10/07] Inference Code release. 
- [2024/09/25] Human 3Diffusion is accepted to NeurIPS 2024.
- [2024/06/14] Human 3Diffusion paper is available on [ArXiv](https://yuxuan-xue.com/human-3diffusion).
- [2024/06/14] Inference code and model weights is scheduled to be released after CVPR 2024.

## Key Insight :raised_hands:
- 2D foundation models are powerful but output lacks 3D consistency!
- 3D generative models can reconstruct 3D representation but is poor in generalization!
- How to combine 2D foundation models with 3D generative models?:
  - they are both diffusion-based generative models => **Can be synchronized at each diffusion step**
  - 2D foundation model helps 3D generation => **provides strong prior informations about 3D shape**
  - 3D representation guides 2D diffusion sampling => **use rendered output from 3D reconstruction for reverse sampling, where 3D consistency is guaranteed**

## Install
```
# Conda environment
conda create -n human3diffusion python=3.10
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu121

# Gaussian Opacity Fields
git clone https://github.com/YuxuanSnow/gaussian-opacity-fields.git
cd gaussian-opacity-fields && pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/ && cd ..
export CPATH=/usr/local/cuda-12.1/targets/x86_64-linux/include:$CPATH

# Dependencies
pip install -r requirements.txt

# TSDF Fusion (Mesh extraction) Dependencies
pip install --user numpy opencv-python scikit-image numba
pip install --user pycuda
pip install scipy==1.11
```

## Pretrained Weights
Our pretrained weight can be downloaded from huggingface.
```
mkdir checkpoints && cd checkpoints
wget https://huggingface.co/yuxuanx/human3diffusion/resolve/main/model.safetensors
wget https://huggingface.co/yuxuanx/human3diffusion/resolve/main/model_1.safetensors
wget https://huggingface.co/yuxuanx/human3diffusion/resolve/main/pifuhd.pt
cd ..
```

## Inference
```
# given one image, generate 3D-GS
# subject should be centered in a square image, please crop properly
python infer.py --test_imgs test_imgs --output output --checkpoints checkpoints

# given generated 3D-GS, perform TSDF mesh extraction
python infer_mesh.py --test_imgs test_imgs --output output --checkpoints checkpoints --mesh_quality high
```

## Citation :writing_hand:

```bibtex
@inproceedings{xue2023human3diffusion,
  title     = {{Human 3Diffusion: Realistic Avatar Creation via Explicit 3D Consistent Diffusion Models}},
  author    = {Xue, Yuxuan and Xie, Xianghui and Marin, Riccardo and Pons-Moll, Gerard.},
  journal   = {NeurIPS 2024},
  year      = {2024},
}