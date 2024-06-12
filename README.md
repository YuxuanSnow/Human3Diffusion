# Human 3Diffusion: Realistic Avatar Creation via Explicit 3D Consistent Diffusion Models
#### [Project Page](https://yuxuan-xue.com/human-3diffusion) | [Paper](https://yuxuan-xue.com/human-3diffusion/paper/human-3diffusion.pdf)

Arxiv, 2024

[Yuxuan Xue](https://yuxuan-xue.com/)<sup>1 </sup>, [Xianghui Xie](https://virtualhumans.mpi-inf.mpg.de/people/Xie.html)<sup>1, 2</sup>, [Riccardo Marin](https://ricma.netlify.app/)<sup>1</sup>, [Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html)<sup>1, 2</sup>


<sup>1</sup>Real Virtual Human Group @ University of Tübingen & Tübingen AI Center \
<sup>2</sup>Max Planck Institute for Informatics, Saarland Informatics Campus

![](https://github.com/YuxuanSnow/Human3Diffusion/blob/main/assets/teaser_vid_short.gif)

## News :triangular_flag_on_post:
- [2024/06/14] Human 3Diffusion paper is available on [ArXiv](https://yuxuan-xue.com/human-3diffusion).
- [2024/06/14] Inference code and model weights is scheduled to be released after CVPR 2024.

## Key Insight :raised_hands:
- 2D foundation models are powerful but output lacks 3D consistency!
- 3D generative models can reconstruct 3D representation but is poor in generalization!
- How to combine 2D foundation models with 3D generative models?:
  - they are both diffusion-based generative models => **Can be synchronized at each diffusion step**
  - 2D foundation model helps 3D generation => **provides strong prior informations about 3D shape**
  - 3D representation guides 2D diffusion sampling => **use rendered output from 3D reconstruction for reverse sampling, where 3D consistency is guaranteed**

![](https://github.com/YuxuanSnow/Human3Diffusion/blob/main/assets/3diffusion_pipeline.png)


## Citation :writing_hand:

```bibtex
@inproceedings{xue2023human3diffusion,
  title     = {{Human 3Diffusion: Realistic Avatar Creation via Explicit 3D Consistent Diffusion Models}},
  author    = {Xue, Yuxuan and Xie, Xianghui and Marin, Riccardo and Pons-Moll, Gerard.},
  journal   = {Arxiv},
  year      = {2024},
}