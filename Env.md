### Conda environment
- python 3.10: `conda create -n human3diffusion python=3.10`
- pytorch 2.1.0 with CUDA 12.1: `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121`
- xformers with CUDA 12.1: `pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu121`

### Gaussian Opacity Fields
- git clone gof: `git clone https://github.com/YuxuanSnow/gaussian-opacity-fields.git`, see `https://github.com/YuxuanSnow/gaussian-opacity-fields`
- install `diff-gaussian-rasterization`: `cd gaussian-opacity-fields && pip install submodules/diff-gaussian-rasterization`, `pip install submodules/simple-knn/ && cd ..`
- export CPATH for the camke (replace with your CUDA path): `export CPATH=/usr/local/cuda-11.8/targets/x86_64-linux/include:$CPATH`


### Packages
- pip install -r requirements.txt

### TSDF Fusion (Mesh extraction)
- tsdf fusion: https://github.com/andyzeng/tsdf-fusion-python, `pip install --user numpy opencv-python scikit-image numba`, `pip install --user pycuda`, `pip install scipy==1.11`