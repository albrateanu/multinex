# Multinex 
Low-light Object Detection

**Please note**: Detection versions of Multinex use input downsampling by a factor of 4 for inference speed.

## Environment Setup:

#### 1. Environment creation
```bash
conda create -n Multinex_mmdet python=3.8 -y
conda activate Multinex_mmdet
```

#### 2. Pytorch 1.10 installation.

```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

In case your machine runs newer CUDA, conda will take very long trying to resolve dependencies. Therefore, use the below command to install specific versions via `pip` rather than through `conda`.

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

#### 3. MMCV Installation

```bash
# Install MMCV-full 1.4.0
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install remaining requirements
pip install opencv-python scipy
pip install -r requirements/build.txt
pip install -v -e .
```

## Dataset Setup

Download Exdark dataset and unzip it under `data/` directory.

From `Multinex/Detection/`, it should match the structure below:

```bash
data/EXDark
    ├── JPEGImages
    │   ├── IMGS               # Original low-light
    │   ├── IMGS_Kind          # Enhanced versions...
    ├── Annotations
    ├── main
    └── label
```

## Test

Please download model weights (Multinex + YOLOv3) from [Google Drive](https://drive.google.com/drive/folders/1l_hGZBRNG4v6tWu7-XKWBBvL0PNGtMi4) and place the `.pth` files under a new `weights/` directory.

```bash
# Lightweight
python tools/test.py configs/yolo/yolov3_Multinex_Exdark.py weights/MultinexYOLO.pth --eval mAP

# Nano
python tools/test.py configs/yolo/yolov3_MultinexNano_Exdark.py weights/MultinexNanoYOLO.pth --eval mAP
```

## Train

```bash
# Lightweight
python tools/train.py configs/yolo/yolov3_Multinex_Exdark.py

# Nano
python tools/train.py configs/yolo/yolov3_MultinexNano_Exdark.py
```
