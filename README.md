<div align="center">
<p align="center"> <img src="assets/Multinex_logo.png" width="100px"> </p>

# Multinex: Lightweight Low-Light Image Enhancement via Multi-prior Retinex (CVPR 2026)

**[Alexandru Brateanu](https://albrateanu.github.io/), [Tingting Mu](https://personalpages.manchester.ac.uk/staff/tingting.mu/Site/About_Me.html), [Codruta O. Ancuti](https://ro.linkedin.com/in/codruta), [Cosmin Ancuti](https://www.linkedin.com/in/cosmin-ancuti-86b3872/)**

[![Paper](https://img.shields.io/badge/Abstract-arXiv-green)](LINK_TO_PAPER)
[![Paper](https://img.shields.io/badge/PDF-arXiv-orange)](LINK_TO_PAPER)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://albrateanu.github.io/Multinex/)


</div>

### News
- **XX.04.2026 :** [Pre-print](https://albrateanu.github.io/Multinex/), [code](https://albrateanu.github.io/Multinex/), and [paper page](https://albrateanu.github.io/Multinex/) for **Multinex** (accepted at CVPR 2026) are released!

<hr />

> **Abstract:** *Low-light image enhancement (LLIE) aims to restore natural visibility, color fidelity, and structural detail under severe illumination degradation. State-of-the-art (SOTA) LLIE techniques often rely on large models and multi-stage training, limiting practicality for edge deployment. Moreover, their dependence on a single color space introduces instability and visible exposure or color artifacts. To address these, we propose Multinex, an ultra-lightweight structured framework that integrates multiple fine-grained representations within a principled Retinex residual formulation. It decomposes an image into illumination and color prior stacks derived from distinct analytic representations, and learns to fuse these representations into luminance and reflectance adjustments required to correct exposure. By prioritizing enhancement over reconstruction and exploiting lightweight neural operations, Multinex significantly reduces computational cost, exemplified by its lightweight (45K parameters) and nano (0.7K parameters) versions. Extensive benchmarks show that all lightweight variants significantly outperform their corresponding lightweight SOTA models, and reach comparable performance to heavy models.* 
<hr />

## Network Architecture

<img src = "assets/framework.png"> 

### Introduction
This repository contains the official implementation of **Multinex** for low-light image enhancement. It provides training and testing code for paired-image enhancement on standard benchmarks, together with pretrained checkpoints for direct evaluation.


## 1. Create Environment

We use a PyTorch 2 environment.

### 1.1 Create the environment

```bash
conda create -n multinex python=3.9 -y
conda activate multinex
```

### 1.2 Install PyTorch 2

Example for CUDA 11.8:

```bash
# With Pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# With Conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Change this to accommodate your CUDA version requirements.

### 1.3 Install dependencies

```bash
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
```

### 1.4 Install BasicSR

```bash
pip install -e .
```

&nbsp;

## 2. Prepare Dataset

The current release supports:
- LOL-v1
- LOL-v2-real
- LOL-v2-synthetic

Organize the datasets as follows:

<details close>
<summary><b>Dataset structure</b></summary>

```text
data/
├── LOLv1/
│   ├── Train/
│   │   ├── input/
│   │   └── target/
│   └── Test/
│       ├── input/
│       └── target/
└── LOLv2/
    ├── Real_captured/
    │   ├── Train/
    │   │   ├── Low/
    │   │   └── Normal/
    │   └── Test/
    │       ├── Low/
    │       └── Normal/
    └── Synthetic/
        ├── Train/
        │   ├── Low/
        │   └── Normal/
        └── Test/
            ├── Low/
            └── Normal/
```

</details>

The option files currently used in this repository are:
- `Options/Multinex_LOL-v1.yaml`
- `Options/Multinex_LOL-v2-real.yaml`
- `Options/Multinex_LOL-v2-syn.yaml`
- `Options/MultinexNano_LOLv1.yaml`
- `Options/MultinexNano_LOL-v2-real.yaml`
- `Options/MultinexNano_LOL-v2-synthetic.yaml`

&nbsp;

## 3. Testing

Download pretrained checkpoints and place them in:

```text
pretrained_weights/
```

Recommended checkpoint names:

```text
pretrained_weights/
├── Multinex_LOLv1.pth
├── Multinex_LOLv2_real.pth
├── Multinex_LOLv2_syn.pth
├── MultinexNano_LOLv1.pth
├── MultinexNano_LOLv2_real.pth
└── MultinexNano_LOLv2_syn.pth
```

Activate the environment first:

```bash
conda activate multinex
```

### Standard Multinex

```bash
# LOL-v1
python Enhancement/test.py --opt Options/Multinex_LOL-v1.yaml --weights pretrained_weights/Multinex_LOLv1.pth --dataset LOL_v1

# LOL-v2-real
python Enhancement/test.py --opt Options/Multinex_LOL-v2-real.yaml --weights pretrained_weights/Multinex_LOLv2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python Enhancement/test.py --opt Options/Multinex_LOL-v2-syn.yaml --weights pretrained_weights/Multinex_LOLv2_syn.pth --dataset LOL_v2_synthetic
```

### Multinex-Nano

```bash
# LOL-v1
python Enhancement/test.py --opt Options/MultinexNano_LOLv1.yaml --weights pretrained_weights/MultinexNano_LOLv1.pth --dataset LOL_v1

# LOL-v2-real
python Enhancement/test.py --opt Options/MultinexNano_LOL-v2-real.yaml --weights pretrained_weights/MultinexNano_LOLv2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python Enhancement/test.py --opt Options/MultinexNano_LOL-v2-synthetic.yaml --weights pretrained_weights/MultinexNano_LOLv2_syn.pth --dataset LOL_v2_synthetic
```

- #### Self-ensemble testing strategy

For stronger results, add `--self_ensemble` argument.

```bash
python Enhancement/test.py --opt Options/Multinex_LOL-v1.yaml --weights pretrained_weights/Multinex_LOLv1.pth --dataset LOL_v1 --self_ensemble
```

&nbsp;

## 4. Training

Training is launched through the BasicSR entrypoint.

```bash
# activate the environment
conda activate multinex

# Multinex on LOL-v1
python -m basicsr.train --opt Options/Multinex_LOL-v1.yaml

# Multinex on LOL-v2-real
python -m basicsr.train --opt Options/Multinex_LOL-v2-real.yaml

# Multinex on LOL-v2-synthetic
python -m basicsr.train --opt Options/Multinex_LOL-v2-syn.yaml

# Multinex-Nano on LOL-v1
python -m basicsr.train --opt Options/MultinexNano_LOLv1.yaml

# Multinex-Nano on LOL-v2-real
python -m basicsr.train --opt Options/MultinexNano_LOL-v2-real.yaml

# Multinex-Nano on LOL-v2-synthetic
python -m basicsr.train --opt Options/MultinexNano_LOL-v2-synthetic.yaml
```

&nbsp;

## 5. Citation

Cite our work if Multinex is useful to your research.

```
@inproceedings{multinex2026,
  title     = {Multinex: Lightweight Low-light Image Enhancement via Multi-prior Retinex},
  author    = {Alexandru Brateanu and Tingting Mu and Codruta O. Ancuti and Cosmin Ancuti},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

