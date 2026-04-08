&nbsp;

<div align="center">
<p align="center"> <img src="assets/Multinex_logo.png" width="100px"> </p>

# Multinex: Lightweight Low-Light Image Enhancement via Multi-prior Retinex

**Alexandru Brateanu, Tingting Mu, Codruta O. Ancuti, Cosmin Ancuti**

[![Paper](https://img.shields.io/badge/paper-arXiv-179bd3)](LINK_TO_PAPER)
[![Project Page](https://img.shields.io/badge/project-page-179bd3)](LINK_TO_PROJECT_PAGE)
[![Models](https://img.shields.io/badge/models-download-179bd3)](LINK_TO_MODELS)

&nbsp;

</div>

### Introduction
This repository contains the official implementation of **Multinex** for low-light image enhancement. It provides training and testing code for paired-image enhancement on standard benchmarks, together with pretrained checkpoints for direct evaluation.

### News
<!-- - **2026.xx.xx :** placeholder-->

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
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 1.3 Install dependencies

```bash
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
```

### 1.4 Install BasicSR

```bash
python setup.py develop --no_cuda_ext
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
python Enhancement/test_from_dataset.py --opt Options/Multinex_LOL-v1.yaml --weights pretrained_weights/Multinex_LOLv1.pth --dataset LOL_v1

# LOL-v2-real
python Enhancement/test_from_dataset.py --opt Options/Multinex_LOL-v2-real.yaml --weights pretrained_weights/Multinex_LOLv2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python Enhancement/test_from_dataset.py --opt Options/Multinex_LOL-v2-syn.yaml --weights pretrained_weights/Multinex_LOLv2_syn.pth --dataset LOL_v2_synthetic
```

### Multinex-Nano

```bash
# LOL-v1
python Enhancement/test_from_dataset.py --opt Options/MultinexNano_LOLv1.yaml --weights pretrained_weights/MultinexNano_LOLv1.pth --dataset LOL_v1

# LOL-v2-real
python Enhancement/test_from_dataset.py --opt Options/MultinexNano_LOL-v2-real.yaml --weights pretrained_weights/MultinexNano_LOLv2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python Enhancement/test_from_dataset.py --opt Options/MultinexNano_LOL-v2-synthetic.yaml --weights pretrained_weights/MultinexNano_LOLv2_syn.pth --dataset LOL_v2_synthetic
```

- #### Self-ensemble testing strategy

If supported by your testing script, add `--self_ensemble` to the command:

```bash
python Enhancement/test_from_dataset.py --opt Options/Multinex_LOL-v1.yaml --weights pretrained_weights/Multinex_LOLv1.pth --dataset LOL_v1 --self_ensemble
```

&nbsp;

## 4. Training

Training is launched through the BasicSR entrypoint.

```bash
# activate the environment
conda activate multinex

# Multinex on LOL-v1
python basicsr/train.py --opt Options/Multinex_LOL-v1.yaml

# Multinex on LOL-v2-real
python basicsr/train.py --opt Options/Multinex_LOL-v2-real.yaml

# Multinex on LOL-v2-synthetic
python basicsr/train.py --opt Options/Multinex_LOL-v2-syn.yaml

# Multinex-Nano on LOL-v1
python basicsr/train.py --opt Options/MultinexNano_LOLv1.yaml

# Multinex-Nano on LOL-v2-real
python basicsr/train.py --opt Options/MultinexNano_LOL-v2-real.yaml

# Multinex-Nano on LOL-v2-synthetic
python basicsr/train.py --opt Options/MultinexNano_LOL-v2-synthetic.yaml
```

&nbsp;

## 5. Notes

- Make sure the dataset paths in the YAML files match your local directory layout.
- Make sure the checkpoint matches the corresponding option file.
- If you encounter CUDA memory issues, reduce batch size or patch size in the YAML config.
- Logs and checkpoints are controlled by the training config.

### Optional TensorBoard

If enabled in the config, you can monitor training with:

```bash
tensorboard --logdir experiments
```
