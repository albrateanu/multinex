&nbsp;

# Multinex

Lightweight low-light image enhancement codebase with paired-image training and evaluation pipelines.

This repository provides:
- training and testing for paired low-light enhancement
- support for LOL-v1, LOL-v2-real, and LOL-v2-synthetic
- standard and Nano model variants
- YAML-based experiment configuration
- a PyTorch 2 environment setup

## Overview

The enhancement pipeline is driven through YAML option files and the BasicSR training and evaluation entrypoints.

Available enhancement configs:
- `Options/Multinex_LOL-v1.yaml`
- `Options/Multinex_LOL-v2-real.yaml`
- `Options/Multinex_LOL-v2-syn.yaml`
- `Options/MultinexNano_LOLv1.yaml`
- `Options/MultinexNano_LOL-v2-real.yaml`
- `Options/MultinexNano_LOL-v2-synthetic.yaml`

## Repository Layout

```text
.
├── Enhancement/
│   ├── test_from_dataset.py
│   ├── utils.py
│   └── ...
├── Options/
│   ├── Multinex_LOL-v1.yaml
│   ├── Multinex_LOL-v2-real.yaml
│   ├── Multinex_LOL-v2-syn.yaml
│   ├── MultinexNano_LOLv1.yaml
│   ├── MultinexNano_LOL-v2-real.yaml
│   └── MultinexNano_LOL-v2-synthetic.yaml
├── basicsr/
├── pretrained_weights/
├── data/
└── setup.py
```

## 1. Environment Setup

This project is intended to run with a PyTorch 2 environment only.

### 1.1 Create environment

```bash
conda create -n multinex python=3.9 -y
conda activate multinex
```

### 1.2 Install PyTorch 2

Install the PyTorch build matching your CUDA version. Example for CUDA 11.8:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 1.3 Install Python dependencies

```bash
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
```

### 1.4 Install the repository

```bash
python setup.py develop --no_cuda_ext
```

## 2. Dataset Preparation

The current configs expect the following paired dataset structure under `data/`.

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

### Expected paths from the provided option files

#### LOL-v1

```text
data/LOLv1/Train/input
data/LOLv1/Train/target
data/LOLv1/Test/input
data/LOLv1/Test/target
```

#### LOL-v2-real

```text
data/LOLv2/Real_captured/Train/Low
data/LOLv2/Real_captured/Train/Normal
data/LOLv2/Real_captured/Test/Low
data/LOLv2/Real_captured/Test/Normal
```

#### LOL-v2-synthetic

```text
data/LOLv2/Synthetic/Train/Low
data/LOLv2/Synthetic/Train/Normal
data/LOLv2/Synthetic/Test/Low
data/LOLv2/Synthetic/Test/Normal
```

Make sure filenames match between low-light and target images.

## 3. Pretrained Weights

Place pretrained checkpoints in:

```text
pretrained_weights/
```

Recommended naming:

```text
pretrained_weights/
├── Multinex_LOLv1.pth
├── Multinex_LOLv2_real.pth
├── Multinex_LOLv2_syn.pth
├── MultinexNano_LOLv1.pth
├── MultinexNano_LOLv2_real.pth
└── MultinexNano_LOLv2_syn.pth
```

## 4. Testing

Activate the environment first:

```bash
conda activate multinex
```

### 4.1 Test standard models

#### LOL-v1

```bash
python Enhancement/test_from_dataset.py \
  --opt Options/Multinex_LOL-v1.yaml \
  --weights pretrained_weights/Multinex_LOLv1.pth \
  --dataset LOL_v1
```

#### LOL-v2-real

```bash
python Enhancement/test_from_dataset.py \
  --opt Options/Multinex_LOL-v2-real.yaml \
  --weights pretrained_weights/Multinex_LOLv2_real.pth \
  --dataset LOL_v2_real
```

#### LOL-v2-synthetic

```bash
python Enhancement/test_from_dataset.py \
  --opt Options/Multinex_LOL-v2-syn.yaml \
  --weights pretrained_weights/Multinex_LOLv2_syn.pth \
  --dataset LOL_v2_synthetic
```

### 4.2 Test Nano models

#### LOL-v1

```bash
python Enhancement/test_from_dataset.py \
  --opt Options/MultinexNano_LOLv1.yaml \
  --weights pretrained_weights/MultinexNano_LOLv1.pth \
  --dataset LOL_v1
```

#### LOL-v2-real

```bash
python Enhancement/test_from_dataset.py \
  --opt Options/MultinexNano_LOL-v2-real.yaml \
  --weights pretrained_weights/MultinexNano_LOLv2_real.pth \
  --dataset LOL_v2_real
```

#### LOL-v2-synthetic

```bash
python Enhancement/test_from_dataset.py \
  --opt Options/MultinexNano_LOL-v2-synthetic.yaml \
  --weights pretrained_weights/MultinexNano_LOLv2_syn.pth \
  --dataset LOL_v2_synthetic
```

### 4.3 Optional self-ensemble testing

If supported by your testing pipeline, append:

```bash
--self_ensemble
```

Example:

```bash
python Enhancement/test_from_dataset.py \
  --opt Options/Multinex_LOL-v1.yaml \
  --weights pretrained_weights/Multinex_LOLv1.pth \
  --dataset LOL_v1 \
  --self_ensemble
```

## 5. Training

Training is launched through the BasicSR entrypoint.

### 5.1 Train standard models

#### LOL-v1

```bash
python basicsr/train.py --opt Options/Multinex_LOL-v1.yaml
```

#### LOL-v2-real

```bash
python basicsr/train.py --opt Options/Multinex_LOL-v2-real.yaml
```

#### LOL-v2-synthetic

```bash
python basicsr/train.py --opt Options/Multinex_LOL-v2-syn.yaml
```

### 5.2 Train Nano models

#### LOL-v1

```bash
python basicsr/train.py --opt Options/MultinexNano_LOLv1.yaml
```

#### LOL-v2-real

```bash
python basicsr/train.py --opt Options/MultinexNano_LOL-v2-real.yaml
```

#### LOL-v2-synthetic

```bash
python basicsr/train.py --opt Options/MultinexNano_LOL-v2-synthetic.yaml
```

## 6. Notes on the YAML Configs

A typical option file defines:
- dataset paths
- batch size and workers
- validation settings
- optimizer and scheduler
- model architecture settings
- checkpoint and logging behavior

Common fields to edit first:
- `datasets.train.dataroot_gt`
- `datasets.train.dataroot_lq`
- `datasets.val.dataroot_gt`
- `datasets.val.dataroot_lq`
- `num_gpu`
- `batch_size_per_gpu`
- `train.total_iter`
- `path.pretrain_network_g`
- `logger.use_tb_logger`

## 7. Monitoring and Checkpoints

Logs and checkpoints are written according to the YAML config.

Useful fields:

```yaml
logger:
  print_freq: 500
  save_checkpoint_freq: 1000
  use_tb_logger: true
```

If TensorBoard logging is enabled:

```bash
tensorboard --logdir experiments
```

## 8. Model Complexity Utility

If your repository includes the summary helper in `Enhancement/utils.py`, model complexity can be inspected as:

```python
from utils import my_summary
my_summary(model, 256, 256, 3, 1)
```

Adjust the input size as needed.

## 9. Common Issues

### Dataset path errors

Check that the folder names exactly match the paths in the YAML files.

### Pair mismatch

Make sure low-light and target images have identical filenames and aligned contents.

### CUDA out of memory

Reduce one or more of:
- `batch_size_per_gpu`
- patch size (`gt_size`)
- worker count
- model width or variant size

### Wrong checkpoint loaded

Ensure the checkpoint matches the option file and model variant being tested.

## 10. Minimal Workflow

### Train

```bash
conda activate multinex
python basicsr/train.py --opt Options/Multinex_LOL-v1.yaml
```

### Test

```bash
python Enhancement/test_from_dataset.py \
  --opt Options/Multinex_LOL-v1.yaml \
  --weights pretrained_weights/Multinex_LOLv1.pth \
  --dataset LOL_v1
```