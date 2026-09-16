# ERDES RD Optimization

Optimizations for **imbalanced 3D ophthalmic ultrasound video classification** (non-RD vs RD), built on PyTorch Lightning + Hydra + MONAI.

## Features

| Module | Description |
|--------|-------------|
| **Baseline** | 3D ResNet + temporal Top-K pooling |
| **Decoupled training** | Two-stage fine-tuning: Mixup → frozen backbone + Logit Adjustment + PR threshold search |
| **UM augmentation** | Unsharp masking on video tensors (online or offline cache) |
| **MONAI SSL Swin** | Official SSL Swin-UNETR (`feature_size=48`) encoder + classification head |
| **Fast path** | Offline `.pt` tensor cache on disk for ~10–20 min/epoch (ResNet) |

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for metrics comparison.

## Setup

```bash
conda create -n erdes python=3.10
conda activate erdes
pip install -r requirements.txt
```

Place videos under `data/erdes/` matching paths in `data/splits/non_rd_vs_rd/*.csv`.

## Quick start

```bash
# 1) Build video cache (once)
python scripts/build_video_cache.py --split all --workers 4

# 2) Baseline ResNet3D
python erdes/train.py experiment=non_rd_vs_rd/resnet3d trainer.max_epochs=15

# 3) Decoupled (recommended for F1 / sensitivity)
python erdes/train.py experiment=non_rd_vs_rd/resnet3d_decoupled_fast

# 4) UM — build UM cache then train
python scripts/build_um_cache.py --split all --workers 4
python erdes/train.py experiment=non_rd_vs_rd/resnet3d_um_fast

# 5) MONAI SSL Swin (downloads ~719MB weights on first run)
python erdes/train.py experiment=non_rd_vs_rd/swinunetr_monai_ssl_fast_2ep
```

Windows: use `scripts/start_*_train.ps1` equivalents.

## SSL weights

Not included in the repo. Auto-downloaded to `weights/ssl_pretrained_weights.pth` from [MONAI extra test data](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth).

## Hardware notes

- Tested on RTX 4060 Laptop 8GB, Windows
- Swin SSL is slow (~hours/epoch); ResNet fast path is much faster with cache
- Set `callbacks.rich_progress_bar=null` on Windows if Rich console errors occur

## License

Add your license here (e.g. MIT).
