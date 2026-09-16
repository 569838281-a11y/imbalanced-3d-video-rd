# imbalanced-3d-video-rd

个人项目：面向 **眼部超声视频** 的视网膜脱离（RD）不平衡分类优化，以及本地 **级联 AI 医生**（RD → 黄斑状态 + Grad-CAM 依据）。

> 训练数据划分沿用公开眼部超声视频数据集的 CSV 路径约定；本仓库代码与实验配置、训练脚本、AI 医生与说明文档为本人后续开发与整理，**不包含**原数据集论文官网式 README / 宣传页内容。

## 本仓库包含什么

| 部分 | 说明 |
|------|------|
| **RD 训练** | ResNet3D + Top-K；二阶段解耦（Mixup → 冻骨干 + Logit Adjustment）；Unsharp Masking（UM）缓存训练 |
| **黄斑二级** | RD 阳性后门控的 macula detached vs intact（3D U-Net，本地 50 epoch） |
| **AI 医生** | `ai_doctor/`：上传超声 → 级联判别 → CaseResult → Findings/Impression + Grad-CAM 依据区 |
| **权重** | `weights/` 下两个最佳 ckpt（Git LFS）：RD（UM+Decoupled）与黄斑 UNet3D |

## 环境

```bash
conda create -n erdes python=3.10
conda activate erdes
pip install -r requirements.txt
# AI 医生 Web：另需 fastapi uvicorn（若环境中未装）
pip install fastapi uvicorn python-multipart
```

数据视频请放到 `data/erdes/`，路径与 `data/splits/**/*.csv` 一致。大体积 `.pt` 缓存默认在本机 `D:/ERDES/cache/`（可按配置修改）。

拉取带权重的仓库时请先安装 [Git LFS](https://git-lfs.com/)，再 `git lfs pull`。

## 训练（本人常用入口）

```bash
# 构建视频缓存 / UM 缓存
python scripts/build_video_cache.py --split all
python scripts/build_um_cache.py --split all

# UM + 二阶段解耦 50 epoch（本地）
# 配置: configs/experiment/non_rd_vs_rd/resnet3d_um_decoupled_50ep_local.yaml
scripts/start_um_decoupled_50ep_local.ps1

# 黄斑二级 3D U-Net 50 epoch（本地，含 resume / 早停）
python scripts/build_macula_cache.py
scripts/start_macula_unet3d_50ep_local.ps1
```

实验对比备忘见 `docs/EXPERIMENTS.md`、`final_summary.txt`；技术流程见 `docs/TECH_PIPELINE_FINAL.txt`。

## AI 医生

```bash
# 默认读取 weights/ 下 best ckpt；也可用环境变量覆盖
# AI_DOCTOR_RD_CKPT / AI_DOCTOR_MACULA_CKPT
scripts/start_ai_doctor.ps1
# 浏览器打开 http://127.0.0.1:7860/
```

流程：一阶段 RD 筛查 → 阳性则二阶段黄斑 → 结论卡 + 依据区（Grad-CAM×帧质量）+ Findings/Impression 报告。输出为辅助参考，不替代面诊。

## 权重文件

| 文件 | 用途 |
|------|------|
| `weights/um_decoupled_best.ckpt` | 一阶段 RD（UM + Decoupled，本地 50ep 最佳） |
| `weights/macula_unet3d_best.ckpt` | 二阶段黄斑（UNet3D，本地 50ep 最佳） |

## 主要目录

```text
ai_doctor/          # Web AI 医生（级联 + 可解释性）
configs/            # Hydra 实验与模型配置
erdes/              # 训练 / 数据 / Grad-CAM 等代码
data/splits/        # 划分 CSV（不含原始视频）
docs/               # 本人实验与流程备忘
scripts/            # 缓存构建、训练启动、可视化
weights/            # 本人训练的最佳权重（LFS）
```

## 硬件备忘

本地验证环境：Windows，RTX 4060 Laptop 8GB。Windows 下若 Rich 进度条报错，可设 `callbacks.rich_progress_bar=null`。

## License

见仓库根目录 `LICENSE`。
