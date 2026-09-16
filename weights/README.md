# 本人训练权重（Git LFS）

| 文件 | 任务 |
|------|------|
| `um_decoupled_best.ckpt` | non_rd_vs_rd，UM + Decoupled 50ep |
| `macula_unet3d_best.ckpt` | macula_detached_vs_intact，UNet3D 50ep |

AI 医生默认从此目录加载；也可用环境变量 `AI_DOCTOR_RD_CKPT` / `AI_DOCTOR_MACULA_CKPT` 指向其它路径。
