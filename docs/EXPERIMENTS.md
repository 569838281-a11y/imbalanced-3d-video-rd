================================================================================
  ERDES non_rd_vs_rd 四实验总览 — 对比、原因分析与架构参数
  任务：视网膜脱离（RD） vs 非 RD 二分类 | 数据约 9% 正类（不平衡）
  汇总日期：2026-06-08
================================================================================

一、四次实验定义与运行状态
--------------------------------------------------------------------------------

  | 序号 | 实验名称           | 配置入口                              | 训练状态 |
  |------|--------------------|---------------------------------------|----------|
  | 1    | 基础 ResNet3D      | non_rd_vs_rd/resnet3d                  | 已完成 15 epoch |
  | 2    | 二阶段解耦训练     | non_rd_vs_rd/resnet3d_decoupled_fast   | 已完成 15 epoch |
  | 3    | UM 处理模型        | non_rd_vs_rd/resnet3d_um_fast          | 已完成 15 epoch |
  | 4    | Swin-UNETR+SSL     | non_rd_vs_rd/swinunetr_monai_ssl_fast_2ep | 进行中 |

  统一任务与数据划分：
    • train 3873 / val 431 / test（同 CSV 体系）
    • 输入体积默认 [1, 96, 128, 128]（灰度 3D 超声视频）
    • 实验 2、3、4 快速路径使用 D:/ERDES/cache/non_rd_vs_rd/*.pt


二、验证集核心指标对比（阈值 0.5，最佳 checkpoint 复评）
--------------------------------------------------------------------------------

  说明：
    • 实验 1 指标来自 15 epoch 最佳 ckpt（epoch_015）在 val 上的复现评估记录。
    • 实验 2、3 来自各自 plots/metrics_summary.json（与 plot_*_fast_results.py 一致）。
    • 实验 4 尚未跑满；下表「训练日志 epoch 末」仅作参考，不能与前三个公平对标。

  ┌─────────────────┬──────────┬──────────┬──────────┬──────────┬─────┬─────┐
  │ 模型            │ Val Acc  │ Val F1   │ PR-AUC   │ Sens(Rec)│ FP  │ FN  │
  ├─────────────────┼──────────┼──────────┼──────────┼──────────┼─────┼─────┤
  │ 1 基础 ResNet3D  │ 0.984    │ 0.904    │ 0.936    │ 0.825    │ 0   │ —   │
  │ 2 二阶段解耦     │ 0.984    │ 0.909    │ 0.941    │ 0.875    │ 2   │ 5   │
  │ 3 UM+ResNet3D    │ 0.984    │ 0.907    │ 0.967    │ 0.850    │ 1   │ 6   │
  │ 4 Swin SSL(2ep)  │ 未完成   │ 未完成   │ 未完成   │ —        │ —   │ —   │
  └─────────────────┴──────────┴──────────┴──────────┴──────────┴─────┴─────┘

  最佳 checkpoint 路径：
    1) logs/train/runs/rd/resnet3d/2026-05-29_22-02-10/checkpoints/resnet3d/
       resnet3d_best_epoch_015.ckpt
    2) .../resnet3d_decoupled_fast/2026-06-02_11-02-26/checkpoints/resnet3d_decoupled/
       decoupled_fast_epoch_010.ckpt  （monitor val/f1，epoch 10 最佳）
    3) .../resnet3d_um_fast/2026-06-02_20-01-30/checkpoints/resnet3d_um/
       resnet3d_um_fast_epoch_012.ckpt  （monitor val/acc，epoch 12）
    4) .../swinunetr_monai_ssl_fast_2ep/<时间戳>/checkpoints/swinunetr_monai_ssl/
       encoder_epoch_000/001.pth + swinunetr_monai_ssl_epoch_000/001.ckpt

  训练曲线 Highlights（val F1 @0.5，日志）：
    • 实验 3 UM：epoch 12 达 val F1≈0.907；epoch 14 达 0.911
    • 实验 2 解耦：epoch 10 达 val F1≈0.909（阶段一末）；阶段二（11–14）F1 波动下降
    • 实验 4 Swin：epoch 0 训前 val F1=0（几乎全预测负类），需更多 epoch 才有意义


三、各实验特点：谁更好、解决了什么、仍缺什么
--------------------------------------------------------------------------------

【实验 1 — 基础 ResNet3D】

  更优秀之处：
    • 流程最简单，易复现；在 0.5 阈值下 FP=0，特异性极高，临床「误报 RD」风险低。
    • Val 复评（epoch_015 ckpt）：Acc=0.984, F1=0.904, PR-AUC=0.936, Sens=0.825, Prec=1.0。
    • 配合 D: 缓存后单 epoch 约 10–20 min（与 ResNet 系列一致）。

  解决的痛点：
    • 建立 non_rd_vs_rd 基线；Top-K 时序池化让 CNN 聚焦运动最明显的帧，适合超声视频。

  不足：
    • 监控指标为 val/acc，对不平衡数据有误导性（Acc 高但 F1/敏感度可偏低）。
    • 无针对少数类 RD 的 Logit 校正、Mixup、PR 阈值搜索。
    • 在线 mp4 训练极慢（未缓存时）；未利用边缘增强（UM）。

  原因（结构/训练）：
    • MONAI ResNet3D（BasicBlock×[4,4,4,4]，通道 64→512）+ 自定义 Top-K 池化头；
      参数量与单步算力远小于 3D Swin，故训练快。
    • 标准 BCE + Adam，全程同一学习率策略，决策边界未针对 π_RD≈9% 做后处理。


【实验 2 — 二阶段解耦训练（Decoupled）】

  更优秀之处：
    • Val F1（0.5 阈值）四者中最高：0.909；敏感度 0.875，在少 FP 前提下多检出 RD。
    • 显式优化 val/f1、val/pr_auc；训练结束可做 PR 曲线阈值搜索（Precision≥0.5 最大 Recall）。
    • 阶段一 Mixup + 阶段二 Logit Adjustment，直接针对「高 Acc、低召回」不平衡痛点。

  解决的痛点：
    • 类别极不平衡下，单纯 ERM 训练易偏向预测负类；阶段二冻结骨干、只调头并校正 logit，
      等价于降低 RD 阳性判定门槛，提升 Recall/F1。
    • 与基础模型同骨干，额外训练策略而非换大模型，工程上可接受（fast 路径 ~3–5 h/15ep）。

  不足：
    • 阶段二（epoch 11–14）指标波动大，部分 epoch val F1 跌至 0.79；需依赖 epoch 10 最佳 ckpt。
    • 训练结束 PR 阈值回调曾因 float16 缓存与模型 dtype 不一致报错（Half vs Float）。
    • FP=2，略高于基础/UM；PR 优化阈值下 F1 仅 ~0.66（高召回、低精度权衡）。
    • train 侧 F1 日志为 0（compute_train_metrics 关闭），不便看欠拟合。

  原因（结构/训练）：
    • 骨干与实验 1 相同；增益几乎全部来自 DecoupledModelModule 训练日程，而非新 backbone。
    • stage1_end=11：前 11 epoch 全网络 + Mixup(α=0.2)；第 11 epoch 起 freeze_backbone，
      LogitAdjuster(τ=1.0) 启用，仅 ClassificationHead 更新。
    • ReduceLROnPlateau(mode=max) 监控验证指标；与实验 1 的 min-loss 调度不同。


【实验 3 — UM（Unsharp Masking）+ ResNet3D】

  更优秀之处：
    • PR-AUC 最高：0.967，排序/筛查能力最好；FP=1，特异性仍极强。
    • Val F1≈0.907，接近解耦；对边缘敏感任务有明确先验（锐化突出膜/病变边界）。

  解决的痛点：
    • 超声对比度低、边界模糊；UM 在数据侧增强高频细节，不改变网络结构即可做消融。
    • 离线 UM 缓存（non_rd_vs_rd_um）后速度与实验 1 fast 同级，避免在线 OpenCV 拖慢 I/O。

  不足：
    • 敏感度 0.850，低于解耦（0.875）；FN=6，漏诊 RD 略多。
    • 依赖额外建 UM 缓存；test 集默认不施加 UM，train/val 与 test 分布略不一致。
    • 仍用 val/acc 选模，与 F1 最优 epoch 可能不完全一致。

  原因（结构/训练）：
    • 网络与实验 1 完全相同；提升来自输入分布（Sharpened = 1.5×Orig − 0.5×Blur）。
    • 逐帧 2D 高斯模糊+融合，强化局部梯度，使 ResNet 卷积更容易捕捉边缘运动模式。
    • 与解耦相比：偏「数据增强」，非「损失/决策校正」；故 PR-AUC 优、Recall 略逊。


【实验 4 — MONAI SSL Swin-UNETR 加载预训练权重】

  更优秀之处（预期，需跑满微调后验证）：
    • 骨干在大量 3D 医学影像上自监督预训练（157 个 encoder 张量载入），表征能力强于随机初始化 CNN。
    • feature_size=48（768 维瓶颈），容量大于项目内 feature_size=24 的自建 Swin。

  解决的痛点：
    • 探索更强时空建模（3D Window Attention）是否优于 ResNet3D；
    • 利用公开 SSL 权重，减轻从零训练 Swin 的成本。

  不足（当前实测）：
    • 训练极慢：RTX 4060 8GB 上单 epoch 常达数小时（即使 D: 缓存 + bs2）。
    • 仅计划 2 epoch 时，分类头几乎未收敛；epoch 0 常出现「高 Acc、F1=0」假象。
    • 显存接近满载（~7.7/8 GB），bs4+gradient checkpoint 反而更慢。
    • 未完成与前三个实验同标准的 val 复评与 test 集报告。

  原因（结构/训练）：
    • Swin-UNETR 含多层 3D Window Attention、Patch Merging，FLOPs 与显存远大于 ResNet。
    • 新初始化的 ClassificationHead（768→384→1）需多 epoch 适配；SSL 仅初始化 swinViT。
    • Adam lr=1e-4、ReduceLROnPlateau(patience=10)，2 epoch 内学习率几乎不降。
    • 与 ResNet 系列不同模块（ModelModule 标准路径，无 Mixup/LogitAdj/UM）。


四、综合对比结论
--------------------------------------------------------------------------------

  指标维度：
    • 要最高 F1 + 敏感度（少漏诊）：优先实验 2 解耦（F1=0.909, Sens=0.875）。
    • 要最高 PR-AUC + 极少 FP（筛查）：优先实验 3 UM（PR-AUC=0.967, FP=1）。
    • 要最简单可交付基线：实验 1（FP=0，实现成本最低）。
    • 大模型潜力：实验 4 需 ≥5–8 epoch 及以上才有公平对比价值；当前 2 epoch 仅作试点。

  工程维度：
    • 训练时间：1 ≈ 3 fast << 4（Swin 小时级/epoch）。
    • 数据准备：3 需 UM 缓存；2/3/4 fast 需视频 .pt 缓存。

  方法互补关系：
    • 实验 2 = 同骨干 + 训练/决策策略
    • 实验 3 = 同骨干 + 输入增强
    • 实验 4 = 换骨干 + SSL 预训练（可与 2/3 的思想组合，但本项目尚未实现）


五、产生上述结果的结构性原因
--------------------------------------------------------------------------------

  实验 1/2/3 共享：
    Video [B,1,96,128,128]
      → ResNet3D encoder (MONAI, layers [4,4,4,4], 64-512)
      → 5D feature map
      → Top-K temporal pooling (k=50% frames, L2 重要性)
      → FC(512→256→1) + BCE

  实验 2 额外：
      阶段一: Mixup 软标签
      阶段二: 冻结 layer1-4，Logit z' = z - τ·log(π1/π0)

  实验 3 额外：
      缓存 tensor 后 apply_unsharp_masking(strength=1.5)

  实验 4：
    Video [B,1,96,128,128]
      → SwinUNETR.swinViT (feature_size=48, SSL weights)
      → 末层特征 [B, 768, d,h,w]
      → ClassificationHead (avg/topk, 768→384→1)
      → BCE（无 Mixup / 无 LogitAdj）


六、四模型架构与训练参数明细
--------------------------------------------------------------------------------

【实验 1 — 基础 ResNet3D】

  网络：
    • Lightning：ModelModule
    • Net：ResNet3DClassifier
        - Encoder：MONAI ResNet3D, BasicBlock, layers=[4,4,4,4],
          block_inplanes=[64,128,256,512], in_channels=1, feed_forward=False
        - Head：ClassificationHead(input_dim=512, hidden=256, num_classes=1,
          pooling=topk, topk_ratio=0.5)
    • 损失：BCEWithLogitsLoss
    • 决策：sigmoid(logit) > 0.5

  训练参数（configs/experiment/non_rd_vs_rd/resnet3d.yaml + configs/model/resnet3d.yaml）：
    • Optimizer：Adam, lr=1.5e-5, weight_decay=0
    • Scheduler：ReduceLROnPlateau, mode=min, factor=0.1, patience=10
    • Trainer：max_epochs=50（该次 run 实际保存 epoch 15 best）, precision=16-mixed, GPU×1
    • Data：erdes, batch_size=24（在线 mp4）或后续缓存路径 batch 4
    • Callback：ModelCheckpoint monitor=val/acc, save_top_k=1

  典型 run：logs/train/runs/rd/resnet3d/2026-05-29_22-02-10/


【实验 2 — 二阶段解耦 ResNet3D】

  网络：
    • Lightning：DecoupledModelModule（同 ResNet3DClassifier 骨干+头）
    • 先验：ClassPriorEstimator.from_csv → π_nonRD≈0.9065, π_RD≈0.0935
    • LogitAdjuster：τ=1.0，阶段二 enable

  训练参数（resnet3d_decoupled_fast.yaml + resnet3d_decoupled.yaml）：
    • Optimizer：Adam, lr=1.5e-5（阶段一）；阶段二重建 Adam 仅 head，lr=1.5e-5
    • Scheduler：ReduceLROnPlateau, mode=max, patience=8
    • total_epochs=15, stage1_end_epoch=11, stage2_start_epoch=11
    • mixup_enabled=true, mixup_alpha=0.2
    • Trainer：max_epochs=15, gradient_clip_val=1.0, precision=16-mixed
    • Data：erdes_cached, batch_size=4, num_workers=0, cache D:
    • Callback：checkpoint monitor=val/f1；PRThresholdSearchCallback(min_precision=0.5)

  典型 run：logs/train/runs/rd/resnet3d_decoupled_fast/2026-06-02_11-02-26/


【实验 3 — UM + ResNet3D】

  网络：与实验 1 相同（ModelModule + ResNet3DClassifier topk）

  数据增强（唯一结构差异在输入）：
    • apply_unsharp_masking：strength=1.5, ksize=(5,5), sigma=1.0
    • Train/Val：use_um=true；Test：use_um=false（datamodule 默认）
    • Fast 缓存：D:/ERDES/cache/non_rd_vs_rd_um（离线 UM 后存 .pt）

  训练参数（resnet3d_um_fast.yaml）：
    • Optimizer：Adam, lr=1.5e-5
    • Scheduler：ReduceLROnPlateau, mode=min, factor=0.1, patience=10
    • Trainer：max_epochs=15, precision=16-mixed
    • Data：erdes_um_cached, batch_size=4, num_workers=0
    • Callback：ModelCheckpoint monitor=val/acc, filename=resnet3d_um_fast_epoch_{epoch}

  典型 run：logs/train/runs/rd/resnet3d_um_fast/2026-06-02_20-01-30/


【实验 4 — MONAI SSL Swin-UNETR】

  网络：
    • Lightning：ModelModule
    • Net：MonaiSSLSwinUNETRClassifier
        - Backbone：MONAI SwinUNETR(in=1, out_channels=2, feature_size=48,
          norm_name=instance, normalize=True, use_checkpoint=可配置)
        - 权重：weights/ssl_pretrained_weights.pth（copy_model_state, 157 tensors）
        - encode：backbone.swinViT → hidden_states[-1]
        - Head：ClassificationHead(input_dim=768, hidden=384, num_classes=1,
          pooling=avg, topk_ratio=0.5)
    • 损失：BCEWithLogitsLoss

  训练参数（swinunetr_monai_ssl_fast_2ep.yaml，当前用户选用 2 epoch）：
    • Optimizer：Adam, lr=1e-4, weight_decay=0
    • Scheduler：ReduceLROnPlateau, mode=min, factor=0.1, patience=10
    • Trainer：max_epochs=2, min_epochs=2, precision=16-mixed
    • Data：erdes_cached, batch_size=2, use_checkpoint=false, num_workers=0
    • Callback：
        - 每 epoch：encoder_epoch_{epoch}.pth（仅 backbone）
        - 每 epoch：swinunetr_monai_ssl_epoch_{epoch}.ckpt（全量）
        - save_last.ckpt
    • 启动：scripts/start_swinunetr_monai_ssl_fast_2ep_train.ps1

  硬件注意：8GB 显存建议 bs=2、关闭 use_checkpoint；单 epoch 仍可能数小时。


七、复现与评估命令速查
--------------------------------------------------------------------------------

  # 实验 1 训练
  python erdes/train.py experiment=non_rd_vs_rd/resnet3d

  # 实验 2 快速解耦
  powershell -File scripts/start_decoupled_train_fast.ps1

  # 实验 3 UM 快速
  powershell -File scripts/start_resnet3d_um_fast_train.ps1

  # 实验 4 Swin 2 epoch
  powershell -File scripts/start_swinunetr_monai_ssl_fast_2ep_train.ps1

  # 指标图与 val 复评（实验 2、3）
  python scripts/plot_decoupled_fast_results.py
  python scripts/plot_resnet3d_um_fast_results.py

  # 实验 1 曲线
  python scripts/plot_training_results.py --run-dir logs/train/runs/rd/resnet3d/2026-05-29_22-02-10


八、后续计划
--------------------------------------------------------------------------------

  1. 实验 4 若继续：至少 5–8 epoch + val 上 PR 阈值；或与 ResNet 对齐 monitor=val/f1。
  2. 可尝试「解耦 + UM」或「Swin + 小头微调多 epoch」组合，但需额外算力预算。
  3. 统一报告：阈值 0.5 + PR 约束下最优阈值，并给出 test 集一次最终评估。
  4. 修复 decoupled 结束时的 float16/float32 与 PR 回调，避免 15 epoch 白跑。
