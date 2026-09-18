# Stage-3 subtype classification (macula-gated)

## Cascade

```text
Stage-1  non-RD vs RD
Stage-2  RD → Macula Intact vs Detached
Stage-3a Macula Intact  → TD vs ND
Stage-3b Macula Detached → TD vs Bilateral
```

## Labels

| Task | CSV dir | label=0 | label=1 |
|------|---------|---------|---------|
| 3a | `data/splits/macula_intact_td_vs_nd` | ND | TD |
| 3b | `data/splits/macula_detached_td_vs_bilateral` | Bilateral | TD |

## Build splits

```bash
python scripts/build_stage3_splits.py
```

## Train (local, UNet3D 50ep, resume + early stop on val/f1)

```powershell
# Branch after macula intact
scripts/start_stage3_intact_td_vs_nd_50ep_local.ps1

# Branch after macula detached
scripts/start_stage3_detached_td_vs_bilateral_50ep_local.ps1

# Or both sequentially
scripts/start_stage3_both_50ep_local.ps1
```

## Train (recommended): Swin-UNETR + MONAI SSL, freeze-then-finetune

Loads `weights/ssl_pretrained_weights.pth`, **freezes backbone for 15 epochs** (head only),
then **unfreezes** all params with `finetune_lr=1e-5` for the remaining epochs (max 50).

**Best checkpoint / early-stop** monitor `val/f1_finetune`, which is logged **only after
unfreeze (Stage B)**. Stage-A freeze metrics cannot become `*_best.ckpt`.
`trainer.min_epochs=16` ensures training always enters Stage B before early-stop can end.

```powershell
scripts/start_stage3_intact_swin_ssl_protect_50ep_local.ps1
scripts/start_stage3_detached_swin_ssl_protect_50ep_local.ps1
# or:
scripts/start_stage3_both_swin_ssl_protect_50ep_local.ps1
```

Checkpoints:

- `D:/ERDES/checkpoints/stage3_intact_swin_ssl_protect_50ep/`
- `D:/ERDES/checkpoints/stage3_detached_swin_ssl_protect_50ep/`

Cache reuses `D:/ERDES/cache/macula_detached_vs_intact` (same videos as Stage-2).

Checkpoints:

- `D:/ERDES/checkpoints/stage3_intact_td_vs_nd_50ep_local/`
- `D:/ERDES/checkpoints/stage3_detached_td_vs_bilateral_50ep_local/`

## Important limitation

In the current folder layout, **each subtype directory maps to a single patient ID**.
Video-level stratified splits therefore leak identity. Treat Stage-3 metrics as
**prototype / template-development only** until multi-patient data is available.
Do not claim clinical generalization from these numbers alone.
