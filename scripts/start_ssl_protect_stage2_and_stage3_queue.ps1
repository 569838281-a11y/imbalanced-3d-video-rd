# Sequential: Stage-3b → Stage-3a. Stage-2 keeps existing UNet3D macula (no Swin retrain).
# Best Stage-3 ckpts only after unfreeze (val/f1_finetune).
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ADMIN\Desktop\test_5"
Set-Location $repo

Write-Host "=== [1/2] Stage-3b detached (resume if last.ckpt) ==="
& powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\start_stage3_detached_swin_ssl_protect_50ep_local.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "Stage-3b failed: $LASTEXITCODE"; exit $LASTEXITCODE }

Write-Host "=== [2/2] Stage-3a intact (FRESH if needed) ==="
& powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\start_stage3_intact_swin_ssl_protect_50ep_local.ps1" -Fresh
if ($LASTEXITCODE -ne 0) { Write-Host "Stage-3a failed: $LASTEXITCODE"; exit $LASTEXITCODE }

Write-Host "Stage-3 done. Stage-2: keep existing macula_unet3d_best.ckpt (no Swin retrain)."
exit 0
