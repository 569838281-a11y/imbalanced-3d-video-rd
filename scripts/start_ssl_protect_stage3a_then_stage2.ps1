# After 3b done: finish Stage-3a only. Stage-2 keeps existing UNet3D macula.
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ADMIN\Desktop\test_5"
Set-Location $repo

Write-Host "=== Stage-3a intact (resume if last.ckpt) ==="
& powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\scripts\start_stage3_intact_swin_ssl_protect_50ep_local.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "Stage-3a failed: $LASTEXITCODE"; exit $LASTEXITCODE }

Write-Host "Stage-3a done. Stage-2: keep existing macula_unet3d_best.ckpt (no Swin retrain)."
exit 0
