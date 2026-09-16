# Swin-UNETR VideoMAE pretrain — progress every 10 steps
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path "D:\ERDES\logs", "C:\Users\ADMIN\Desktop\test_5\weights" | Out-Null
Set-Location "C:\Users\ADMIN\Desktop\test_5"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# Fast profile: train-only, patch-level loss, 2-layer decoder, try bs=3 if VRAM allows
& D:\anaconda3\envs\erdes\python.exe -u run_pretrain_swinunetr.py `
  --epochs 15 `
  --batch-size 2 `
  --train-only `
  --decoder-layers 2 `
  --use-checkpoint `
  --log-every 10 `
  --progress-log "D:/ERDES/logs/pretrain_swinunetr_progress.log" `
  2>&1 | Tee-Object -FilePath "D:\ERDES\logs\pretrain_swinunetr_videomae.log"
exit $LASTEXITCODE
