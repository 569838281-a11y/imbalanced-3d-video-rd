# MONAI SSL Swin-UNETR: 2 epochs, save encoder + full Lightning ckpt each epoch.
$cacheRoot = "D:\ERDES\cache\non_rd_vs_rd"
$trainCount = (Get-ChildItem "$cacheRoot\*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($trainCount -lt 4300) {
    Write-Host "Cache incomplete ($trainCount / ~4304). Run:"
    Write-Host "  python scripts/build_video_cache.py --split all --workers 4"
    exit 1
}

$sslWeights = "C:\Users\ADMIN\Desktop\test_5\weights\ssl_pretrained_weights.pth"
if (-not (Test-Path $sslWeights)) {
    Write-Host "SSL weights missing. Will auto-download on first run, or place at:"
    Write-Host "  $sslWeights"
}

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path "D:\ERDES\temp", "D:\ERDES\logs" | Out-Null
Set-Location "C:\Users\ADMIN\Desktop\test_5"
$env:TMP = "D:\ERDES\temp"
$env:TEMP = "D:\ERDES\temp"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    $c = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine
    if ($c -match "train\.py") {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep 2

Write-Host "Starting 2-epoch run. Per epoch saves:"
Write-Host "  encoder_epoch_000.pth / encoder_epoch_001.pth  (Swin backbone only)"
Write-Host "  swinunetr_monai_ssl_epoch_000.ckpt / _001.ckpt  (full Lightning, all params)"
Write-Host "  last.ckpt"
Write-Host "Logs: D:\ERDES\logs\swinunetr_monai_ssl_fast_2ep_train.log"

& D:\anaconda3\envs\erdes\python.exe -u erdes/train.py `
  experiment=non_rd_vs_rd/swinunetr_monai_ssl_fast_2ep `
  2>&1 | Tee-Object -FilePath "D:\ERDES\logs\swinunetr_monai_ssl_fast_2ep_train.log"
exit $LASTEXITCODE
