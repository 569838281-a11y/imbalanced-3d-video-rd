# Fast MONAI SSL Swin-UNETR fine-tune (D: cache, bs=4, per-epoch encoder .pth)
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

& D:\anaconda3\envs\erdes\python.exe -u erdes/train.py `
  experiment=non_rd_vs_rd/swinunetr_monai_ssl_fast `
  tags=[erdes,swinunetr,monai_ssl,fast,15ep] `
  data.use_um=false `
  data.use_cache=true `
  data.batch_size=4 `
  data.num_workers=0 `
  model.optimizer.lr=1e-4 `
  model.net.use_checkpoint=true `
  trainer.max_epochs=15 `
  trainer.min_epochs=15 `
  callbacks.rich_progress_bar=null `
  trainer.log_every_n_steps=10 `
  2>&1 | Tee-Object -FilePath "D:\ERDES\logs\swinunetr_monai_ssl_fast_train.log"
exit $LASTEXITCODE
