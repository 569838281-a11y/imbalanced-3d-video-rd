# Fast decoupled training (requires D: video cache). Target ~10-20 min/epoch.
$cacheRoot = "D:\ERDES\cache\non_rd_vs_rd"
$trainCount = (Get-ChildItem "$cacheRoot\*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($trainCount -lt 4300) {
    Write-Host "Cache incomplete ($trainCount / ~4304 .pt). Run:"
    Write-Host "  python scripts/build_video_cache.py --split train --workers 2"
    Write-Host "  python scripts/build_video_cache.py --split val --workers 2"
    exit 1
}

New-Item -ItemType Directory -Force -Path "D:\ERDES\temp","D:\ERDES\logs" | Out-Null
Set-Location "C:\Users\ADMIN\Desktop\test_5"
$env:TMP = "D:\ERDES\temp"
$env:TEMP = "D:\ERDES\temp"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$ErrorActionPreference = "Continue"
& D:\anaconda3\envs\erdes\python.exe -u erdes/train.py `
  experiment=non_rd_vs_rd/resnet3d_decoupled_fast `
  tags=[erdes,decoupled,fast,15ep] `
  data.batch_size=4 `
  data.num_workers=0 `
  model.gpu_preprocess=false `
  model.compute_train_metrics=false `
  callbacks.rich_progress_bar=null `
  2>&1 | Tee-Object -FilePath "D:\ERDES\logs\decoupled_fast_train.log"
exit $LASTEXITCODE
