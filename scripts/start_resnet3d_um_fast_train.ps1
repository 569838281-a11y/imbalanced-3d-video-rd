# Fast UM training: offline UM cache on D:, no per-batch OpenCV.
$srcCache = "D:\ERDES\cache\non_rd_vs_rd"
$umCache = "D:\ERDES\cache\non_rd_vs_rd_um"
$needUm = 4300

$srcCount = (Get-ChildItem "$srcCache\*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($srcCount -lt 4300) {
    Write-Host "Base video cache incomplete ($srcCount / ~4304). Run:"
    Write-Host "  python scripts/build_video_cache.py --split all --workers 4"
    exit 1
}

$umCount = (Get-ChildItem "$umCache\*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($umCount -lt $needUm) {
    Write-Host "UM cache incomplete ($umCount / ~$needUm). Building offline UM (train+val)..."
    & D:\anaconda3\envs\erdes\python.exe scripts/build_um_cache.py --split train_val --workers 4 --strength 1.5
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $umCount = (Get-ChildItem "$umCache\*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($umCount -lt $needUm) {
        Write-Host "UM cache still incomplete ($umCount). Aborting."
        exit 1
    }
}

New-Item -ItemType Directory -Force -Path "D:\ERDES\temp", "D:\ERDES\logs" | Out-Null
Set-Location "C:\Users\ADMIN\Desktop\test_5"
$env:TMP = "D:\ERDES\temp"
$env:TEMP = "D:\ERDES\temp"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$ErrorActionPreference = "Continue"

Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine -match "resnet3d_um"
} | Stop-Process -Force -ErrorAction SilentlyContinue

& D:\anaconda3\envs\erdes\python.exe -u erdes/train.py `
  experiment=non_rd_vs_rd/resnet3d_um_fast `
  tags=[erdes,resnet3d,um,fast,15ep] `
  data.batch_size=4 `
  data.num_workers=0 `
  callbacks.rich_progress_bar=null `
  trainer.enable_progress_bar=false `
  trainer.log_every_n_steps=10 `
  2>&1 | Tee-Object -FilePath "D:\ERDES\logs\resnet3d_um_fast_train.log"
exit $LASTEXITCODE
