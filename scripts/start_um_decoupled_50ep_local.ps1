# Local RD binary: UM + Decoupled 50ep, checkpoint every 5 epochs, auto-resume from last.ckpt
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ADMIN\Desktop\test_5"
$ckptDir = "D:\ERDES\checkpoints\um_decoupled_50ep_local"
$umCache = "D:\ERDES\cache\non_rd_vs_rd_um"
$baseCache = "D:\ERDES\cache\non_rd_vs_rd"
$py = "D:\anaconda3\envs\erdes\python.exe"
$logFile = "D:\ERDES\logs\um_decoupled_50ep_train.log"

New-Item -ItemType Directory -Force -Path $ckptDir, "D:\ERDES\logs", "D:\ERDES\temp" | Out-Null

$umCount = (Get-ChildItem "$umCache\*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
$baseCount = (Get-ChildItem "$baseCache\*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "UM cache: $umCount .pt | base cache: $baseCount .pt"
if ($umCount -lt 4000) {
    Write-Host "UM cache looks incomplete. Build with:"
    Write-Host "  python scripts/build_um_cache.py --split train_val --workers 4"
    exit 1
}

$lastCkpt = Join-Path $ckptDir "last.ckpt"
$ckptArg = "ckpt_path=null"
if (Test-Path $lastCkpt) {
    $ckptUnix = ($lastCkpt -replace "\\", "/")
    $ckptArg = "ckpt_path=$ckptUnix"
    Write-Host "RESUME from $lastCkpt"
} else {
    Write-Host "FRESH start (no last.ckpt in $ckptDir)"
}

# Optional: force fresh with -Fresh
if ($args -contains "-Fresh") {
    $ckptArg = "ckpt_path=null"
    Write-Host "Forced FRESH start (-Fresh)"
}

Set-Location $repo
$env:TMP = "D:\ERDES\temp"
$env:TEMP = "D:\ERDES\temp"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
# Avoid Rich/Lightning UnicodeEncodeError on Windows GBK consoles
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try { chcp 65001 | Out-Null } catch {}

Write-Host "Checkpoints every 5 epochs + last.ckpt -> $ckptDir"
Write-Host "Progress log -> D:\ERDES\logs\um_decoupled_50ep_progress.log"

& $py -u erdes/train.py `
  experiment=non_rd_vs_rd/resnet3d_um_decoupled_50ep_local `
  $ckptArg `
  extras.print_config=false `
  test=false `
  'callbacks.early_stopping=null' `
  'callbacks.model_summary=null' `
  tags=[erdes,um,decoupled,50ep,local] `
  2>&1 | Tee-Object -FilePath $logFile

exit $LASTEXITCODE
