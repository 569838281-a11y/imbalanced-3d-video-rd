# Local stage-2 macula UNet3D 50ep: resume from last.ckpt, early stopping on val/f1
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ADMIN\Desktop\test_5"
$ckptDir = "D:\ERDES\checkpoints\macula_unet3d_50ep_local"
$cache = "D:\ERDES\cache\macula_detached_vs_intact"
$py = "D:\anaconda3\envs\erdes\python.exe"
$logFile = "D:\ERDES\logs\macula_unet3d_50ep_train.log"

New-Item -ItemType Directory -Force -Path $ckptDir, "D:\ERDES\logs", "D:\ERDES\temp" | Out-Null

$cacheCount = (Get-ChildItem "$cache\*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "Macula cache: $cacheCount .pt"
if ($cacheCount -lt 400) {
    Write-Host "Cache incomplete. Building..."
    & $py -u "$repo\scripts\build_macula_cache.py"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$lastCkpt = Join-Path $ckptDir "last.ckpt"
$ckptArg = "ckpt_path=null"
if ($args -contains "-Fresh") {
    $ckptArg = "ckpt_path=null"
    Write-Host "Forced FRESH start (-Fresh)"
} elseif (Test-Path $lastCkpt) {
    $ckptUnix = ($lastCkpt -replace "\\", "/")
    $ckptArg = "ckpt_path=$ckptUnix"
    Write-Host "RESUME from $lastCkpt"
} else {
    Write-Host "FRESH start (no last.ckpt)"
}

Set-Location $repo
$env:TMP = "D:\ERDES\temp"
$env:TEMP = "D:\ERDES\temp"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try { chcp 65001 | Out-Null } catch {}

Write-Host "Checkpoints -> $ckptDir (every 5 ep + last + best; early stop patience=12 on val/f1)"

& $py -u erdes/train.py `
  experiment=macula_detached_vs_intact/unet3d_50ep_local `
  $ckptArg `
  extras.print_config=false `
  test=false `
  'callbacks.model_summary=null' `
  tags=[erdes,unet3d,macula,50ep,local] `
  2>&1 | Tee-Object -FilePath $logFile

exit $LASTEXITCODE
