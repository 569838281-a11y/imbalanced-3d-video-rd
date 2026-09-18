# Stage-3a local: Macula Intact → TD vs ND (UNet3D 50ep, resume + early stop)
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ADMIN\Desktop\test_5"
$ckptDir = "D:\ERDES\checkpoints\stage3_intact_td_vs_nd_50ep_local"
$py = "D:\anaconda3\envs\erdes\python.exe"
$logFile = "D:\ERDES\logs\stage3_intact_td_vs_nd_train.log"

New-Item -ItemType Directory -Force -Path $ckptDir, "D:\ERDES\logs", "D:\ERDES\temp" | Out-Null

# Ensure splits exist
if (-not (Test-Path "$repo\data\splits\macula_intact_td_vs_nd\train.csv")) {
  & $py -u "$repo\scripts\build_stage3_splits.py"
}

$lastCkpt = Join-Path $ckptDir "last.ckpt"
$ckptArg = "ckpt_path=null"
if ($args -contains "-Fresh") {
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

Write-Host "Stage-3a Macula_Intact: TD vs ND | ckpts -> $ckptDir"
Write-Host "WARN: single-patient-per-class in current data — prototype metrics only."

& $py -u erdes/train.py `
  experiment=macula_intact_td_vs_nd/unet3d_50ep_local `
  $ckptArg `
  extras.print_config=false `
  test=false `
  'callbacks.model_summary=null' `
  tags=[erdes,unet3d,stage3,intact,td_vs_nd,50ep,local] `
  2>&1 | Tee-Object -FilePath $logFile

exit $LASTEXITCODE
