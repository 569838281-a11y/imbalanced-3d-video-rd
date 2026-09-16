# Launch local AI Doctor (cascade RD → macula) at http://127.0.0.1:7860
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ADMIN\Desktop\test_5"
$py = "D:\anaconda3\envs\erdes\python.exe"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:AI_DOCTOR_RD_CKPT = "$repo\weights\um_decoupled_best.ckpt"
$env:AI_DOCTOR_MACULA_CKPT = "$repo\weights\macula_unet3d_best.ckpt"
if (-not (Test-Path $env:AI_DOCTOR_RD_CKPT)) {
  $env:AI_DOCTOR_RD_CKPT = "D:\ERDES\checkpoints\um_decoupled_50ep_local\um_decoupled_best.ckpt"
}
if (-not (Test-Path $env:AI_DOCTOR_MACULA_CKPT)) {
  $env:AI_DOCTOR_MACULA_CKPT = "D:\ERDES\checkpoints\macula_unet3d_50ep_local\macula_unet3d_best.ckpt"
}

Set-Location $repo
Write-Host "Starting ERDES AI Doctor on http://127.0.0.1:7860"
& $py -m ai_doctor.app
