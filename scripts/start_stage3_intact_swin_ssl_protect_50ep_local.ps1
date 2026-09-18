# Stage-3a: Macula Intact TD vs ND — Swin-UNETR + SSL, freeze-then-finetune
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ADMIN\Desktop\test_5"
$ckptDir = "D:\ERDES\checkpoints\stage3_intact_swin_ssl_protect_50ep"
$py = "D:\anaconda3\envs\erdes\python.exe"
$logFile = "D:\ERDES\logs\stage3_intact_swin_ssl_protect_train.log"
$sslWeights = "$repo\weights\ssl_pretrained_weights.pth"

New-Item -ItemType Directory -Force -Path $ckptDir, "D:\ERDES\logs", "D:\ERDES\temp" | Out-Null

if (-not (Test-Path "$repo\data\splits\macula_intact_td_vs_nd\train.csv")) {
  & $py -u "$repo\scripts\build_stage3_splits.py"
}
if (-not (Test-Path $sslWeights)) {
  Write-Host "SSL weights missing at $sslWeights — will try auto-download on start."
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

Write-Host "Stage-3a Swin SSL-protect: Intact TD vs ND"
Write-Host "  freeze_epochs=15 -> unfreeze; best/early-stop on val/f1_finetune only"
Write-Host "  ckpts -> $ckptDir"

& $py -u erdes/train.py `
  experiment=macula_intact_td_vs_nd/swinunetr_ssl_protect_50ep_local `
  $ckptArg `
  extras.print_config=false `
  test=false `
  'callbacks.model_summary=null' `
  tags=[erdes,swin,ssl_protect,stage3,intact,50ep] `
  2>&1 | Tee-Object -FilePath $logFile

exit $LASTEXITCODE
