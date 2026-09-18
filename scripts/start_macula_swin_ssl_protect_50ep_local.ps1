# Stage-2: Macula detached vs intact — Swin-UNETR + SSL, freeze-then-finetune
# Best ckpt only after unfreeze (val/f1_finetune). Use -Fresh to ignore last.ckpt.
# Production cascade currently uses macula_unet3d_best; skip Swin retrain unless needed:
#   delete SKIP_MACULA_SWIN_SSL_PROTECT in repo root, or pass nothing and remove that file.
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ADMIN\Desktop\test_5"
$skipFlag = Join-Path $repo "SKIP_MACULA_SWIN_SSL_PROTECT"
if (($args -contains "-Skip") -or (Test-Path $skipFlag)) {
  Write-Host "SKIP Stage-2 Swin SSL-protect — keep existing macula_unet3d_best.ckpt"
  if (Test-Path $skipFlag) { Write-Host "  (flag: $skipFlag)" }
  exit 0
}
$ckptDir = "D:\ERDES\checkpoints\macula_swin_ssl_protect_50ep"
$py = "D:\anaconda3\envs\erdes\python.exe"
$logFile = "D:\ERDES\logs\macula_swin_ssl_protect_train.log"
$sslWeights = "$repo\weights\ssl_pretrained_weights.pth"

New-Item -ItemType Directory -Force -Path $ckptDir, "D:\ERDES\logs", "D:\ERDES\temp" | Out-Null

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

Write-Host "Stage-2 Swin SSL-protect: Macula detached vs intact"
Write-Host "  freeze_epochs=15 -> unfreeze; best/early-stop on val/f1_finetune only"
Write-Host "  ckpts -> $ckptDir"

& $py -u erdes/train.py `
  experiment=macula_detached_vs_intact/swinunetr_ssl_protect_50ep_local `
  $ckptArg `
  extras.print_config=false `
  test=false `
  'callbacks.model_summary=null' `
  tags=[erdes,swin,ssl_protect,stage2,macula,50ep] `
  2>&1 | Tee-Object -FilePath $logFile

exit $LASTEXITCODE
