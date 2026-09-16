# Baseline ResNet3D + Unsharp Masking, 15 epochs
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path "D:\ERDES\temp", "D:\ERDES\logs" | Out-Null
Set-Location "C:\Users\ADMIN\Desktop\test_5"
$env:TMP = "D:\ERDES\temp"
$env:TEMP = "D:\ERDES\temp"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"

& D:\anaconda3\envs\erdes\python.exe -u erdes/train.py `
  experiment=non_rd_vs_rd/resnet3d_um `
  tags=[erdes,resnet3d,um,15ep] `
  data.use_um=true `
  data.um_strength=1.5 `
  trainer.max_epochs=15 `
  trainer.min_epochs=15 `
  callbacks.rich_progress_bar=null `
  trainer.log_every_n_steps=10 `
  2>&1 | Tee-Object -FilePath "D:\ERDES\logs\resnet3d_um_train.log"
exit $LASTEXITCODE
