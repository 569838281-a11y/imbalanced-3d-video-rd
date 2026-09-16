# Single decoupled training job (15 epochs): unbuffered logs + GPU preprocess + D: temp/logs
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "D:\ERDES\temp", "D:\ERDES\logs" | Out-Null
Set-Location "C:\Users\ADMIN\Desktop\test_5"
$env:TMP = "D:\ERDES\temp"
$env:TEMP = "D:\ERDES\temp"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

D:\anaconda3\envs\erdes\python.exe -u erdes/train.py `
  experiment=non_rd_vs_rd/resnet3d_decoupled `
  tags=[erdes,decoupled,gpu_v3] `
  data.batch_size=4 `
  data.num_workers=2 `
  model.gpu_preprocess=true `
  callbacks.rich_progress_bar=null `
  2>&1 | Tee-Object -FilePath "D:\ERDES\logs\decoupled_train_live.log"
