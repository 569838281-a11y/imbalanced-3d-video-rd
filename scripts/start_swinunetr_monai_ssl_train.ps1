# MONAI SSL Swin-UNETR — 15 epoch supervised fine-tune (no UM)
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path "D:\ERDES\temp", "D:\ERDES\logs", "C:\Users\ADMIN\Desktop\test_5\weights" | Out-Null
Set-Location "C:\Users\ADMIN\Desktop\test_5"
$env:TMP = "D:\ERDES\temp"
$env:TEMP = "D:\ERDES\temp"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

& D:\anaconda3\envs\erdes\python.exe -u erdes/train.py `
  experiment=non_rd_vs_rd/swinunetr_monai_ssl `
  tags=[erdes,swinunetr,monai_ssl,15ep] `
  data.use_um=false `
  data.batch_size=2 `
  data.num_workers=0 `
  model.optimizer.lr=1e-4 `
  trainer.max_epochs=15 `
  trainer.min_epochs=15 `
  callbacks.rich_progress_bar=null `
  trainer.log_every_n_steps=10 `
  callbacks.model_checkpoint.save_top_k=-1 `
  callbacks.model_checkpoint.every_n_epochs=1 `
  2>&1 | Tee-Object -FilePath "D:\ERDES\logs\swinunetr_monai_ssl_train.log"
exit $LASTEXITCODE
