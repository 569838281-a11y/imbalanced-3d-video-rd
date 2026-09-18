# Run both Stage-3 Swin SSL-protect jobs sequentially
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ADMIN\Desktop\test_5"
Set-Location $repo

Write-Host "=== Stage-3a Swin SSL-protect: Intact TD vs ND ==="
& "$repo\scripts\start_stage3_intact_swin_ssl_protect_50ep_local.ps1" @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Stage-3b Swin SSL-protect: Detached TD vs Bilateral ==="
& "$repo\scripts\start_stage3_detached_swin_ssl_protect_50ep_local.ps1" @args
exit $LASTEXITCODE
