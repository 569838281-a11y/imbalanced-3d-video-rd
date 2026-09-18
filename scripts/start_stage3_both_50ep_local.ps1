# Run both Stage-3 branches sequentially (intact then detached)
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ADMIN\Desktop\test_5"
Set-Location $repo

Write-Host "=== Stage-3a: Macula Intact TD vs ND ==="
& "$repo\scripts\start_stage3_intact_td_vs_nd_50ep_local.ps1" @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Stage-3b: Macula Detached TD vs Bilateral ==="
& "$repo\scripts\start_stage3_detached_td_vs_bilateral_50ep_local.ps1" @args
exit $LASTEXITCODE
