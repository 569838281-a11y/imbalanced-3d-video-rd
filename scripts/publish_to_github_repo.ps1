# Publish ERDES RD optimization package to a target directory (for GitHub push).
param(
    [string]$TargetDir = "C:\Users\ADMIN\Desktop\569838281-a11y"
)

$Root = "C:\Users\ADMIN\Desktop\test_5"
$ErrorActionPreference = "Stop"

if (Test-Path $TargetDir) {
    Remove-Item $TargetDir -Recurse -Force
}
New-Item -ItemType Directory -Path $TargetDir | Out-Null

function Copy-Tree($Rel) {
    $src = Join-Path $Root $Rel
    $dst = Join-Path $TargetDir $Rel
    if (Test-Path $src) {
        $parent = Split-Path $dst -Parent
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        Copy-Item $src $dst -Recurse -Force
    }
}

# Core package
Copy-Tree "erdes"
if (Test-Path "$TargetDir\erdes\logs") { Remove-Item "$TargetDir\erdes\logs" -Recurse -Force }

# Configs (minimal runnable set)
$configItems = @(
    "configs\train.yaml",
    "configs\paths",
    "configs\hydra",
    "configs\trainer",
    "configs\extras",
    "configs\callbacks",
    "configs\logger\tensorboard.yaml",
    "configs\data\erdes.yaml",
    "configs\data\erdes_cached.yaml",
    "configs\data\erdes_um.yaml",
    "configs\data\erdes_um_cached.yaml",
    "configs\model\resnet3d.yaml",
    "configs\model\resnet3d_decoupled.yaml",
    "configs\model\swinunetr_monai_ssl.yaml",
    "configs\experiment\non_rd_vs_rd\resnet3d.yaml",
    "configs\experiment\non_rd_vs_rd\resnet3d_decoupled.yaml",
    "configs\experiment\non_rd_vs_rd\resnet3d_decoupled_fast.yaml",
    "configs\experiment\non_rd_vs_rd\resnet3d_um.yaml",
    "configs\experiment\non_rd_vs_rd\resnet3d_um_fast.yaml",
    "configs\experiment\non_rd_vs_rd\swinunetr_monai_ssl.yaml",
    "configs\experiment\non_rd_vs_rd\swinunetr_monai_ssl_fast.yaml",
    "configs\experiment\non_rd_vs_rd\swinunetr_monai_ssl_fast_2ep.yaml"
)
foreach ($item in $configItems) {
    $src = Join-Path $Root $item
    $dst = Join-Path $TargetDir $item
    if (Test-Path $src) {
        New-Item -ItemType Directory -Path (Split-Path $dst -Parent) -Force | Out-Null
        Copy-Item $src $dst -Recurse -Force
    }
}

# Data splits (paths + labels only)
Copy-Tree "data\splits\non_rd_vs_rd"

# Scripts
$scripts = @(
    "build_video_cache.py",
    "build_um_cache.py",
    "start_decoupled_train_fast.ps1",
    "start_resnet3d_um_fast_train.ps1",
    "start_swinunetr_monai_ssl_fast_2ep_train.ps1",
    "plot_decoupled_fast_results.py",
    "plot_resnet3d_um_fast_results.py",
    "plot_swinunetr_monai_ssl_fast_2ep_results.py",
    "plot_training_results.py"
)
New-Item -ItemType Directory -Path (Join-Path $TargetDir "scripts") -Force | Out-Null
foreach ($s in $scripts) {
    Copy-Item (Join-Path $Root "scripts\$s") (Join-Path $TargetDir "scripts\$s") -Force
}

# Root files
Copy-Item (Join-Path $Root "requirements.txt") $TargetDir -Force
Copy-Item (Join-Path $Root ".project-root") $TargetDir -Force
if (Test-Path (Join-Path $Root "final_summary.txt")) {
    New-Item -ItemType Directory -Path (Join-Path $TargetDir "docs") -Force | Out-Null
    Copy-Item (Join-Path $Root "final_summary.txt") (Join-Path $TargetDir "docs\EXPERIMENTS.md") -Force
}

Write-Host "Package copied to $TargetDir"
