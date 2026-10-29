param([string]$Python = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $projectRoot '.venv\Scripts\python.exe' }
Push-Location $projectRoot
try {
    & $Python examples/commercial_demo/generate_demo_data.py
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic data generation failed' }
    foreach ($strategy in @('classic_trend', 'cross_sectional_alpha158')) {
        & $Python commercial_mvp.py full --cache examples/commercial_demo/demo_cache.pkl --start 2025-07-01 --end 2025-12-31 --strategy $strategy
        if ($LASTEXITCODE -ne 0) { throw "Demo failed: $strategy" }
    }
    Write-Host 'SYNTHETIC demonstration only. Reports: output/commercial_reports'
} finally { Pop-Location }
