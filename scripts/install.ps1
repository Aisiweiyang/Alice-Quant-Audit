param([switch]$SkipTests)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
    }
    $python = Join-Path $projectRoot '.venv\Scripts\python.exe'
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    & $python examples/commercial_demo/generate_demo_data.py
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic data generation failed' }
    & $python scripts/doctor.py
    if ($LASTEXITCODE -ne 0) { throw 'Project check failed' }
    if (-not $SkipTests) {
        & $python -m unittest discover -s tests
        if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
    }
    Write-Host 'Ready. Start the local UI: .venv\Scripts\python.exe commercial_web.py'
} finally { Pop-Location }
