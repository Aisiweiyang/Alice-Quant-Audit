$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        throw 'Run scripts\install.ps1 first to create the local environment.'
    }
    Write-Host 'Open http://127.0.0.1:8765 in your browser. Ctrl+C stops the server.'
    & $python commercial_web.py
    if ($LASTEXITCODE -ne 0) { throw 'Workbench exited with an error.' }
} finally { Pop-Location }
