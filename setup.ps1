# Skyy voice test kit - one-time setup for Windows 11
# Run from this folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

Write-Host "`n[1/5] Checking Python 3.12..." -ForegroundColor Cyan
$py = $null
try { $py = (& py -3.12 -c "import sys; print(sys.executable)") 2>$null } catch {}
if (-not $py) {
    Write-Host "Python 3.12 not found. Installing with winget..."
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    Write-Host "Python installed. Close this window, open a NEW PowerShell, and run setup.ps1 again." -ForegroundColor Yellow
    exit 1
}
Write-Host "Using $py"

Write-Host "`n[2/5] Creating virtual environment .venv ..." -ForegroundColor Cyan
if (-not (Test-Path .venv)) { & py -3.12 -m venv .venv }
$venvPy = ".\.venv\Scripts\python.exe"

Write-Host "`n[3/5] Installing packages (a few minutes)..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed - see the error above." -ForegroundColor Red; exit 1 }

Write-Host "`n[4/5] Downloading models (~1.5 GB)..." -ForegroundColor Cyan
& $venvPy download_models.py
if ($LASTEXITCODE -ne 0) { Write-Host "Model download failed - rerun setup.ps1 to resume." -ForegroundColor Red; exit 1 }

Write-Host "`n[5/5] Checking the setup..." -ForegroundColor Cyan
& $venvPy check_setup.py

Write-Host "`nNext time, activate the environment with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
