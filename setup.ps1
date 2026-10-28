# Skyy voice test kit - one-time setup for Windows 11
# Run from this folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$PyVersion = "3.13"
$venvPy = ".\.venv\Scripts\python.exe"

Write-Host "`n[1/5] Checking Python $PyVersion..." -ForegroundColor Cyan
$pyExe = $null
# Try the py launcher first
try { $pyExe = (& py "-$PyVersion" -c "import sys; print(sys.executable)") 2>$null } catch {}
# Fall back to whatever `python` is on PATH, if it is the right version
if (-not $pyExe) {
    try {
        $v = (& python -c "import sys; print('{0}.{1}'.format(*sys.version_info[:2]))" 2>$null)
        if ($v -and "$v".Trim() -eq $PyVersion) {
            $pyExe = (& python -c "import sys; print(sys.executable)" 2>$null)
        }
    } catch {}
}
if (-not $pyExe) {
    Write-Host "Python $PyVersion not found. Installing with winget..."
    winget install -e --id "Python.Python.$PyVersion" --accept-source-agreements --accept-package-agreements
    Write-Host "Python installed. Close this window, open a NEW PowerShell, and run setup.ps1 again." -ForegroundColor Yellow
    exit 1
}
Write-Host "Using $pyExe"

Write-Host "`n[2/5] Creating virtual environment .venv ..." -ForegroundColor Cyan
# Rebuild .venv if it was made with a different Python
if (Test-Path .venv) {
    $venvVer = ""
    if (Test-Path $venvPy) {
        $venvVer = "" + (& $venvPy -c "import sys; print('{0}.{1}'.format(*sys.version_info[:2]))" 2>$null)
    }
    if ($venvVer.Trim() -ne $PyVersion) {
        Write-Host "Existing .venv is not Python $PyVersion; removing and rebuilding..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force .venv
    }
}
if (-not (Test-Path .venv)) { & $pyExe -m venv .venv }

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