# One-time setup. Run from the project folder:
#     powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".\agent\__main__.py")) {
    Write-Host "Wrong folder. Run this from the folder that contains 'agent', 'scripts' and 'requirements.txt'." -ForegroundColor Red
    Write-Host "Tip: Windows 'Extract All' often makes rh-local-agent\rh-local-agent. Go one level deeper." -ForegroundColor Yellow
    exit 1
}

# Find a real Python (the Microsoft Store stub prints a message and exits with an error)
$py = $null
foreach ($cmd in @("python", "py")) {
    try { $v = & $cmd --version 2>&1; if ($LASTEXITCODE -eq 0 -and "$v" -match "Python 3\.(\d+)") { if ([int]$Matches[1] -ge 10) { $py = $cmd; break } } } catch {}
}
if (-not $py) {
    Write-Host "Python 3.10 or newer was not found." -ForegroundColor Red
    Write-Host "Install it, then open a NEW PowerShell window and run this script again:" -ForegroundColor Yellow
    Write-Host "    winget install Python.Python.3.12"
    exit 1
}
Write-Host "Using: $(& $py --version)" -ForegroundColor Cyan

if (-not (Test-Path ".\venv\Scripts\python.exe")) { & $py -m venv venv }
.\venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.\venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path config.yaml)) {
    Copy-Item config.example.yaml config.yaml
    Write-Host "Created config.yaml from the example." -ForegroundColor Yellow
}

Write-Host "`nRunning the safety tests (about 30 seconds)..." -ForegroundColor Cyan
.\venv\Scripts\python.exe -m pytest -q
Write-Host "`nNext:  .\venv\Scripts\python.exe -m agent doctor" -ForegroundColor Green
