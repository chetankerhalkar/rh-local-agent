# Update the code from a newer zip WITHOUT touching your venv, config.yaml, .secrets, journal or state.
# Run from inside your project folder:
#     powershell -ExecutionPolicy Bypass -File .\scripts\update.ps1 -Zip "$HOME\Downloads\rh-local-agent-v0.6.0.zip"
param([Parameter(Mandatory = $true)][string]$Zip)
$ErrorActionPreference = "Stop"
$project = Split-Path $PSScriptRoot -Parent
$tmp = Join-Path $env:TEMP ("rh-update-" + [guid]::NewGuid().ToString("N"))
Expand-Archive -Path $Zip -DestinationPath $tmp -Force
$src = Join-Path $tmp "rh-local-agent"
if (-not (Test-Path (Join-Path $src "agent\__main__.py"))) { throw "That zip does not look like rh-local-agent." }
Copy-Item (Join-Path $src "*") -Destination $project -Recurse -Force      # the zip has no venv and no config.yaml
Remove-Item $tmp -Recurse -Force
Set-Location $project
.\venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
.\venv\Scripts\python.exe -m agent --version
