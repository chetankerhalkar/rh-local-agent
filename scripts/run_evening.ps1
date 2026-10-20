# Unattended two-session style, part 2: cancel the stops, sell everything, check nothing is left.
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = "logs\{0}.log" -f (Get-Date -Format "yyyy-MM-dd")
.\venv\Scripts\python.exe -m agent auto-evening 2>&1 | Tee-Object -FilePath $log -Append
.\venv\Scripts\python.exe -m agent report  2>&1 | Tee-Object -FilePath $log -Append
