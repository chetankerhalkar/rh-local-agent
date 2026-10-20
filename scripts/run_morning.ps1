# Unattended two-session style, part 1 (your universe list only): decide, buy, place protective stops.
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = "logs\{0}.log" -f (Get-Date -Format "yyyy-MM-dd")
.\venv\Scripts\python.exe -m agent auto-morning 2>&1 | Tee-Object -FilePath $log -Append
