# Runs the agent until the market closes. Output also goes to logs\YYYY-MM-DD.log
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = "logs\{0}.log" -f (Get-Date -Format "yyyy-MM-dd")
.\venv\Scripts\python.exe -m agent loop 2>&1 | Tee-Object -FilePath $log -Append
