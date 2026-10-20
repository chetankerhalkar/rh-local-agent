# Creates the Windows scheduled tasks that run the agent on weekdays.
#
#   .\scripts\register_task.ps1                         continuous: one task, checks every few minutes all day
#   .\scripts\register_task.ps1 -Style two_sessions     two tasks: morning buy, evening sell
#
# TIMES ARE YOUR PC'S LOCAL TIME. The defaults assume US Central time:
#   market open 8:30 AM CT, close 3:00 PM CT
#   -Morning 08:50  = 20 min after the open (the agent refuses buys in the first 15 min)
#   -Evening 14:40  = 20 min before the close
# On another time zone pass your own times:  -Morning "09:50" -Evening "15:40"   (US Eastern)
#
# Remove everything:
#   "RH Local Agent","RH Agent Morning","RH Agent Evening" | ForEach-Object { Unregister-ScheduledTask -TaskName $_ -Confirm:$false -ErrorAction SilentlyContinue }
#
# A scheduled task has no console, so "approve" mode cannot ask you anything there. Use paper or auto.
param(
    [ValidateSet("continuous", "two_sessions")] [string]$Style = "continuous",
    [string]$At = "08:25",
    [string]$Morning = "08:50",
    [string]$Evening = "14:40"
)

$project  = Split-Path $PSScriptRoot -Parent
$days     = "Monday","Tuesday","Wednesday","Thursday","Friday"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 8)

function Add-AgentTask([string]$Name, [string]$Script, [string]$Time) {
    $action  = New-ScheduledTaskAction -Execute "powershell.exe" `
               -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$project\scripts\$Script`"" -WorkingDirectory $project
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At $Time
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
    Write-Host "Registered '$Name' at $Time (weekdays)"
}

"RH Local Agent","RH Agent Morning","RH Agent Evening" | ForEach-Object {
    Unregister-ScheduledTask -TaskName $_ -Confirm:$false -ErrorAction SilentlyContinue      # never run both styles at once
}
if ($Style -eq "two_sessions") {
    Add-AgentTask "RH Agent Morning" "run_morning.ps1" $Morning
    Add-AgentTask "RH Agent Evening" "run_evening.ps1" $Evening
} else {
    Add-AgentTask "RH Local Agent" "run_loop.ps1" $At
}
Write-Host "Tasks run only while you are logged in (needed for the GPU and Ollama)."
