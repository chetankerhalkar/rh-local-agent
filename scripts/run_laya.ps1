# Starts the Laya decision server (engine: laya). Leave this window open while the agent runs.
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_laya.ps1
# Uses the GPU when PyTorch can see it, the CPU otherwise. The first start downloads the model (about 2 GB).

$exe = (Get-Command laya-serve -ErrorAction SilentlyContinue).Source
if (-not $exe) {
    # pip install --user puts it here, a folder Windows does not search by itself
    $exe = Get-ChildItem "$env:APPDATA\Python\Python3*\Scripts\laya-serve.exe" -ErrorAction SilentlyContinue |
        Select-Object -Last 1 -ExpandProperty FullName
}
if (-not $exe) {
    Write-Host 'laya-serve not found. Install it:  python -m pip install "laya[serve]"'
    exit 1
}

$cuda = (python -c "import torch; print(torch.cuda.is_available())" 2>$null) -eq "True"
$env:LAYA_DEVICE = if ($cuda) { "cuda" } else { "cpu" }
$env:LAYA_PRELOAD = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Write-Host "Starting Laya on $($env:LAYA_DEVICE.ToUpper()) at http://localhost:8000  (Ctrl+C stops it)"
if (-not $cuda) { Write-Host "PyTorch cannot see a GPU (CPU-only build). CPU works, just slower. README: Decision engine." }
& $exe
