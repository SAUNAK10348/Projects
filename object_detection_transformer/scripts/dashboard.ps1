param(
    [switch]$Headless
)

$ErrorActionPreference = 'Stop'
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

if (-not (Test-Path .venv)) {
    Write-Error "Virtualenv not found. Run scripts/setup_workspace.ps1 first."
}

. ./.venv/Scripts/Activate.ps1
$headlessFlag = $Headless.IsPresent ? '--server.headless true' : ''
streamlit run src/dashboard.py $headlessFlag
