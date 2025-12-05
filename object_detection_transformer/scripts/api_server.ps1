param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

if ($env:FUSIONNET_API_TOKEN) {
    Write-Host "Using token from FUSIONNET_API_TOKEN"
}

uvicorn src.api:app --host 0.0.0.0 --port $Port
