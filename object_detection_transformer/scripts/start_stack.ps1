Param()

$Root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Root = Join-Path $Root ".."
Set-Location $Root

if (-not (Test-Path ".venv")) {
    Write-Error "Virtualenv not found. Run scripts/setup_workspace.ps1 first."
    exit 1
}

. .\.venv\Scripts\Activate.ps1

python -m src.stack
