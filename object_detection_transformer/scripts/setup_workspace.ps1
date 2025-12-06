param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$venvPath = ".\.venv"
if ($Recreate -and (Test-Path $venvPath)) {
    Remove-Item -Recurse -Force $venvPath
}

python -m venv $venvPath
$activate = Join-Path $venvPath "Scripts" | Join-Path -ChildPath "Activate.ps1"
. $activate

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Virtual environment created. Activate with:`n.\\.venv\\Scripts\\Activate.ps1"
