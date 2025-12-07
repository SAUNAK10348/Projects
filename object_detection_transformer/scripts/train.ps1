param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..")

. ..\.venv\Scripts\Activate.ps1
python -m src.train @Args

Pop-Location
