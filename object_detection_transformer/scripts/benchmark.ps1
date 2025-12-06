param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [Parameter()][string]$Device = "cpu",
    [Parameter()][int]$Iterations = 20,
    [Parameter()][switch]$Quantize
)

$argsList = @($Checkpoint, $DataDir, "--device", $Device, "--iterations", $Iterations)
if ($Quantize) { $argsList += "--quantize" }

python -m src.benchmark @argsList
