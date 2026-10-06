<# Phone-free, dashboard-free real-YOLO slideshow for professor demonstrations. #>
[CmdletBinding()]
param(
    [ValidateRange(1, 120)][int]$SceneSeconds = 5,
    [ValidateRange(0, 1000)][int]$Cycles = 0,
    [ValidateRange(1, 30)][int]$InferenceFps = 12,
    [ValidateSet('auto', 'cpu', '0')][string]$Device = '0',
    [string]$Model,
    [switch]$BaselineDetector,
    [switch]$Headless
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
$resolvedModel = $null
if ($Model) {
    $resolvedModel = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Model)
    if (-not (Test-Path -LiteralPath $resolvedModel -PathType Leaf)) { throw "Model file does not exist: $resolvedModel" }
}
$arguments = @(
    '-m', 'nidar_survivor_demo.offline_showcase',
    '--scene-seconds', $SceneSeconds,
    '--cycles', $Cycles,
    '--inference-fps', $InferenceFps,
    '--device', $Device
)
$null = & $pythonPath -m nidar_survivor_demo.setup_showcase
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare checksum-verified showcase assets.' }
if ($resolvedModel) { $arguments += @('--model', $resolvedModel) }
if (-not $BaselineDetector) { $arguments += '--hybrid-detector' }
if ($Headless) { $arguments += '--headless' }
Write-Host 'NIDAR multi-scene replay: continuous real-YOLO detection in the original camera-test layout.'
Write-Host 'Controls: N/right next, P/left previous, R rerun inference, Space pause, Q/Esc quit.'
Push-Location $PSScriptRoot
try {
    & $pythonPath @arguments
    $demoExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $demoExit
