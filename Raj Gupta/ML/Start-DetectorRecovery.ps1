<# Prepare, train and gate the COCO recovery candidate. #>
[CmdletBinding()]
param(
    [ValidateRange(1, 50)][int]$Epochs = 8,
    [ValidateRange(1, 32)][int]$Batch = 16,
    [ValidateRange(0, 8)][int]$Workers = 2,
    [ValidateSet('0', 'cpu')][string]$Device = '0',
    [switch]$PrepareOnly,
    [switch]$Resume,
    [switch]$Background
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'Project .venv is missing.'
}
$arguments = @(
    '-m', 'nidar_survivor_demo.recover_nidar_detector', '--epochs', "$Epochs",
    '--batch', "$Batch", '--workers', "$Workers", '--device', $Device
)
if ($PrepareOnly) { $arguments += '--prepare-only' }
if ($Resume) { $arguments += '--resume-training' }
if ($Background) {
    $logDirectory = Join-Path $PSScriptRoot 'logs'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $stdout = Join-Path $logDirectory 'detector-recovery.stdout.log'
    $stderr = Join-Path $logDirectory 'detector-recovery.stderr.log'
    if (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*nidar_survivor_demo.recover_nidar_detector*' }) {
        throw 'A detector recovery pipeline is already running.'
    }
    $old = $env:PYTHONUNBUFFERED
    try {
        $env:PYTHONUNBUFFERED = '1'
        $process = Start-Process -FilePath $pythonPath -ArgumentList $arguments `
            -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    } finally {
        $env:PYTHONUNBUFFERED = $old
    }
    Write-Host "Detector recovery started in background. PID: $($process.Id)"
    Write-Host "Progress: Get-Content '$stdout' -Tail 30 -Wait"
    Write-Host "Errors:   Get-Content '$stderr' -Tail 30 -Wait"
    exit 0
}
Push-Location $PSScriptRoot
try {
    & $pythonPath @arguments
    $pipelineExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $pipelineExit
