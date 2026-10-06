<# Build/train/evaluate the complete VisDrone person-candidate research path. #>
[CmdletBinding()]
param(
    [ValidateRange(1, 300)][int]$Epochs = 20,
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
    throw 'Project .venv is missing. Complete the environment setup first.'
}
$pipelineArguments = @(
    '-m', 'nidar_survivor_demo.full_visdrone_pipeline',
    '--epochs', "$Epochs", '--batch', "$Batch", '--workers', "$Workers", '--device', $Device
)
if ($PrepareOnly) { $pipelineArguments += '--prepare-only' }
if ($Resume) { $pipelineArguments += '--resume-training' }
if ($Background) {
    $logDirectory = Join-Path $PSScriptRoot 'logs'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $stdout = Join-Path $logDirectory 'full-visdrone-training.stdout.log'
    $stderr = Join-Path $logDirectory 'full-visdrone-training.stderr.log'
    if (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*nidar_survivor_demo.full_visdrone_pipeline*' }) {
        throw 'A full VisDrone pipeline is already running.'
    }
    $previousUnbuffered = $env:PYTHONUNBUFFERED
    try {
        $env:PYTHONUNBUFFERED = '1'
        $process = Start-Process -FilePath $pythonPath -ArgumentList $pipelineArguments `
            -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    } finally {
        $env:PYTHONUNBUFFERED = $previousUnbuffered
    }
    Write-Host "Full VisDrone pipeline started in background. PID: $($process.Id)"
    Write-Host "Progress: Get-Content '$stdout' -Tail 30 -Wait"
    Write-Host "Errors:   Get-Content '$stderr' -Tail 30 -Wait"
    exit 0
}
Push-Location $PSScriptRoot
try {
    & $pythonPath @pipelineArguments
    $pipelineExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $pipelineExit
