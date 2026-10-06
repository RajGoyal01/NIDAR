<# Native USB/built-in webcam alternative to the wireless IP Webcam launcher. #>
[CmdletBinding(DefaultParameterSetName = 'Run')]
param(
    [Parameter(ParameterSetName = 'List', Mandatory = $true)][switch]$ListCameras,
    [Parameter(ParameterSetName = 'Run')][ValidateRange(0, 20)][int]$CameraIndex = 0,
    [Parameter(ParameterSetName = 'List')][ValidateRange(0, 20)][int]$MaxIndex = 5,
    [ValidateSet('Full', 'Detect', 'Track', 'Camera')][string]$Mode = 'Full',
    [ValidateSet('auto', 'dshow', 'msmf')][string]$Backend = 'dshow',
    [ValidateSet('auto', 'cpu', '0')][string]$Device = '0',
    [ValidateRange(160, 3840)][int]$Width = 1280,
    [ValidateRange(120, 2160)][int]$Height = 720,
    [ValidateRange(1, 120)][int]$Fps = 30,
    [ValidateRange(0, 86400)][int]$Seconds = 0,
    [ValidateSet(0, 90, 180, 270)][int]$Rotation = 0,
    [ValidateSet(512, 640)][int]$ImageSize = 640,
    [Parameter(ParameterSetName = 'Run')][string]$MissionDb,
    [Parameter(ParameterSetName = 'Run')][switch]$ResumeMission,
    [string]$Model,
    [switch]$BaselineDetector
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project .venv missing. Complete Phase 0 setup first.' }
if ($ResumeMission -and -not $MissionDb) { throw '-ResumeMission requires -MissionDb.' }
if (($MissionDb -or $ResumeMission) -and $Mode -ne 'Full') { throw 'Mission options require -Mode Full.' }
$resolvedModel = $null
if ($Model) {
    $resolvedModel = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Model)
    if (-not (Test-Path -LiteralPath $resolvedModel -PathType Leaf)) { throw "Model file does not exist: $resolvedModel" }
}
Push-Location $PSScriptRoot
try {
    if ($ListCameras) {
        Write-Host 'Searching for Windows cameras. A camera light may turn on briefly during validation.'
        & $pythonPath -m nidar_survivor_demo.usb_camera --list --max-index $MaxIndex `
            --width $Width --height $Height --fps $Fps --backend $Backend
        $listExit = $LASTEXITCODE
        exit $listExit
    }

    Write-Host "Validating camera index $CameraIndex using $Backend..."
    & $pythonPath -m nidar_survivor_demo.usb_camera --index $CameraIndex `
        --width $Width --height $Height --fps $Fps --backend $Backend
    if ($LASTEXITCODE -ne 0) {
        throw "Camera index $CameraIndex did not return a valid frame. Run .\Start-USBCameraDemo.ps1 -ListCameras or try -Backend msmf."
    }

    $pipelineArgs = @()
    switch ($Mode) {
        'Camera' { }
        'Detect' { $pipelineArgs += @('--detect', '--imgsz', "$ImageSize") }
        'Track'  { $pipelineArgs += @('--track', '--imgsz', "$ImageSize") }
        'Full' {
            $pipelineArgs += @('--manage', '--imgsz', "$ImageSize")
            if (-not $MissionDb) {
                $runId = [guid]::NewGuid().ToString('N')
                $MissionDb = Join-Path $PSScriptRoot "runs\usb-missions\$runId\mission.sqlite3"
            }
            $pipelineArgs += @('--mission-db', $MissionDb)
            if ($ResumeMission) { $pipelineArgs += '--resume-mission' }
            Write-Host "Fresh/local mission database: $MissionDb"
        }
    }
    if ($resolvedModel) { $pipelineArgs += @('--model', $resolvedModel) }
    if ($Mode -ne 'Camera' -and -not $BaselineDetector) { $pipelineArgs += '--hybrid-detector' }
    Write-Host "Starting $Mode pipeline from camera index $CameraIndex. Q/Esc closes the demo."
    & $pythonPath -m nidar_survivor_demo.main --source $CameraIndex --backend $Backend `
        --width $Width --height $Height --fps $Fps --rotation $Rotation `
        --display-fps 60 --seconds $Seconds --device $Device @pipelineArgs
    $demoExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $demoExit
