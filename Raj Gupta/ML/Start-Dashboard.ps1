<# Camera-free dashboard. -Replay runs actual OSNet with oracle tracks; -MissionDb reviews saved evidence. #>
[CmdletBinding(DefaultParameterSetName = 'Replay')]
param(
    [Parameter(ParameterSetName = 'Replay')][switch]$Replay,
    [Parameter(Mandatory = $true, ParameterSetName = 'Archive')][string]$MissionDb,
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [ValidateRange(0, 86400)][int]$HoldSeconds = 3600
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
Write-Host "Dashboard: http://127.0.0.1:$Port (read-only, local laptop only). Ctrl+C stops the server."
Push-Location $PSScriptRoot
try {
    if ($PSCmdlet.ParameterSetName -eq 'Archive') {
        & $pythonPath -m nidar_survivor_demo.dashboard --mission $MissionDb --port $Port
    } else {
        $runId = [guid]::NewGuid().ToString('N')
        & $pythonPath -m nidar_survivor_demo.demo_reid_offline --pairs 1 --dashboard --dashboard-port $Port --hold-dashboard $HoldSeconds --manage-dir "runs/dashboard-$runId" --output "logs/dashboard-$runId.json"
    }
    $dashboardExit = $LASTEXITCODE
} finally { Pop-Location }
exit $dashboardExit
