<#
Run from PowerShell: .\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080
Applies the selected IP Webcam profile explicitly, then opens the preview.
No passwords, images or videos are saved. Phone must already be serving video.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PhoneUrl,
    [ValidateSet('720p', '540p')][string]$Profile = '720p',
    [ValidateRange(0, 86400)][int]$Seconds = 0,
    [switch]$Detect,
    [switch]$Track,
    [switch]$Verify,
    [switch]$ReID,
    [switch]$Manage,
    [switch]$Dashboard,
    [ValidateRange(1024, 65535)][int]$DashboardPort = 8765,
    [string]$MissionDb,
    [switch]$ResumeMission,
    [string]$Model,
    [switch]$BaselineDetector,
    [ValidateSet(512, 640)][int]$ImageSize = 640,
    [ValidateSet(0, 90, 180, 270)][int]$Rotation = 0
)
$ErrorActionPreference = 'Stop'
if ($Dashboard) { $Manage = $true }
if (($MissionDb -or $ResumeMission) -and -not $Manage) { throw 'Mission options require -Manage.' }
if ($ResumeMission -and -not $MissionDb) { throw '-ResumeMission requires -MissionDb.' }
$cameraUri = $null
if (-not [Uri]::TryCreate($PhoneUrl, [UriKind]::Absolute, [ref]$cameraUri) -or
    $cameraUri.Scheme -notin @('http', 'https') -or $cameraUri.UserInfo -or
    $cameraUri.Query -or $cameraUri.Fragment -or $cameraUri.AbsolutePath -ne '/') {
    throw 'Use the base HTTP(S) camera address, without credentials, query or /video path.'
}
$cameraBase = $cameraUri.GetLeftPart([UriPartial]::Authority)
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project .venv missing. Complete Phase 0 setup first.' }
$resolvedModel = $null
if ($Model) {
    $resolvedModel = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Model)
    if (-not (Test-Path -LiteralPath $resolvedModel -PathType Leaf)) { throw "Model file does not exist: $resolvedModel" }
}
$resolution = if ($Profile -eq '720p') { '1280x720' } else { '960x540' }
$desired = [ordered]@{ video_size = $resolution; quality = '35'; motion_detect = 'off' }
$status = Invoke-RestMethod "$cameraBase/status.json?show_avail=1" -Method Post -TimeoutSec 5
if (-not $status.curvals -or $status.avail.video_size -notcontains $resolution) {
    throw 'Camera status is unsupported or requested resolution is unavailable.'
}
$original = @{}
$changed = [System.Collections.Generic.List[string]]::new()
try {
    foreach ($name in $desired.Keys) {
        $original[$name] = [string]$status.curvals.$name
        if (-not $original[$name]) { throw "Camera setting unavailable: $name" }
        $changed.Add($name)
        $null = Invoke-RestMethod "$cameraBase/settings/${name}?set=$($desired[$name])" -Method Post -TimeoutSec 8
    }
    $verified = $false
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        $status = Invoke-RestMethod "$cameraBase/status.json?show_avail=1" -Method Post -TimeoutSec 5
        $verified = $true
        foreach ($name in $desired.Keys) {
            if ([string]$status.curvals.$name -ne $desired[$name]) { $verified = $false }
        }
        if ($verified) { break }
        Start-Sleep -Milliseconds 200
    }
    if (-not $verified) { throw 'Phone did not confirm the requested profile.' }
} catch {
    foreach ($name in $changed) {
        try {
            $value = [Uri]::EscapeDataString($original[$name])
            $null = Invoke-RestMethod "$cameraBase/settings/${name}?set=$value" -Method Post -TimeoutSec 5
        } catch { Write-Warning "Could not restore $name; check phone settings manually." }
    }
    throw 'Camera setup failed; previous settings restoration attempted. Check server/address and retry.'
}
Write-Host "Verified profile: $resolution, JPEG quality 35, phone motion detection OFF. Use 5 GHz Wi-Fi."
Write-Host 'Q / Esc quits. -Detect: boxes. -Track: temporary IDs. -Verify: temporal evidence. -ReID: appearance references. -Manage: persistent estimated identities. Profile remains until phone app reset.'
Push-Location $PSScriptRoot
try {
    $detectArgs = @()
    if ($Detect) { $detectArgs = @('--detect', '--imgsz', "$ImageSize") }
    if ($Track) { $detectArgs = @('--track', '--imgsz', "$ImageSize") }
    if ($Verify) { $detectArgs = @('--verify', '--imgsz', "$ImageSize") }
    if ($ReID) { $detectArgs = @('--reid', '--imgsz', "$ImageSize") }
    if ($Manage) {
        $detectArgs = @('--manage', '--imgsz', "$ImageSize")
        if ($MissionDb) { $detectArgs += @('--mission-db', $MissionDb) }
        if ($ResumeMission) { $detectArgs += '--resume-mission' }
    }
    if ($Dashboard) { $detectArgs += @('--dashboard', '--dashboard-port', "$DashboardPort", '--headless') }
    if ($detectArgs.Count -gt 0 -and -not $BaselineDetector) { $detectArgs += '--hybrid-detector' }
    if ($resolvedModel) { $detectArgs += @('--model', $resolvedModel) }
    & $pythonPath -m nidar_survivor_demo.main --source "$cameraBase/video" --transport mjpeg --rotation $Rotation --display-fps 60 --seconds $Seconds @detectArgs
    $demoExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $demoExitCode
