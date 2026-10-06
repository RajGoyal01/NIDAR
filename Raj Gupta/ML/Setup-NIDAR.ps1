<# One-command Windows setup for the packaged NIDAR laptop demonstration. #>
[CmdletBinding()]
param(
    [switch]$RunTests,
    [switch]$SkipShowcaseAssets
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'

Push-Location $root
try {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw 'Python launcher not found. Install 64-bit Python 3.11, then run this script again.'
        }
        Write-Host 'Creating isolated Python 3.11 environment...'
        & py -3.11 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required but was not found by the Windows py launcher.' }
    }

    Write-Host 'Installing pinned CUDA/ML/backend dependencies...'
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
    & $python -m pip install -r nidar_survivor_demo\requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

    $manifest = Get-Content models\manifest.json -Raw | ConvertFrom-Json
    foreach ($entry in $manifest.models) {
        $path = Join-Path $root $entry.path
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required model is missing: $($entry.path)" }
        $item = Get-Item -LiteralPath $path
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($item.Length -ne $entry.bytes -or $hash -ne $entry.sha256) {
            throw "Model integrity failed: $($entry.path)"
        }
    }
    Write-Host 'All three runtime model files passed size and SHA-256 validation.'

    if (-not $SkipShowcaseAssets) {
        & $python -m nidar_survivor_demo.setup_showcase
        if ($LASTEXITCODE -ne 0) { throw 'Showcase asset preparation failed.' }
    }

    & $python -m nidar_survivor_demo.check_environment
    if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed. Review the JSON checks above.' }

    if ($RunTests) {
        & $python -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw 'Automated tests failed.' }
    }
    Write-Host ''
    Write-Host 'NIDAR setup complete.' -ForegroundColor Green
    Write-Host 'Phone-free demo: .\Start-OfflineDemo.ps1' -ForegroundColor Cyan
    Write-Host 'USB demo:        .\Start-USBCameraDemo.ps1 -ListCameras' -ForegroundColor Cyan
    Write-Host 'Phone demo:      .\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Manage' -ForegroundColor Cyan
} finally {
    Pop-Location
}
