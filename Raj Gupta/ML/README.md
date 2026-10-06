# NIDAR AirMouse — ML workspace

This directory is the complete runnable laptop perception package. For the visual
project overview, architecture, results and full usage guide, open
[`../README.md`](../README.md).

## Fastest verified run

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\Setup-NIDAR.ps1
.\Start-OfflineDemo.ps1
```

## Live camera

```powershell
# Android IP Webcam on the same 5 GHz network
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Manage

# Native USB / Windows camera
.\Start-USBCameraDemo.ps1 -ListCameras
.\Start-USBCameraDemo.ps1 -CameraIndex 0 -Mode Full
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\Start-OfflineDemo.ps1 -Headless -Cycles 1
```

Expected offline counts: `1, 2, 4, 6, 4, 6` with stable motion variants.

The system detects visible person/body candidates. It does not determine death,
injury, consciousness or medical status. Persistent S IDs are mission-scoped,
appearance-based estimates; final-drone duplicate suppression additionally needs
depth and SLAM/map position.
