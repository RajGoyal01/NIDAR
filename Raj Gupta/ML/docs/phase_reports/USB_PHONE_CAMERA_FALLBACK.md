# USB phone-camera fallback — learning and handoff report

Date: 2026-10-05  
Decision: D-052  
Scope: additive camera-input option; no perception or identity redesign

## 1. Objective

Add a wired phone-camera option for presentations when IP Webcam or local Wi-Fi
is unavailable, while preserving the existing Wi-Fi and offline demos exactly.

The important design choice is that USB changes only **where frames come from**.
It does not create a second detector, tracker or database path.

## 2. Beginner-friendly concept

Windows gives cameras small numeric addresses called **camera indices**. A laptop
camera may be index `0`; a USB phone webcam may be index `1`. OpenCV opens that
index and returns images one after another.

Think of the camera source as changing the plug at the beginning of a pipe. The
rest of the pipe remains the same:

```text
Phone USB webcam / built-in webcam
  -> Windows camera index
  -> OpenCV capture thread
  -> one latest-frame slot
  -> YOLO person/body boxes
  -> BoT-SORT temporary tracks
  -> temporal verification
  -> selective appearance Re-ID
  -> persistent S identity manager
  -> fresh SQLite mission + native preview
```

The one-frame slot is important. It replaces the old frame whenever a newer one
arrives, rather than building a queue that turns into visible delay.

## 3. What was built

### `nidar_survivor_demo/usb_camera.py`

This small discovery tool opens one or more Windows camera indices, requests the
selected width/FPS, and requires an actual decoded BGR frame before declaring a
camera ready. It always releases the camera in a `finally` block, including after
failure. It prints structured JSON so a human or future launcher can read it.

### `Start-USBCameraDemo.ps1`

This is the beginner-friendly entry point. It:

1. checks the project virtual environment;
2. validates the optional model path and launcher parameters;
3. lists cameras or validates the requested index;
4. starts the unchanged `nidar_survivor_demo.main` pipeline;
5. uses the native camera-test window by default, not the dashboard;
6. creates a fresh SQLite mission for every Full run unless an explicit mission
   is supplied; and
7. returns the underlying program exit code.

Available modes are:

- `Camera`: live video and latency only;
- `Detect`: YOLO boxes only;
- `Track`: YOLO plus temporary BoT-SORT IDs;
- `Full` (default): all perception, identity and persistence stages.

### VS Code tasks and debugger

Two tasks were added: **Find USB and built-in cameras** and **Start FULL USB
camera demo**. A USB full-pipeline debugger profile was also added. Existing
phone tasks remain available.

## 4. Files involved

| File | Responsibility |
|---|---|
| `Start-USBCameraDemo.ps1` | Validate parameters, select camera, create a fresh mission and launch the chosen mode. |
| `nidar_survivor_demo/usb_camera.py` | Discover/validate camera indices and release devices safely. |
| `nidar_survivor_demo/camera/phone_stream.py` | Existing single-owner capture thread and latest-frame slot; reused unchanged. |
| `nidar_survivor_demo/main.py` | Existing complete perception/identity pipeline; reused unchanged. |
| `.vscode/tasks.json` | One-click discovery and Full USB tasks. |
| `.vscode/launch.json` | Breakpoint/debug configuration for a USB camera. |
| `nidar_survivor_demo/tests/test_usb_camera.py` | Camera probe and launcher safety tests. |
| `docs/VSCODE_DEMO_SETUP.md` | Exact phone, Windows, VS Code and PowerShell steps. |

## 5. Libraries and configuration choices

- OpenCV `4.13.0.92` is already pinned and already powers the other camera
  sources. No new package was needed.
- DirectShow (`dshow`) is the Windows default because it works well with ordinary
  webcam devices. Media Foundation (`msmf`) is available as a fallback.
- Requested defaults are 1280x720 at 30 FPS. Camera drivers may negotiate a
  different format; the probe reports the actual decoded frame size.
- Full mode defaults to CUDA device `0`, YOLO input 640 and a new local mission.
- Rotation can be `0`, `90`, `180` or `270` degrees and is applied before both
  inference and display.

No secret, phone address or fixed camera index is stored in the project.

## 6. Exact Windows/PowerShell commands

Open PowerShell in `C:\path\to\NIDAR\Raj Gupta\ML`.

Find camera indices:

```powershell
.\Start-USBCameraDemo.ps1 -ListCameras
```

If DirectShow finds nothing:

```powershell
.\Start-USBCameraDemo.ps1 -ListCameras -Backend msmf
```

Start the complete live native-window demo (replace `1` with the discovered
phone index):

```powershell
.\Start-USBCameraDemo.ps1 -CameraIndex 1 -Mode Full
```

Correct a sideways phone:

```powershell
.\Start-USBCameraDemo.ps1 -CameraIndex 1 -Mode Full -Rotation 90
```

Isolate a problem:

```powershell
.\Start-USBCameraDemo.ps1 -CameraIndex 1 -Mode Camera
.\Start-USBCameraDemo.ps1 -CameraIndex 1 -Mode Detect
.\Start-USBCameraDemo.ps1 -CameraIndex 1 -Mode Track
```

Press `Q` or `Esc` in the preview to release the camera cleanly.

## 7. Phone preparation

1. Use an unlocked phone and a data-capable USB cable.
2. Connect the phone to the laptop.
3. In the phone USB options, select **Webcam** if the device provides it.
4. Allow Windows camera and desktop-app camera permissions.
5. Close Camera, Teams, Meet or any program that may own the device.

A USB cable is transport, not automatically a webcam. If the phone does not
offer Webcam mode, a compatible virtual-webcam phone/client solution must first
create a Windows camera device. USB tethering only creates a network connection;
it does not itself create a webcam.

## 8. Validation performed

### Automated validation

- A successful fake camera must return its actual frame dimensions/FPS.
- A no-frame camera is rejected.
- Camera handles are released after success and failure.
- Invalid negative indices are rejected before camera access.
- A missing custom model is rejected before camera probing.
- Existing source parsing, reconnect, latest-frame, stale-frame and clean-shutdown
  tests continue to pass.

Final closeout results:

- 179 project tests passed in 49.574 seconds;
- Windows discovery found three frame-producing indices (`0`, `1`, `2`) at the
  time of the test;
- camera-only index `0` produced 1280x720 at about 29.6-29.8 FPS, 166 unique
  consumed frames and a clean shutdown;
- Full index `0` started CUDA YOLO, BoT-SORT, verification, Re-ID and a fresh
  SQLite mission, consumed 243 processed frames and shut down cleanly;
- the Full smoke observed no person in that camera view, so it validates pipeline
  wiring and lifecycle, not person accuracy; and
- dependency consistency passed (`pip check`).

### Hardware validation

A Windows camera device was exercised, but it was not ground-truth-verified as the
user's Moto phone. Therefore Moto USB mode, phone-view image quality and physical
phone delay remain **HARDWARE-VERIFY**, not falsely reported as passed.

## 9. Manual acceptance test

1. Run camera discovery and identify the phone index.
2. Run `-Mode Camera`; wave/pan the phone and confirm the view is live and upright.
3. Quit and run `-Mode Detect`; show standing, sitting and lying people and confirm
   boxes appear when the current model can see them.
4. Quit and run `-Mode Track`; confirm temporary IDs usually remain during
   continuous visibility.
5. Quit and run `-Mode Full`; confirm `S1`, `S2`, etc. and the count overlay.
6. Move a person out and back in. Confirm a confident appearance match reuses the
   persistent identity; uncertain evidence may remain pending.
7. Unplug/reconnect once and confirm safe reconnect behaviour.
8. Quit with `Q`/`Esc`, then reopen discovery to prove the camera was released.

Pass requires live frames, clean release, and no stale-frame accumulation. It
does not prove final-drone field accuracy.

## 10. Errors and how to debug them

### No indices are available

Cause: the phone is charging/file-transfer only, Windows permission is denied,
the cable is power-only, or another program owns the camera.

Recognition: discovery returns an empty `available_indices` list.

Fix: enable Webcam mode, change cable/port, allow desktop camera access, close
other camera apps, then retry `dshow` and `msmf`.

### Wrong view opens

Cause: index `0` is usually the built-in camera rather than the phone.

Recognition: the preview angle is from the laptop lid.

Fix: rerun discovery and try the next available index.

### Sideways image

Cause: the phone sensor orientation differs from the presentation orientation.

Fix: relaunch with `-Rotation 90`, `180` or `270`. Rotation affects both the
display and detector, so box coordinates stay aligned.

### Camera works in Windows Camera but not NIDAR

Cause: Windows Camera may still own the device, or the selected backend may not
support that driver.

Fix: close Windows Camera completely and try `-Backend msmf`.

## 11. Limitations

- Native phone USB-webcam availability depends on phone firmware/USB mode.
- The program cannot turn a charging-only cable into a webcam.
- Windows camera indices can change after reconnect/reboot; discover before a
  presentation instead of hard-coding one.
- A USB link can improve transport stability but cannot make detection or Re-ID
  perfect. Blur, occlusion, tiny people and domain shift remain model limits.
- Unique people are appearance-based estimates, not medical status, liveness or
  final-drone map-grounded identities.

## 12. What you learned

- A camera backend is an adapter that converts a device into frames.
- Keeping input adapters separate lets one ML pipeline support Wi-Fi, USB and
  recorded files without duplicating business logic.
- Validation should require a decoded frame, not merely an “opened” flag.
- `finally` cleanup prevents a failed test from leaving a camera locked.
- Fresh mission databases prevent an old presentation count from contaminating a
  new one.

## 13. Professor/viva questions and answers

**Q: Did you create a different model for USB?**  
A: No. USB only changes the capture source. All ML weights and identity logic are
the same, which keeps results comparable.

**Q: Why discover camera indices?**  
A: Windows identifies webcams numerically, and the phone is not guaranteed to be
index zero. Discovery verifies which indices can return real frames.

**Q: Why keep DirectShow and MSMF?**  
A: They are two Windows camera backends. Driver compatibility varies, so one is a
controlled fallback for the other.

**Q: Does USB prove the final drone will work?**  
A: No. It validates a live wired camera input for the laptop proof of concept.
The final drone still needs OAK-D, depth, SLAM, onboard compute and field tests.

**Q: Are `S1` and `S2` medically verified survivors?**  
A: No. They are persistent appearance-based person-candidate estimates. The
system does not infer injury, death or liveness.
