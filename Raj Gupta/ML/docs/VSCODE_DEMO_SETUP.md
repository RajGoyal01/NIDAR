# Run the complete NIDAR demo from VS Code

## What this setup provides

VS Code is only the editor and launcher. The project, Python environment, model
weights, datasets and mission databases remain in this same NIDAR folder. Codex
and VS Code therefore use the same files; nothing needs to be copied.

The supplied `.vscode` configuration provides:

- the correct project Python interpreter;
- one-click GPU/environment validation;
- one-click automated tests;
- a full phone-camera dashboard task;
- USB phone-camera discovery and full-pipeline tasks;
- smaller camera-only and detection-only diagnostic tasks; and
- debugger profiles for phone streams and local video files.

The full runtime flow is:

```text
Android IP Webcam OR Windows USB-webcam device
  -> latest-frame wireless receiver
  -> YOLO visible-person/body detection
  -> BoT-SORT temporary track IDs
  -> temporal evidence verification
  -> appearance Re-ID
  -> persistent survivor/person manager + SQLite mission database
  -> local FastAPI dashboard at 127.0.0.1
```

The dashboard's unique count is an appearance-based estimate. It is not proof of
injury, life status, death or medical condition.

## 1. One-time software setup

1. Install the current VS Code release.
2. Open VS Code and choose **File -> Open Folder**.
3. Select exactly:

   ```text
   C:\path\to\NIDAR\Raj Gupta\ML
   ```

   Do not open only the `nidar_survivor_demo` subfolder. `${workspaceFolder}` in
   the tasks must point to the NIDAR root.
4. When VS Code recommends extensions, install:
   - Python (`ms-python.python`)
   - Python Debugger (`ms-python.debugpy`)
   - Pylance (`ms-python.vscode-pylance`)
5. Press `Ctrl+Shift+P`, run **Python: Select Interpreter**, and select:

   ```text
   C:\path\to\NIDAR\Raj Gupta\ML\.venv\Scripts\python.exe
   ```

The workspace already stores this interpreter choice in `.vscode/settings.json`.
Selecting it manually once is useful if VS Code had another Python cached.

## 2. Verify the installation before using the phone

In VS Code:

1. Press `Ctrl+Shift+P`.
2. Choose **Tasks: Run Task**.
3. Run **NIDAR: Verify environment and GPU**.
4. A small OpenCV test window may appear briefly. The terminal must end with
   successful checks and exit code `0`.
5. Run **NIDAR: Check installed packages**. It should print:

   ```text
   No broken requirements found.
   ```

6. Run **NIDAR: Run all automated tests**. All tests must pass before a demo.

These checks use the local `.venv`; they do not start the camera, download a
model or upload data.

## 3. Prepare the wireless phone camera

1. Connect the laptop to the phone's **5 GHz hotspot**.
2. Open **IP Webcam** on the Android phone.
3. Start its server.
4. Read the base address shown on the phone, for example:

   ```text
   http://PHONE_IP:8080
   ```

5. Test that base address in the laptop browser. If it does not open there, the
   ML program cannot reach it either.

The address can change after reconnecting the hotspot. Always use the address
currently displayed by the phone. Enter only the base URL in the normal task:
do not append `/video`, add a query string, or include a password.

## 4. Start the complete demo (recommended)

Fastest method:

1. Press `Ctrl+Shift+B`.
2. VS Code selects **NIDAR: Start FULL phone demo + dashboard**.
3. Enter the current IP Webcam base URL.
4. Keep `720p` unless the connection is unstable; `540p` is the fallback.
5. Select rotation:
   - `0` when the dashboard image is upright;
   - `90`, `180` or `270` only to correct a rotated phone image.
6. Keep dashboard port `8765`. Use `8766` if another process already uses 8765.
7. Wait for the terminal to print the dashboard URL, then open:

   ```text
   http://127.0.0.1:8765
   ```

The task first asks IP Webcam to use 1280x720, JPEG quality 35 and phone-side
motion detection OFF. It verifies those settings, then starts the entire pipeline.
The browser dashboard is local to this laptop and is not exposed publicly.

To stop the demo, focus its VS Code terminal and press `Ctrl+C`. Do not close
VS Code while the task is running.

## 4A. Use the phone through USB when IP Webcam is unavailable

USB is an **alternative camera input**, not another AI pipeline. Windows first
needs to recognise the phone as a webcam. The project then receives a numbered
camera device such as `0` or `1` and sends its frames through the same YOLO,
tracking, verification, Re-ID and counting stages.

### Phone and Windows preparation

1. Connect the unlocked phone to the laptop with a data-capable USB cable.
2. Open the phone's USB notification/options.
3. Select **Webcam** if that option is available. The exact wording depends on
   the phone and Android version.
4. In Windows, open **Settings -> Privacy & security -> Camera** and allow camera
   access and desktop-app camera access.
5. Close Windows Camera, Teams, Meet and other apps that may already own the
   camera. Only one application should use it during the demo.

A cable by itself does not guarantee camera video. If the phone has no **Webcam**
USB option, Windows needs a trusted virtual-webcam phone/client solution first.
After that software creates a Windows webcam, this project uses it in exactly the
same way. USB tethering is network sharing and is not the same as webcam mode.

### Find the correct camera number

1. In VS Code choose **Terminal -> Run Task**.
2. Run **NIDAR: Find USB and built-in cameras**.
3. Note the values under `available_indices`. A built-in laptop camera is often
   index `0`, so the connected phone may be `1` or another number.
4. If no camera appears, run the task again with `msmf`, reconnect the cable, and
   verify Windows camera permissions.

### Run the complete native-window demo

1. Choose **Terminal -> Run Task**.
2. Run **NIDAR: Start FULL USB camera demo**.
3. Enter the phone camera index discovered above.
4. Keep `dshow` first; choose `msmf` only if DirectShow cannot open the camera.
5. Choose the rotation that makes the person upright.
6. Press `Q` or `Esc` in the preview window to stop cleanly.

This is intentionally the familiar native camera-test layout, with no dashboard.
Each Full launch creates a fresh SQLite mission so old counts do not appear in a
new presentation.

## 5. What happens when the full task runs

`Start-PhoneDemo.ps1` validates the URL before touching the camera. It then calls
the project interpreter with `--dashboard`. That flag intentionally enables all
earlier layers: manager, Re-ID, verification, tracking and detection.

The camera receiver keeps only the newest frame. This prevents an old-frame queue
from growing into seconds of delay. YOLO produces visible-person/body boxes.
BoT-SORT gives short-lived tracking IDs. Several fresh observations are required
before verification. Re-ID compares selected crops on new/reappearing tracks.
The manager stores mission events in a new local SQLite database under `runs/`.
FastAPI serves a read-only operator dashboard bound to localhost.

Each launch without `-MissionDb` creates a new mission, so an old mission's unique
count is not silently reused. The **New Mission / Reset Demo** dashboard button
also rotates to a fresh mission during a running demo.

## 6. Debug the Python code with breakpoints

Use the normal task above for presentations. Use debugging when learning or
investigating code:

1. Open a Python file and click left of a line number to add a red breakpoint.
2. Open **Run and Debug** with `Ctrl+Shift+D`.
3. Select **NIDAR: Debug full phone pipeline**.
4. Press `F5`.
5. This prompt expects the complete stream URL, including `/video`, for example:

   ```text
   http://PHONE_IP:8080/video
   ```

The debugger launches Python directly. Unlike the recommended PowerShell task,
it does not configure the phone first. Run the normal task once or manually keep
IP Webcam at 720p/JPEG35 before debugging.

Other debugger choices:

- **Debug detection-only phone stream** isolates YOLO boxes.
- **Debug full pipeline with local video** works without a phone and asks for an
  absolute video path.

While stopped at a breakpoint, the video receiver may consider frames stale.
That is expected during debugging and is a safety feature, not a camera failure.

## 7. Diagnostic tasks

Use **Terminal -> Run Task** for these smaller tests:

- **Start camera-only latency test**: tests network/video smoothness without ML.
- **Start detection-only phone test**: tests person/body boxes without identity
  tracking or unique counting.
- **Run all automated tests**: validates the software logic without the phone.
- **Find USB and built-in cameras**: reports only camera indices that return a
  real decoded frame, then releases them.
- **Start FULL USB camera demo**: runs the same complete pipeline in the native
  camera-test window, using a fresh mission.

This isolation is useful because it tells you where a problem starts. For
example, if camera-only is already delayed, changing Re-ID cannot fix the network.

## 8. Equivalent terminal commands

The VS Code buttons run ordinary commands. You can run the same full demo from
the integrated PowerShell terminal:

```powershell
.\Start-PhoneDemo.ps1 -PhoneUrl http://PHONE_IP:8080 -Dashboard -DashboardPort 8765 -Rotation 0
```

Environment check:

```powershell
.\.venv\Scripts\python.exe .\nidar_survivor_demo\check_environment.py --gui
```

Automated tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s nidar_survivor_demo\tests -t . -v
```

USB camera discovery and full demo:

```powershell
.\Start-USBCameraDemo.ps1 -ListCameras
.\Start-USBCameraDemo.ps1 -CameraIndex 1 -Mode Full
```

Useful isolated USB modes:

```powershell
.\Start-USBCameraDemo.ps1 -CameraIndex 1 -Mode Camera
.\Start-USBCameraDemo.ps1 -CameraIndex 1 -Mode Detect
.\Start-USBCameraDemo.ps1 -CameraIndex 1 -Mode Track
```

If DirectShow cannot open an otherwise visible camera:

```powershell
.\Start-USBCameraDemo.ps1 -ListCameras -Backend msmf
.\Start-USBCameraDemo.ps1 -CameraIndex 1 -Backend msmf -Mode Full
```

## 9. Recreate `.venv` only if it is missing or broken

Do not reinstall a working environment before a demo. If `.venv` is genuinely
missing, install 64-bit Python 3.11, open PowerShell in the NIDAR root, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r .\nidar_survivor_demo\requirements-lock.txt
.\.venv\Scripts\python.exe .\nidar_survivor_demo\check_environment.py --gui
```

The lock file pins the verified Windows/Python/CUDA package versions. A compatible
NVIDIA driver is still required; the environment checker performs a real CUDA
matrix operation rather than trusting the GPU name alone.

## 10. Common problems and fixes

### Camera setup failed or waiting for fresh video

- Confirm IP Webcam still says **server started**.
- Confirm the laptop is connected to the phone's 5 GHz hotspot.
- Open the base camera URL in the laptop browser.
- Re-enter the phone's current IP; hotspot reconnection can change it.
- Keep the phone app in the foreground and disable battery optimisation for it
  during the demo.

### USB camera list is empty

- Confirm the phone USB menu is set to **Webcam**, not charging or file transfer.
- Use a data-capable cable and try another laptop USB port.
- Allow Windows camera and desktop-app camera permissions.
- Close every other program using a camera.
- Retry with `-Backend msmf`.
- If the phone offers no Webcam mode, install/configure a compatible virtual
  webcam solution; the NIDAR project cannot turn a charging-only USB connection
  into a camera by itself.

### Dashboard page does not open

- Wait until the terminal prints `dashboard_ready`.
- Use `http://127.0.0.1:8765`, not the phone IP.
- If the port is busy, stop the older demo or launch with port `8766`.
- Run only one live pipeline at a time; two processes would compete for GPU and
  camera bandwidth.

### Video is sideways

Stop the task and choose the correct rotation on the next launch. Rotation occurs
before detection, so it corrects both the displayed image and YOLO input.

### Smoothness or delay is poor

- First run the camera-only task.
- Keep 5 GHz and place the phone near the laptop.
- Close other downloads and old demo terminals.
- Use the `540p` task profile only when 720p is unstable.
- Do not start training while presenting; training and live inference share GPU.

### VS Code uses the wrong Python

Check the interpreter shown in the bottom status bar. It must end with
`.venv\Scripts\python.exe`. Then open a new terminal so the change takes effect.

### PowerShell execution policy warning

The supplied VS Code tasks use `-ExecutionPolicy Bypass` for that one child
process only. They do not change the machine-wide PowerShell policy.

## 11. Files and their responsibilities

- `.vscode/settings.json`: interpreter, test discovery and workspace behaviour.
- `.vscode/tasks.json`: repeatable one-click commands and user prompts.
- `.vscode/launch.json`: Python debugger configurations.
- `.vscode/extensions.json`: recommends only the official Python tooling.
- `Start-PhoneDemo.ps1`: validates/configures the phone and launches the demo.
- `Start-USBCameraDemo.ps1`: discovers/validates a Windows webcam and launches
  the unchanged pipeline without a dashboard.
- `nidar_survivor_demo/usb_camera.py`: safely probes camera indices and always
  releases the device.
- `nidar_survivor_demo/main.py`: connects all perception and dashboard stages.
- `nidar_survivor_demo/requirements-lock.txt`: exact verified Python environment.

No camera password or fixed mission identity is stored in the VS Code files.
Generated weights, datasets, logs and SQLite databases remain excluded from Git.

## 12. Complete VisDrone training tasks

The command palette now has two training tasks:

- `NIDAR: Prepare complete VisDrone training data` validates and assembles the full
  research dataset without GPU training;
- `NIDAR: Start complete VisDrone training in background` starts training, held-out
  evaluation and the promotion gate while returning control to VS Code.

Watch `logs\full-visdrone-training.stdout.log` for progress and
`logs\full-visdrone-training.stderr.log` for errors. If a genuine interruption occurs
after `weights\last.pt` exists, resume from PowerShell:

```powershell
.\Start-FullVisDroneTraining.ps1 -Background -Resume
```

Do not train during a live presentation. A completed candidate is not the default
until regression, licence, OAK-D and edge-runtime gates pass.

## Final pre-demo checklist

- [ ] Opened the NIDAR root folder in VS Code.
- [ ] `.venv\Scripts\python.exe` selected.
- [ ] Environment/GPU task passed.
- [ ] Automated tests passed.
- [ ] Phone and laptop are on the same 5 GHz hotspot.
- [ ] IP Webcam server is started and browser-reachable.
- [ ] Old demo/training terminals are stopped.
- [ ] Full dashboard task started with the current phone URL.
- [ ] Dashboard says Camera, YOLO, Tracker, Re-ID and Database are active.
- [ ] Reset/new mission behaviour was tested before presenting.

USB alternative checklist:

- [ ] Phone is visible to Windows as a webcam, not only as a USB storage device.
- [ ] Camera discovery reported the intended index.
- [ ] No other app owns that camera.
- [ ] Full USB task opened the native preview and showed live movement.
- [ ] Person boxes/IDs/counts were tested using a fresh mission.
