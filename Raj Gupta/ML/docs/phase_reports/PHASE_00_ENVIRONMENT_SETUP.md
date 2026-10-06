# Phase 0 — Environment setup and GPU validation

Date: 2026-09-27. Evidence category: PROJECT-DECISION / locally measured results.

## 1. Objective and scope

Prepare the laptop to receive the moving Android phone stream in Phase 1 and run
GPU person detection in Phase 2. The phone simulates the drone camera; the laptop
temporarily hosts perception. Phase 0 validates tools, not detection accuracy or
phone latency. No application stream, tracker, Re-ID, or dashboard is built here.

## 2. Starting condition

The repository contained AGENTS.md and eight project planning documents, with
empty demo folders from an earlier interrupted session. No application existed.
System Python was 3.13.7. Earlier inspection found torch 2.11.0+cpu, CUDA false,
headless OpenCV, and no Ultralytics. This phase leaves that global installation
alone and creates a separate repository-root `.venv`.

The current hardware query reports NVIDIA GeForce RTX 3050 A Laptop GPU,
4094 MiB dedicated memory and driver 610.88. Free disk space before installation
was approximately 186 GB (decimal).

## 3. Concepts in simple English

| Term | Meaning and why we need it |
|---|---|
| Python interpreter | Program that executes our Python files. We use Python 3.11. |
| Virtual environment | Separate package box for this project. Other projects can use different versions. |
| pip / uv | Tools that download and install Python packages. uv also supplies managed Python. |
| Dependency | A library our program uses, such as OpenCV. |
| Transitive dependency | A library needed by one of our libraries. |
| Version pin | An exact version number to reduce unexpected changes between installations. |
| CUDA | NVIDIA's GPU computing platform; PyTorch uses it for tensor operations. |
| Tensor | An array of numbers. Images and neural-network weights are represented as tensors. |
| GPU driver | Windows software that lets applications communicate with the NVIDIA GPU. |
| CUDA runtime | Libraries supplied with the selected PyTorch wheel for GPU execution. |
| VRAM | GPU memory, distinct from the laptop's 24 GB system RAM. |
| Headless OpenCV | OpenCV without desktop display support; unsuitable for our intended imshow window. |
| Smoke test | A small real operation proving a component works before larger features are built. |
| Exit code | 0 means all selected checks passed; 1 means at least one failed. |

The CUDA version displayed by nvidia-smi is not proof that the installed PyTorch
has CUDA support. A CPU-only wheel cannot use the GPU even with a working driver.
For these prebuilt PyTorch wheels we do not need to install a separate full CUDA
compiler toolkit merely to run the smoke test.

## 4. Chosen dependencies

| Package | Pin | Responsibility |
|---|---|---|
| Python | 3.11.15 installed via uv | Compatible project interpreter |
| torch | 2.11.0+cu128 | CUDA tensor computation |
| torchvision | 0.26.0+cu128 | Matching PyTorch vision utilities |
| numpy | 2.2.6 | Image arrays and synthetic test frames |
| opencv-python | 4.13.0.92 | Video encoding/decoding and GUI support |
| ultralytics | 8.4.163 | Future model loading and inference interface |

These are a pinned baseline, not a claim that every package is the newest.
PyTorch's official version table documents the selected torch/torchvision pair
and CUDA 12.8 index. Tracker and Re-ID-specific dependencies are deferred to their
phases. Installing Ultralytics does not mean we have implemented person detection.

## 5. Files and responsibilities

| File | Responsibility |
|---|---|
| `.gitignore` | Excludes environment, secrets, weights, recordings and generated data from future Git tracking |
| `nidar_survivor_demo/requirements.txt` | Direct dependency pins and official CUDA package index |
| `nidar_survivor_demo/requirements-lock.txt` | Full installed dependency versions for reproduction |
| `nidar_survivor_demo/check_environment.py` | Reusable Phase 0 diagnostics |
| `nidar_survivor_demo/README.md` | Commands and phase boundary |
| This report | Study material, evidence, debugging and viva preparation |

Project status and dated decisions are maintained in PROJECT_CONTEXT.md,
ML_DEMO_PLAN.md and DECISIONS_AND_ASSUMPTIONS.md. The repository is not yet a Git
repository; adding .gitignore does not initialise or commit one.

## 6. What happens when the script runs

```text
Project Python starts
  -> verify Python version and virtual environment
  -> import packages and reject conflicting OpenCV distributions
  -> pip checks package requirements
  -> CUDA creates two-dimensional numbers on RTX 3050
  -> GPU multiplies a matrix and result is checked
  -> OpenCV writes then reads ten synthetic video frames
  -> optional GUI window opens and closes
  -> print overall PASS/FAIL and return an exit code
```

This phase has no API, HTTP server, database, authentication or deployment. All
checks execute locally. Installation needs internet; the check itself does not
need a camera, external inference service or training dataset.

## 7. Code walkthrough

`check_python()` compares sys.prefix with sys.base_prefix. Different values mean
Python is running in a virtual environment. It also enforces our Python 3.11
baseline so accidentally invoking global Python is visible.

`check_packages()` imports the actual modules and reads distribution versions.
Package name and import name can differ: opencv-python is imported as cv2.
It rejects additional OpenCV distributions because they share the same cv2 name.

`check_pip()` calls pip through sys.executable, ensuring it checks the same
interpreter. capture_output collects the diagnostic text; timeout prevents an
unbounded subprocess wait.

`check_cuda()` checks availability and GPU identity, then creates a 128 by 128
matrix of ones on cuda:0. Multiplying it by itself must produce 128 in each cell.
torch.cuda.synchronize() waits for GPU work to finish before evaluating success.
torch.inference_mode() avoids tracking gradients because we are not training.
This small operation tests basic CUDA execution, not model speed or FP16 support.

`check_video()` uses a temporary directory, MJPEG AVI, and ten synthetic frames.
It validates frame count, shape, approximate brightness and sequence. finally
blocks release the writer and capture even on failure; the temporary directory
removes the generated file. No footage of people is recorded.

`check_gui()` runs only with --gui, opens a small blank window briefly and closes
it. This tests the GUI API; it does not claim a human has reviewed the rendering.

`main()` runs all checks and prints one JSON record per outcome. Exceptions become
visible FAIL records, and any failure makes the overall result fail. JSON is a
structured text format that humans and future tools can both inspect.

## 8. Commands and tests

Run from the project root in Windows PowerShell:

```powershell
uv venv --python 3.11 --seed .venv
.\.venv\Scripts\python.exe -m pip install -r .\nidar_survivor_demo\requirements.txt
.\.venv\Scripts\python.exe .\nidar_survivor_demo\check_environment.py --gui
$LASTEXITCODE
```

The first two commands are setup commands for a fresh environment. Once installed,
only run the check and inspect its exit code. To use the installed environment
interactively, optionally run `.\.venv\Scripts\Activate.ps1`.

During this implementation uv pip was used for download speed with the same pins:

```powershell
uv pip install --python .venv\Scripts\python.exe -r nidar_survivor_demo\requirements.txt --index-strategy unsafe-best-match
```

The index strategy lets uv resolve PyPI packages alongside the official CUDA
index. Direct package versions are pinned. Reproduction with pip is documented
in README.md, including the full dependency lock.

Manual checklist:

1. Open PowerShell at the project root.
2. Run the check with the explicit .venv interpreter.
3. Confirm CUDA available true and expected RTX 3050 name.
4. Confirm matrix multiplication PASS and ten decoded video frames.
5. Observe the brief blank GUI test window with --gui.
6. Confirm overall PASS and exit code 0.

## 9. Pretrained dataset versus pretrained model

A dataset contains example images and annotations. Training learns numerical
parameters (weights) from those examples. A pretrained model contains the network
and already learned parameters; a .pt file is the checkpoint we later load.

Think of COCO as the study material and the weights as the learned result.
We need the weights for inference, not a full local copy of the study material.

Official sources:

- Model list and loading examples: https://docs.ultralytics.com/tasks/detect/
- Official model assets: https://github.com/ultralytics/assets/releases
- COCO background and dataset access: https://cocodataset.org/
- PyTorch wheel pairs: https://pytorch.org/get-started/previous-versions/

Phase 2 will check the installed package against the official model list, choose
a Nano checkpoint, download it once, store its source/version/checksum, verify the
class-name mapping, then load the local file on CUDA. Current documentation lists
yolo26n.pt as an official COCO-pretrained candidate. It is not locked or downloaded
in Phase 0. A local cached weight allows subsequent inference without internet.

The future data path is phone frame -> model -> person boxes/confidence -> tracker
-> verification -> persistent identity -> display. Restricting output to person
does NOT eliminate most network computation: the pretrained network still runs.
This corrects the earlier overly broad claim that class filtering avoids all
compute for other classes. It mainly restricts returned detections/downstream work.

Confidence is not measured accuracy or proof that a detected human is a survivor.
COCO person performance on competition dummies remains unverified. Fine-tuning is
later, using actual failure cases, consented mission-like data and separate scenes
for train/validation/test. Detector training alone will not solve re-entry identity.

## 10. Errors: cause, diagnosis, fix, recognition

| Symptom | Cause / diagnosis | Fix and future recognition |
|---|---|---|
| nvidia-smi sees GPU but CUDA is false | CPU-only torch wheel or wrong interpreter | Inspect torch.__version__, torch.version.cuda and sys.executable; install pinned CUDA build inside .venv |
| imshow unsupported | Headless OpenCV build | Use only opencv-python in the isolated environment |
| Package imports fail | Wrong Python or incomplete installation | Use explicit .venv path and pip check |
| PowerShell blocks activation | Script execution policy | Use explicit .venv python.exe path; activation is optional |
| DLL/driver failure | CUDA wheel/driver mismatch or missing runtime component | Read the FAIL record; compare driver and official wheel support before changing versions |
| CUDA check passes but phone lags later | GPU computation and capture latency are separate | Measure stream delay in Phase 1; tensor success is not a camera benchmark |

## 11. Viva questions and answers

**Why a virtual environment?** It isolates this project's versions from global
Python and other projects, making debugging and reproduction easier.

**Why Python 3.11?** It follows our documented baseline and provides a deliberate
compatibility target. We do not claim Python 3.13 can never work.

**Why did the GPU fail before?** The existing torch wheel was CPU-only. A working
NVIDIA driver does not add CUDA support to a CPU-only Python package.

**Why multiply a matrix?** Device detection only lists the GPU. A real operation
tests that computation can execute and return the expected result.

**Have we trained a model?** No. Phase 0 prepares the software environment.

**Where does person knowledge come from?** A later official pretrained checkpoint
has learned parameters from COCO training. Our first demo uses inference.

**Will this already recognise returning people?** No. Detector boxes do not encode
persistent identity. That requires tracking and the later survivor/Re-ID layers.

**Can the phone move away from the laptop?** That is the intended Phase 1 local
wireless setup. Network range and latency still require physical testing.

**Does CUDA PASS prove 30 FPS?** No. Actual stream and full pipeline benchmarks
come later. This smoke test is deliberately small.

**Does a local video test validate the phone?** No. It validates OpenCV video I/O.
The phone app, stream URL and reconnect behavior are Phase 1 acceptance tests.

**Is data sent to a server?** The validation performs local operations. Packages
are downloaded during setup; no camera stream or dataset is uploaded.

## 12. Results and next-phase boundary

Actual validation on 2026-09-27: **PASS, zero failed checks, process exit code 0**.

| Check | Observed result |
|---|---|
| Project interpreter | Python 3.11.15, virtual environment true |
| Package imports | All five required imports succeeded at the pinned versions |
| Dependency consistency | pip: No broken requirements found |
| CUDA | Available true, runtime 12.8 |
| GPU | NVIDIA GeForce RTX 3050 A Laptop GPU, 4094 MiB |
| Real computation | 128 x 128 GPU matrix multiplication returned expected values |
| Local video | All 10 synthetic frames decoded with correct shape/content/order |
| GUI | WIN32UI build; window create/destroy API calls succeeded |
| Failure behavior | Running through global Python returned FAIL and exit code 1 |
| Syntax | py_compile succeeded |

No full-pipeline FPS or inference latency was measured because no detector or
phone stream has been implemented. No physical camera was opened. A human visual
review of the brief blank window is still available through the manual command;
the automated result confirms successful GUI calls.

The first Ultralytics import reported updating the user-level settings schema at
the standard Windows Roaming/Ultralytics location, preserving existing values
where possible. This is a package import side effect outside the virtual
environment; package isolation does not imply all libraries isolate user settings.
No system Python packages were replaced. Within .venv, uv replaced the seed
setuptools 84.0.0 with the resolved 81.0.0 version. The full dependency snapshot
records it. CUDA wheel preparation took about 14 minutes on this connection;
download time is not an inference benchmark.

Stop after Phase 0 and obtain the user's phase confirmation before Phase 1.
Later phase reports must include objective, concepts, files, code explanation,
runtime flow, commands, tests, measured evidence, errors, limitations and viva Q&A.
