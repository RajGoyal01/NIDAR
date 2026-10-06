# NIDAR AirMouse — Hardware Architecture and Role Split

## 1. Design principle

Use the right computer for the right job. The flight controller is the reflex system, the companion computer is the mission brain, and the depth camera is the perception sensor.

## 2. Pixhawk-class flight controller

### Responsibility

- attitude estimation and stabilisation;
- motor/ESC outputs;
- flight modes and arming logic;
- low-level position/velocity/attitude control;
- battery, link, and flight failsafes supported by the chosen autopilot;
- receiving bounded high-level setpoints from the companion computer;
- emergency landing/abort behaviour as designed and rulebook-permitted.

### Must not be responsible for

- running the full YOLO/Re-ID pipeline;
- storing the survivor identity database;
- rendering the dashboard;
- acting as the primary 2D mapping computer.

Why: flight stabilisation is time-critical. It must not compete with variable-latency AI work.

## 3. Raspberry Pi 5 companion computer

### Responsibility

- ROS 2 node orchestration;
- OAK-D and autopilot communication;
- visual-inertial odometry / SLAM;
- coordinate-frame transforms and timestamp alignment;
- obstacle/map interpretation and high-level planning;
- survivor observation fusion and persistent identity;
- mission state machine;
- logging, health monitoring, and GCS communication.

### Engineering needs

- active cooling;
- reliable storage rather than a fragile unprotected card where possible;
- regulated power with adequate current headroom;
- watchdog/restart strategy;
- measured CPU, memory, temperature, and latency under full mission load.

## 4. OAK-D Lite-class perception camera

### Responsibility

- RGB images for person/survivor perception;
- stereo depth for obstacle distance and survivor position;
- IMU data where supported and suitably synchronised;
- optional RVC2/OpenVINO-compatible edge inference.

The official Luxonis product page describes OAK-D Lite as an RGB/stereo-depth device with an onboard RVC2 accelerator and 6-axis IMU. Exact focus variant, field of view, mounting orientation, and runtime model must be selected after indoor tests.

Product reference checked 2026-09-26: https://checkout.luxonis.com/products/oak-d-lite-1

## 5. Ground Control Station

### Responsibility

- live camera feed;
- continuously updated 2D map;
- survivor/grid markers;
- vehicle health and mission phase;
- module status, faults, and event log;
- permitted launch/abort/recall controls;
- final mission report and logs.

The GCS is the operator's window into autonomy. It must not hide uncertainty: stale video, lost SLAM, unconfirmed survivor identity, or degraded depth should be visible.

## 6. Data and command interfaces

```text
OAK-D --USB--> Raspberry Pi
Raspberry Pi <--MAVLink/serial--> Pixhawk
Raspberry Pi <--local radio/network--> GCS
Pixhawk --> ESCs --> motors
Battery --> power module --> Pixhawk telemetry
Battery --> regulated BEC/DC-DC --> Pi + OAK-D
```

ROS 2 topics/services are a project decision. MAVLink is the likely bridge between companion and autopilot, but exact transport and message set remain to be validated.

## 7. Coordinate flow for survivor localisation

```text
person pixel / bounding box
  + stereo depth
  -> 3D point in camera frame
  + calibrated camera-to-body transform
  -> 3D point in drone/body frame
  + SLAM pose
  -> 3D point in map frame
  -> 2D floor projection / grid cell
  -> survivor marker with uncertainty
```

Calibration is not optional. A visually correct box can still produce a wrong map location if timestamps or transforms are wrong.

## 8. Airframe and safety hardware categories

The exact airframe is open, but the final design must budget for:

- frame and protected landing structure;
- motors, ESCs, propellers, and power distribution;
- propeller guards suitable for indoor testing if required/feasible;
- battery, safe charger, voltage/current monitoring;
- Pixhawk, GPS/compass if used outside the GPS-denied mission, buzzer and safety switch;
- RC/manual safety link where permitted;
- telemetry/local network;
- OAK-D mount with vibration control and unobstructed stereo view;
- Pi mount, cooling, regulated power, and cable strain relief;
- downward/side range or optical-flow sensors only if testing proves they are needed;
- spare propellers and critical connectors.

## 9. Integration order

1. Bench-test each compute unit independently.
2. Validate OAK-D RGB/depth/IMU timestamps.
3. Validate Pi-to-Pixhawk MAVLink on the bench with motors disarmed.
4. Measure full electrical load and regulator temperature.
5. Build and calibrate the airframe.
6. Perform manual stabilisation tests in a safe area.
7. Run recorded perception and SLAM data offline.
8. Run tethered/guarded indoor hover tests.
9. Add mapping and high-level setpoints one bounded capability at a time.
10. Add survivor localisation and mission flow after safe navigation is stable.

## 10. Open selections

- airframe size and propulsion combination;
- Pixhawk exact model and autopilot firmware;
- battery voltage/capacity and target flight time;
- propeller-guard design;
- optical-flow/range sensor need;
- communication link and permitted radio band;
- ROS 2 distribution and SLAM package;
- OAK-D focus variant and mounting angle.

Resolve these using payload weight, thrust margin, corridor width, flight time, rulebook, and measured compute/power loads—not brand preference alone.

