# NIDAR AirMouse — Planning BOM Under INR 2,00,000

## 1. Status and scope

This is a **team planning budget**, not an official NIDAR requirement and not a purchase quotation. Prices, taxes, shipping, availability, and import costs change. Obtain at least two dated vendor quotes before procurement.

This budget excludes already-owned development equipment:

- Windows laptop with RTX 3050 and 24 GB RAM;
- Android phone;
- ordinary hand tools and general lab equipment.

If the competition requires all equipment to fit inside a formal cost cap, reclassify these items only after checking the official rulebook.

## 2. Recommended budget envelope

| Category | What it covers | Planning cap (INR) |
|---|---|---:|
| Airframe and propulsion | frame, 4 motors, 4 ESCs, propellers, power distribution, landing structure/guards | 32,000 |
| Flight controller | Pixhawk-class controller, power module, safety switch/buzzer, suitable positioning accessories | 30,000 |
| RC and telemetry | transmitter/receiver or safety link, telemetry/local communication hardware | 15,000 |
| Companion computer | Raspberry Pi 5-class board, active cooling, storage, enclosure | 18,000 |
| RGB-depth perception | OAK-D Lite-class camera and protected mount | 35,000 |
| Batteries | at least two suitable flight packs | 12,000 |
| Safe charger | balance charger, leads, LiPo safety bag/basic battery safety accessories | 5,000 |
| Auxiliary sensing | short-range ToF/range/optical-flow allowance if testing requires it | 6,000 |
| Integration | regulators/BECs, wiring, connectors, fasteners, vibration isolation, mounts | 8,000 |
| Critical spares | propellers, one ESC/motor allowance, cables/connectors | 8,000 |
| Contingency | tax/shipping/price movement/replacement allowance | 15,000 |
| **Planned total** |  | **1,84,000** |
| **Remaining headroom** | against INR 2,00,000 target | **16,000** |

The “planning cap” is the maximum allocation for selection, not a claim that a named component currently sells at that price.

## 3. Procurement strategy

### Preferred path

Buy a compatible airframe/propulsion package only after total payload mass is known. Then procure a current supported Pixhawk-class controller, Raspberry Pi 5-class companion computer, and OAK-D Lite-class camera separately so each subsystem can be replaced or upgraded.

### Cost-risk alternative

A bundled development kit can reduce integration mistakes, but a premium kit may consume too much of the INR 2 lakh target. A current Indian reference article listed a Holybro X500 V2/Pixhawk 6C/M10/telemetry development package around INR 90,000 before adding the companion computer, depth camera, batteries, charger, and project-specific payload. Treat this only as a market sanity check, not a selected vendor or final quote.

Reference checked 2026-09-26: https://shop.vebixautomation.com/guides/drone-final-year-project-parts-list-calculations

### Camera price risk

The official Luxonis page listed OAK-D Lite at USD 269 at the time of checking; Indian landed cost may be higher because of currency, tax, shipping, and availability. The INR 35,000 camera cap must be quote-checked before purchase.

Reference checked 2026-09-26: https://checkout.luxonis.com/products/oak-d-lite-1

## 4. Do not buy yet

Do not place final orders until these are known:

1. verified official AirMouse size, weight, safety, and component restrictions;
2. actual payload mass including Pi, OAK-D, mounts, regulators, and guards;
3. required thrust-to-weight margin and desired flight time;
4. corridor/door clearance and propeller-guard geometry;
5. autopilot/ROS compatibility;
6. radio-band legality and venue rules;
7. whether existing team components may be reused and how cost is scored.

## 5. Cost-saving order of operations

- Finish the phone/laptop ML demo before buying flight hardware.
- Prototype ROS, MAVLink, and GCS with simulation or bench hardware.
- Prefer reusable modular parts over a closed ready-to-fly aircraft.
- Spend contingency only after weight, power, and compatibility checks.
- Do not save money by removing battery safety, regulated power, cooling, propeller protection, or critical spares.

## 6. Quote sheet to complete before purchase

For each line item record:

```text
part name / exact SKU
vendor and product URL
unit price, GST and shipping
stock date and lead time
mass and dimensions
electrical specifications
software/firmware compatibility
warranty/return terms
selected / rejected / backup
reason
```

