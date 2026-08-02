# Single-particle sedimentation benchmark

## Reference case

The v1.0 validation target is the two-dimensional single circular-particle
sedimentation case reported by Pu et al., *Physics of Fluids* 35, 053307
(2023). Their validation compares the particle trajectory and settling velocity
with earlier finite-element and immersed-boundary calculations.

## Physical configuration

| Parameter | Value |
|---|---:|
| Channel width | 2 cm |
| Channel height | 6 cm |
| Particle diameter | 0.25 cm |
| Initial particle centre | channel centreline, y = 4 cm |
| Fluid density | 1.00 g/cm^3 |
| Particle density | 1.25 g/cm^3 |
| Fluid kinematic viscosity | 1e-5 m^2/s |
| Gravitational acceleration | 9.81 m/s^2 |
| Side boundaries | no-slip bounce-back |
| Vertical boundaries | periodic fluid boundaries |

The reference uses 100 lattice cells per centimetre and an LBM relaxation time
of tau = 0.8. This gives a 200 x 600 lattice and a particle diameter of 25
lattice units.

## Unit conversion

With dx = 1e-4 m and lattice viscosity nu_lu = (tau - 0.5)/3 = 0.1,

- physical time step: dt = nu_lu dx^2 / nu_phys = 1e-4 s;
- lattice gravity: g_lu = g dt^2 / dx = 9.81e-4;
- one physical second corresponds to 10,000 lattice steps.

## Reference trajectory data

The table below is the N = 100 particle-centre vertical coordinate reported by
Pu et al. Values are in centimetres.

| Time (s) | y (cm) |
|---:|---:|
| 0.0 | 4.00 |
| 0.1 | 3.75 |
| 0.2 | 3.28 |
| 0.3 | 2.75 |
| 0.4 | 2.20 |
| 0.5 | 1.65 |
| 0.6 | 1.10 |
| 0.7 | 0.55 |
| 0.8 | 0.14 |
| 0.9 | 0.12 |
| 1.0 | 0.12 |

## Implementation note

The reported trajectory reaches approximately one particle radius above the
bottom boundary. The particle-wall contact model will therefore be added in the
next commit. Commit 008 only freezes the geometry, units, lattice resolution,
and fluid boundary configuration.
