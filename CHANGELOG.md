## Unreleased
- Added the first CUDA IBM interpolation kernel using one thread per Lagrangian marker and the shared four-point delta kernel.

- Split `LBMSolver.cpp` into initialization, macroscopic, collision, and streaming translation units to prepare a one-to-one CPU/CUDA implementation mapping.
- No numerical algorithm or execution order was changed.

## [0.8.0] - 2026-08-02

- Freeze the published 2D single-particle sedimentation benchmark.
- Add physical-to-lattice unit conversion in `SedimentationBenchmark.hpp`.
- Use a 200 x 600 lattice with a 25-node particle diameter.
- Add periodic vertical fluid boundaries and no-slip side walls.
- Add benchmark reference trajectory data and documentation.

# Changelog

## v0.1.0

- Modular CPU D2Q9 LBM solver.
- Direct-forcing immersed boundary coupling.
- One translating and rotating circular particle.
- Gravity, buoyancy, wall contact, VTK output, and CSV diagnostics.
## [0.6.0] - 2026-08-02

### Added
- Separate particle weight and fluid buoyancy calculations.
- CSV columns for gravity, buoyancy, external force, and total force.
- Particle area accessor used consistently for mass and displaced-fluid calculations.

### Changed
- Deferred particle-wall contact to the next commit so this revision isolates gravity and buoyancy.


## [0.7.0] - 2026-08-02

### Added
- Translational and angular acceleration stored by `IBMParticle`.
- Translational, rotational, and total kinetic-energy diagnostics.
- Particle Reynolds number and vertical force-residual columns in `sedimentation.csv`.
- Runtime reporting of acceleration, net vertical force, and kinetic energy.

### Changed
- Expanded sedimentation CSV output to support hydrodynamic force-balance verification.

## v0.9.0 - Commit 009

- Added linear spring-dashpot contact with all four container walls.
- Added wall-force, overlap, and contact-state diagnostics to the CSV output.
- Prevented the sedimenting particle from passing through the bottom boundary.

## Commit 010 — Automated sedimentation validation

- Added a dependency-free Python validator for the N=100 reference trajectory.
- Added RMSE, relative RMSE, mean absolute error, and maximum absolute error.
- Added automatic comparison CSV, Markdown report, and SVG plot generation.
- Added the optional CMake target `validate_sedimentation`.

## Commit 012 - CUDA macroscopic kernel

- Added optional CUDA 12.5 build support through `ENABLE_CUDA`.
- Added CUDA device reporting and runtime error checking.
- Ported the LBM macroscopic density and velocity calculation to CUDA.
- Added an automatic CPU-GPU comparison before the CPU sedimentation run.
- Kept the complete sedimentation time loop on the CPU for this commit.


## Commit 013 - CUDA collision kernel

- Added a CUDA implementation of the BGK collision and Guo forcing step.
- Added automatic CPU-GPU comparison for all nine post-collision populations.
- Moved the reusable `DeviceArray<T>` RAII wrapper into `CUDAUtils.cuh`.
- Kept the complete sedimentation time loop on the CPU.

## Commit 014

- Added a CUDA pull-streaming kernel with one thread per destination node.
- Added CPU–GPU verification for all nine post-stream populations.
- Added documentation of the pull data-access model and boundary handling.

## Commit 014.5 - Full-domain CUDA verification framework

- Moved CUDA verification out of the normal sedimentation startup path.
- Added `--verify`, `--steps`, and `--interval` command-line options.
- Added repeated CPU-GPU comparison on the full 200 x 600 benchmark lattice.
- Added maximum, L2, and relative-L2 error metrics for density, velocity, and populations.
- Added failure diagnostics identifying the node, population direction, and values.
- Added CSV output and a dependency-free SVG plotting utility.

## Commit 015 - Persistent CUDA LBM solver

- Added `LBMSolverCUDA` with persistent device-resident fluid arrays.
- Connected macroscopic reconstruction, collision, and pull streaming into one
  GPU timestep.
- Preserved independent kernel-level verification functions.
- Updated full-domain repeated verification to exercise the production CUDA
  solver class.
- Added `--gpu-lbm` for timing the device-resident LBM pipeline and reporting
  MLUPS.

### Commit 016B.4 - CUDA IBM interpolation verification

- Added a full-domain CPU-GPU verification mode for IBM marker interpolation.
- Added `--verify-ibm-interpolation` command-line option.
- Compared both interpolated marker-velocity components using maximum and relative L2 errors.

### Commit 016B.5 - Multi-case CUDA IBM interpolation verification

- Expanded IBM interpolation verification from one kernel launch to seven deterministic cases.
- Added centred and fractionally shifted circular-particle configurations.
- Added explicit marker sets near all four domain boundaries, including partially outside interpolation stencils.
- Added uniform-field, spatially varying-field, interior-grid, and 1000-marker pseudo-random tests.
- Reported per-case errors, kernel-launch count, total marker evaluations, and an overall pass/fail result.
### Commit 017 - CUDA IBM direct forcing

- Added a CUDA direct-forcing kernel with one thread per Lagrangian marker.
- Preserved the CPU rigid-boundary velocity and force formulas exactly.
- Added `--verify-ibm-forcing` with six deterministic CPU-GPU test cases.
- Covered stationary, translating, rotating, combined-motion, varying-fluid, and 1000-marker pseudo-random configurations.
- Reported maximum absolute and relative L2 errors for both marker-force components.

### Commit 018.1 - Fix CUDA architecture propagation

- Applied `CMAKE_CUDA_ARCHITECTURES` directly to the executable target.
- Ensured NVCC generates compute capability 8.9 code for the RTX 4090.
- Fixed the CUDA IBM spreading build, whose double-precision `atomicAdd` requires compute capability 6.0 or newer.

### Commit 018 - CUDA IBM force spreading

- Added a CUDA force-spreading kernel with one thread per Lagrangian marker.
- Used double-precision `atomicAdd` because multiple markers can update the same Eulerian fluid node.
- Preserved the CPU four-point delta stencil, boundary clipping, marker spacing, and additive force semantics.
- Added `--verify-ibm-spreading` with seven deterministic CPU-GPU test cases.
- Covered single-marker, circular-particle, shifted, boundary, high-contention, pseudo-random, and preloaded-force configurations.
- Reported maximum absolute and relative L2 errors for both Eulerian force components.

### Commit 019 - CUDA IBM hydrodynamic-load accumulation

- Added a CUDA kernel that accumulates marker forces into total particle hydrodynamic force and torque.
- Used double-precision atomic additions because every marker contributes to the same three particle-load scalars.
- Preserved the CPU sign convention, marker-spacing factor, moment arm, and clear-before-accumulate semantics.
- Added `--verify-ibm-hydrodynamic-loads` with six deterministic CPU-GPU test cases.
- Covered single-marker, uniform-force, torque-dominated, varying-force, high-contention, and 4096-marker pseudo-random configurations.
- Reported absolute and relative errors for particle force X, particle force Y, and torque.

### Commit 020 - CUDA IBM particle integration

- Added a CUDA rigid-particle integrator that preserves the CPU semi-implicit Euler update order.
- Added a second marker-parallel kernel that regenerates all Lagrangian marker coordinates from the updated centre and orientation.
- Kept total force and torque as separate device scalars so the verified hydrodynamic-load kernel can feed the integrator directly.
- Added `--verify-ibm-particle-integration` with six deterministic CPU-GPU cases.
- Included a 10000-step repeated integration test and a particle with more than one CUDA block of markers.
- Reported errors for all particle-state variables and both marker-coordinate arrays.

### Commit 021 - Persistent CUDA IBM particle pipeline

- Added `IBMParticleCUDA` with persistent device-resident particle state,
  marker coordinates, interpolated velocities, marker forces, and accumulated
  force/torque scalars.
- Connected interpolation, direct forcing, force spreading, hydrodynamic-load
  accumulation, particle integration, and marker regeneration without
  host-device transfers between operations or timesteps.
- Updated direct-forcing and hydrodynamic-load kernels to read the rigid
  particle state directly from device memory.
- Added `--verify-ibm-pipeline` for repeated production-path CPU-GPU comparison
  on the full 200 x 600 lattice.
- Added comparisons for LBM fields, Eulerian forces, particle state, marker
  coordinates, interpolated marker velocities, and marker forces.
- Kept gravity, buoyancy, and wall contact outside the CUDA pipeline for this
  commit so the hydrodynamic path can be verified independently.

### Commit 022 - Complete CUDA sedimentation benchmark

- Added a device kernel for gravity, buoyancy, and normal spring-dashpot wall contact.
- Preserved the CPU external-load addition order and compressive-only contact clamp.
- Added persistent device diagnostics for hydrodynamic, external, wall, and total particle loads.
- Added `--verify-cuda-sedimentation` with seven dedicated external-load/contact cases and repeated full CPU-GPU sedimentation comparison.
- Added `--gpu-sedimentation` for the complete device-resident LBM-IBM benchmark.
- Added `sedimentation_gpu.csv` plus interval-based `fluid_gpu_*.vtk` and `particle_gpu_*.vtk` output.
- Limited full fluid and marker device-to-host transfers to VTK output steps; only compact particle/load data are copied for each CSV row.
