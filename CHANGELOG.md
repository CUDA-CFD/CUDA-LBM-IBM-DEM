## Unreleased

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
