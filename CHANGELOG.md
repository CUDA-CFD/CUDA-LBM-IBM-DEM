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
