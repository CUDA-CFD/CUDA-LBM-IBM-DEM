# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-08-05

### Added

- Complete serial CPU and persistent CUDA implementations of the 2D LBM–IBM particle-sedimentation pipeline.
- CUDA kernels for macroscopic reconstruction, BGK collision with Guo forcing, and pull streaming.
- CUDA IBM interpolation, direct forcing, force spreading, hydrodynamic load accumulation, rigid-particle integration, and marker regeneration.
- Device-side gravity, buoyancy, and compressive-only spring-dashpot wall contact.
- Independent multi-case CPU–GPU verification for every major CUDA operation.
- Repeated full-domain LBM, persistent IBM-pipeline, and complete sedimentation verification modes.
- CPU and CUDA sedimentation trajectory output with CSV diagnostics and interval-based VTK files.
- Validation against the published `N=100` reference trajectory with RMSE, relative RMSE, MAE, maximum error, reports, and SVG plots.
- Reproducible compute-only CPU–CUDA performance benchmark with warm-up, repeated measurements, synchronized timing, CSV/Markdown reports, and SVG plotting.
- `--version` command and release metadata.

### Verified result

- Full benchmark domain: `200 × 600` lattice nodes.
- Circular particle: 25 lattice-node diameter and 79 Lagrangian markers.
- CPU and CUDA pipelines agree within configured numerical tolerances.
- Median measured runtime: 78.915 s on the serial CPU path and 1.837 s on an NVIDIA RTX 4090 for 10,000 measured steps.
- Median measured speedup: **42.97×**.

## [0.9.0] - 2026-08-02

- Added normal spring-dashpot contact with all four walls.
- Added wall-force, overlap, and contact-state diagnostics.
- Prevented the sedimenting particle from crossing the bottom wall.

## [0.8.0] - 2026-08-02

- Froze the published 2D single-particle sedimentation benchmark.
- Added physical-to-lattice unit conversion and the `200 × 600` benchmark lattice.
- Added periodic vertical fluid boundaries, no-slip side walls, reference trajectory data, and benchmark documentation.

## [0.7.0] - 2026-08-02

- Added translational/angular acceleration, Reynolds number, force residual, and kinetic-energy diagnostics.

## [0.6.0] - 2026-08-02

- Added separate particle weight and fluid buoyancy calculations and corresponding CSV diagnostics.

## [0.1.0]

- Initial modular CPU D2Q9 LBM solver with direct-forcing IBM, one translating/rotating circular particle, VTK output, and CSV diagnostics.
