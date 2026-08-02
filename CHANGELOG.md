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
