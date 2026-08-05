# CUDA 06 — Persistent IBM particle pipeline

Commit 021 introduces `cuda_backend::IBMParticleCUDA`, the persistent GPU-side
owner for one rigid immersed particle.

## Device-resident data

The class keeps the following arrays and scalars on the GPU between operations
and timesteps:

- rigid-particle state (`x`, `y`, velocity, acceleration, angle, angular
  velocity, radius, mass, and inertia),
- marker coordinates,
- interpolated marker fluid velocities,
- marker direct-forcing values,
- accumulated particle force and torque.

No marker or particle-state transfer is required between the verified IBM
operations:

```text
Eulerian velocity
      ↓
marker interpolation
      ↓
direct forcing
      ↓
force spreading to the Eulerian lattice
      ↓
hydrodynamic force/torque accumulation
      ↓
particle integration and marker regeneration
```

## Device-state kernel interfaces

The direct-forcing and hydrodynamic-load kernels now read the particle centre
and velocity directly from `IBMParticleStateCUDA` in device memory. This avoids
copying scalar particle state back to the host after every integration step.

The original independent kernel-level verification modes remain available and
exercise the updated interfaces.

## Coupled verification

The new production-path check runs a moving and rotating particle coupled to
the full 200 x 600 LBM domain:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe `
    --verify-ibm-pipeline --steps 100 --interval 10
```

At each comparison point it checks:

- all nine LBM populations,
- density and both velocity components,
- both Eulerian force arrays,
- all particle state and load values,
- marker positions,
- interpolated marker velocities,
- marker force components.

This commit intentionally excludes gravity, buoyancy, and wall contact. Those
loads will be connected before the CUDA sedimentation benchmark is enabled.
