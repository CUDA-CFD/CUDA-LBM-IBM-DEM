# CUDA backend

The CUDA backend is ported and verified one production operation at a time.

## Persistent solvers

- `LBMSolverCUDA` owns the device-resident D2Q9 populations, macroscopic
  fields, and Eulerian force arrays.
- `IBMParticleCUDA` owns the device-resident rigid-particle state, Lagrangian
  marker arrays, and accumulated particle loads.

## Verified LBM kernels

- macroscopic density and velocity reconstruction,
- BGK collision with Guo forcing,
- pull streaming with benchmark boundary handling.

## Verified IBM kernels

- Eulerian-to-Lagrangian velocity interpolation,
- direct marker forcing,
- marker-to-Eulerian force spreading,
- hydrodynamic force and torque accumulation,
- rigid-particle integration and marker regeneration.

The complete gravity-driven sedimentation benchmark is available through
`--gpu-sedimentation`. Gravity, buoyancy, and the normal spring-dashpot wall
contact model are evaluated on the device. Use `--verify-cuda-sedimentation`
to compare the complete CPU and CUDA paths in lockstep.
