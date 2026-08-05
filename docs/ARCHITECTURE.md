# Architecture

## CPU production path

- `LBMSolver`: D2Q9 BGK fluid solver with Guo forcing.
- `IBMParticle`: rigid circular particle state and Lagrangian markers.
- `IBMCoupling`: interpolation, direct forcing, spreading, force and torque
  accumulation.
- `Simulation`: sedimentation timestep orchestration.
- `WallContact`: four-wall linear spring-dashpot contact.
- `VTKWriter`: ParaView output.

## CUDA production path under construction

- `LBMSolverCUDA`: persistent device-resident fluid state.
- `IBMParticleCUDA`: persistent device-resident particle and marker state.
- Independent CUDA launchers retain one-to-one mappings to the CPU LBM and IBM
  operations.
- `--verify-ibm-pipeline` compares the connected hydrodynamic GPU path with the
  CPU implementation over repeated timesteps.

Gravity, buoyancy, wall contact, benchmark output, and the final CUDA
sedimentation runner are intentionally deferred to subsequent commits.
