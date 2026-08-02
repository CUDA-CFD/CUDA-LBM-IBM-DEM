# Architecture

- `LBMSolver`: D2Q9 BGK fluid solver with Guo forcing.
- `IBMParticle`: rigid circular particle state and Lagrangian markers.
- `IBMCoupling`: interpolation, direct forcing, spreading, force and torque accumulation.
- `Simulation`: sedimentation timestep orchestration.
- `VTKWriter`: ParaView output.
