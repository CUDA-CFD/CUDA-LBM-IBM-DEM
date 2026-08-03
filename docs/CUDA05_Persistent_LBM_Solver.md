# Persistent CUDA LBM solver

`cuda_backend::LBMSolverCUDA` owns all fluid arrays in device memory for the
lifetime of the GPU solver. A timestep launches the already verified kernels in
this order:

1. macroscopic reconstruction,
2. BGK collision with Guo forcing,
3. pull streaming,
4. population-buffer swap.

No host-device transfer occurs inside `LBMSolverCUDA::step()`. The initial
state and force arrays are uploaded once, and results are downloaded only when
output or verification is requested.

The individual launch functions remain public and independently testable:

- `launchComputeMacroscopic`,
- `launchCollideAndForce`,
- `launchPullStreaming`.

The repeated full-domain verifier now uses the same persistent solver class,
while kernel-level verification remains available for diagnosing individual
operations.

## Run

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe --gpu-lbm --steps 10000
```

The command reports elapsed GPU time and throughput in million lattice updates
per second (MLUPS). This mode accelerates the LBM-only pipeline; the complete
LBM-IBM-DEM sedimentation simulation remains CPU-based until the IBM and rigid-
particle operations are ported.
