# Complete CUDA sedimentation pipeline

Commit 022 connects the benchmark-specific external loads to the persistent
CUDA LBM–IBM pipeline.

## Device timestep

Each timestep executes in this order:

1. clear Eulerian force arrays;
2. reconstruct LBM macroscopic fields;
3. interpolate fluid velocity to IBM markers;
4. compute direct marker forcing;
5. spread marker force to the Eulerian lattice;
6. accumulate hydrodynamic particle force and torque;
7. add gravity, buoyancy, and normal wall-contact forces;
8. reconstruct force-corrected macroscopic fields;
9. collide and pull-stream the LBM populations;
10. integrate the particle and regenerate marker coordinates.

The fluid arrays, particle state, marker coordinates, marker velocities, marker
forces, and particle load scalars remain device-resident between these steps.

## Host output

`--gpu-sedimentation` writes `sedimentation_gpu.csv`. A compact rigid-particle
state and load-diagnostics structure are copied to the host once per timestep
for this trajectory file. Full fluid velocity fields and marker arrays are
copied only at the VTK output interval.

This keeps large device-to-host transfers out of ordinary timesteps while
retaining output compatible with the CPU benchmark.

## Commands

Verify the complete CPU and CUDA paths in lockstep:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe `
    --verify-cuda-sedimentation `
    --steps 100 `
    --interval 10
```

Run the complete CUDA benchmark:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe --gpu-sedimentation
```

Use custom run and VTK intervals:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe `
    --gpu-sedimentation `
    --steps 10000 `
    --interval 500
```

## Benchmark plotting and validation

After generating both CPU and CUDA trajectory CSV files, run:

```powershell
python tools\validate_sedimentation.py
```

The script overlays the published reference, CPU, and CUDA particle-centre
trajectories and creates a separate CUDA-minus-CPU difference plot. See
`CUDA08_Sedimentation_Validation.md` for the complete output description.
