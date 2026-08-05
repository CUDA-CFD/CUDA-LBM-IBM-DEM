# CUDA-LBM-IBM-DEM v1.0.0

This is the first stable release of the verified CPU/CUDA LBM–IBM particle-sedimentation solver.

## Release highlights

- Complete device-resident CUDA implementation of the validated sedimentation pipeline.
- Serial CPU reference implementation retained for correctness checks.
- Independent and coupled CPU–GPU verification covering LBM, IBM, particle dynamics, external loads, and wall contact.
- Validation against the published `N=100` sedimentation trajectory.
- CSV diagnostics, VTK output, Markdown reports, and dependency-free SVG plots.
- Reproducible compute-only performance benchmark.

## Measured performance

On the developer system with an NVIDIA GeForce RTX 4090:

```text
Domain:            200 × 600
Markers:           79
Measured steps:    10,000
Warm-up steps:     500
Repetitions:       5
CPU median:        78.915 s
CUDA median:       1.837 s
Median speedup:    42.97×
```

Performance depends on CPU, GPU, compiler, CUDA version, clocks, thermal state, and system load. Reproduce locally using:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe `
    --benchmark-speedup --steps 10000 --warmup 500 --repeats 5
```

## Recommended release verification

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe --verify --steps 5000 --interval 100
.\build-cuda\Release\lbm_ibm_dem.exe --verify-ibm-interpolation
.\build-cuda\Release\lbm_ibm_dem.exe --verify-ibm-forcing
.\build-cuda\Release\lbm_ibm_dem.exe --verify-ibm-spreading
.\build-cuda\Release\lbm_ibm_dem.exe --verify-ibm-hydrodynamic-loads
.\build-cuda\Release\lbm_ibm_dem.exe --verify-ibm-particle-integration
.\build-cuda\Release\lbm_ibm_dem.exe --verify-ibm-pipeline --steps 100 --interval 10
.\build-cuda\Release\lbm_ibm_dem.exe --verify-cuda-sedimentation --steps 100 --interval 10
```

## Known scope and limitations

- Two-dimensional D2Q9 formulation.
- One circular rigid particle.
- BGK collision model.
- Single-GPU CUDA backend.
- Normal wall contact only; no particle-particle contact in v1.0.0.
- Current benchmark and performance claim are based on the included sedimentation case.
