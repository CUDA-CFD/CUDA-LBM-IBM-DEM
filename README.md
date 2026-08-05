# CUDA-LBM-IBM-DEM

A verified CPU/CUDA research solver for two-dimensional rigid-particle sedimentation using the lattice Boltzmann method (LBM), immersed boundary method (IBM), and particle dynamics/contact modelling.

## Highlights

- D2Q9 BGK LBM with Guo forcing.
- Four-point direct-forcing IBM for a translating and rotating circular particle.
- Hydrodynamic force and torque accumulation.
- Gravity, buoyancy, and compressive-only linear spring-dashpot wall contact.
- Serial CPU reference implementation and persistent device-resident CUDA implementation.
- Independent CPU–GPU verification of every CUDA stage and repeated full-pipeline verification.
- Validation against the published `N=100` sedimentation reference trajectory.
- Reproducible CPU–CUDA performance benchmark and dependency-free SVG plotting tools.
- CSV trajectory diagnostics and VTK output for ParaView.

## Verified release result

For the included `200 × 600` benchmark with 79 Lagrangian markers, double precision, 10,000 measured timesteps, 500 warm-up steps, and five repetitions:

| Backend | Median runtime | Relative performance |
|---|---:|---:|
| Serial CPU | 78.915 s | 1.00× |
| NVIDIA RTX 4090 | 1.837 s | **42.97×** |

These numbers were measured on the developer system and are not universal hardware guarantees. Use `--benchmark-speedup` to reproduce the measurement on your own machine.

## Requirements

### CPU build

- CMake 3.20 or newer
- A C++17 compiler

### CUDA build

- NVIDIA CUDA Toolkit with a C++17-capable NVCC
- A supported NVIDIA GPU
- CMake 3.20 or newer

The default CUDA architecture is `89` (Ada Lovelace / RTX 4090). Override it for another GPU using `-DCMAKE_CUDA_ARCHITECTURES=<value>`.

## Quick start

### CPU — Linux or macOS

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/lbm_ibm_dem
```

### CUDA — Windows / Visual Studio

```powershell
cmake -S . -B build-cuda `
    -DENABLE_CUDA=ON `
    -DCMAKE_CUDA_ARCHITECTURES=89 `
    -A x64

cmake --build build-cuda --config Release --parallel
```

Check the installed version:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe --version
```

Run the complete CUDA sedimentation benchmark:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe --gpu-sedimentation
```

## Core workflows

### 1. Verify the complete CPU and CUDA sedimentation paths

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe `
    --verify-cuda-sedimentation `
    --steps 100 `
    --interval 10
```

### 2. Generate CPU and CUDA trajectories

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe
.\build-cuda\Release\lbm_ibm_dem.exe --gpu-sedimentation
```

Outputs include:

```text
sedimentation.csv
sedimentation_gpu.csv
fluid_*.vtk
particle_*.vtk
fluid_gpu_*.vtk
particle_gpu_*.vtk
```

### 3. Validate against the sedimentation benchmark

```powershell
python .\tools\validate_sedimentation.py
```

Generated files:

```text
results/sedimentation_validation/validation_comparison.csv
results/sedimentation_validation/validation_report.md
results/sedimentation_validation/position_comparison.svg
results/sedimentation_validation/cpu_gpu_difference.svg
```

### 4. Measure CPU–CUDA speedup

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe `
    --benchmark-speedup `
    --steps 10000 `
    --warmup 500 `
    --repeats 5

python .\tools\plot_speedup.py
```

Generated files:

```text
results/performance/speedup_results.csv
results/performance/speedup_report.md
results/performance/speedup_comparison.svg
```

## Verification commands

| Command | Purpose |
|---|---|
| `--verify` | Repeated full-domain CPU–GPU LBM comparison |
| `--verify-ibm-interpolation` | Multi-case IBM interpolation verification |
| `--verify-ibm-forcing` | Multi-case direct-forcing verification |
| `--verify-ibm-spreading` | Multi-case force-spreading and atomic-contention verification |
| `--verify-ibm-hydrodynamic-loads` | Force/torque accumulation verification |
| `--verify-ibm-particle-integration` | Repeated rigid-particle and marker update verification |
| `--verify-ibm-pipeline` | Persistent coupled LBM–IBM pipeline verification |
| `--verify-cuda-sedimentation` | Complete sedimentation verification including external loads/contact |
| `--gpu-lbm` | Device-resident LBM timing mode |
| `--gpu-sedimentation` | Complete CUDA sedimentation execution |
| `--benchmark-speedup` | Repeated compute-only CPU–CUDA benchmark |

Run `lbm_ibm_dem --help` for supported options.

## Repository layout

```text
include/       public CPU/common declarations
src/           serial CPU implementation
cuda/          CUDA kernels, persistent backends, and GPU verification
benchmark/     sedimentation benchmark specification and reference data
docs/          architecture, verification, validation, and performance notes
tests/         verification interfaces and metrics
tools/         dependency-free validation and SVG plotting utilities
results/       generated reports and plots (ignored except .gitkeep)
```

## Numerical and performance scope

The current release focuses on one validated two-dimensional, single circular-particle sedimentation benchmark. The CPU implementation is the numerical reference. CUDA kernels are verified independently and as a complete persistent pipeline. Performance results exclude allocation, initialization, output, and verification transfers unless stated otherwise.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Benchmark definition and validation](docs/Benchmark.md)
- [Complete CUDA sedimentation](docs/CUDA07_Complete_Sedimentation.md)
- [Sedimentation validation workflow](docs/CUDA08_Sedimentation_Validation.md)
- [Performance benchmark methodology](docs/CUDA09_Performance.md)
- [v1.0.0 release notes](RELEASE_NOTES.md)

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). GitHub can generate BibTeX and other citation formats from this file.

## License

Released under the [MIT License](LICENSE).
