# CudaLBM-IBM-DEM

A modular research prototype for two-dimensional particle sedimentation using:

- D2Q9 BGK lattice Boltzmann method
- Guo forcing
- four-point direct-forcing immersed boundary method
- translating and rotating circular rigid particle
- hydrodynamic force and torque extraction
- separately evaluated gravity and buoyancy forces
- VTK output for ParaView
- CSV sedimentation diagnostics, including acceleration, force balance, Reynolds number, and kinetic energy

## Current status

The repository contains the CPU sedimentation baseline and a persistent CUDA
implementation of the complete LBM-IBM benchmark, including gravity, buoyancy,
normal wall contact, particle integration, CSV diagnostics, and interval-based
VTK output. Independent kernel tests and repeated CPU-GPU pipeline comparison
are available from the command line.

## Build on macOS or Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/lbm_ibm_dem
```

Generated files:

- `fluid_*.vtk`
- `particle_*.vtk`
- `sedimentation.csv`

## Source structure

```text
include/       public class and function declarations
src/           CPU implementations
cuda/          future CUDA backend
benchmark/     sedimentation benchmark specification and results
docs/          architecture and method notes
results/       generated or curated result files
```

## License

MIT License.

### Particle-wall contact

The benchmark executable includes a normal linear spring-dashpot interaction
with the container walls. The contact force is compressive only and is applied
when the circular particle overlaps a wall. The CSV output records the wall
force, maximum overlap, and active-contact state.

## Validate the sedimentation trajectory

Run the full benchmark first:

```bash
./build/lbm_ibm_dem
```

Then compare `sedimentation.csv` with the published N=100 particle-position data:

```bash
python3 tools/validate_sedimentation.py
```

The validation utility has no third-party Python dependencies. It generates a
comparison CSV, a Markdown error report, and an SVG plot under
`results/sedimentation_validation/`.

## CPU–GPU verification

Build the CUDA configuration and run the full-domain verification mode:

```powershell
cmake -S . -B build-cuda -DENABLE_CUDA=ON -A x64
cmake --build build-cuda --config Release -j
.\build-cuda\Release\lbm_ibm_dem.exe --verify --steps 1000 --interval 100
```

The CPU and CUDA LBM pipelines are advanced in lockstep on the complete
`200 x 600` benchmark lattice. Error histories are written to
`results/gpu_verification.csv`.

For a longer verification run:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe --verify --steps 5000 --interval 100
python tools\plot_gpu_verification.py
```


## Complete CUDA sedimentation

Build the CUDA configuration:

```powershell
cmake -S . -B build-cuda `
    -DENABLE_CUDA=ON `
    -DCMAKE_CUDA_ARCHITECTURES=89 `
    -A x64
cmake --build build-cuda --config Release --parallel
```

Verify the full CPU and CUDA sedimentation paths:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe `
    --verify-cuda-sedimentation `
    --steps 100 `
    --interval 10
```

Run the full CUDA benchmark:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe --gpu-sedimentation
```

The CUDA run writes `sedimentation_gpu.csv`, `fluid_gpu_*.vtk`, and
`particle_gpu_*.vtk`.
