# CudaLBM-IBM-DEM

A modular research prototype for two-dimensional particle sedimentation using:

- D2Q9 BGK lattice Boltzmann method
- Guo forcing
- four-point direct-forcing immersed boundary method
- translating and rotating circular rigid particle
- hydrodynamic force and torque extraction
- separately evaluated gravity and buoyancy forces
- VTK output for ParaView
- CSV sedimentation diagnostics

## Current status

The repository currently contains the **CPU sedimentation baseline** with explicit gravity and buoyancy force bookkeeping. It is not yet the final literature-validated release. The next stages are:

1. lock one published sedimentation benchmark;
2. add the required moving-particle/fictitious-fluid correction;
3. validate the CPU solution;
4. add the CUDA backend;
5. compare CPU and GPU accuracy and runtime.

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
