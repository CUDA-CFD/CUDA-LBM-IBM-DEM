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

The repository currently contains the **CPU sedimentation baseline** with explicit gravity, buoyancy, acceleration, force-balance, Reynolds-number, and kinetic-energy diagnostics. It is not yet the final literature-validated release. The next stages are:

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

### Particle-wall contact

The benchmark executable includes a normal linear spring-dashpot interaction
with the container walls. The contact force is compressive only and is applied
when the circular particle overlaps a wall. The CSV output records the wall
force, maximum overlap, and active-contact state.
