# CUDA09 — Complete-pipeline performance and speedup

## Purpose

The performance mode compares the validated serial CPU and persistent CUDA
implementations of the complete sedimentation physics timestep:

1. clear Eulerian forces,
2. reconstruct macroscopic LBM fields,
3. interpolate velocity to IBM markers,
4. compute direct marker forcing,
5. spread marker forces to the fluid,
6. accumulate hydrodynamic force and torque,
7. apply gravity, buoyancy, and wall contact,
8. perform LBM collision and pull streaming,
9. integrate the rigid particle and regenerate marker positions.

Both backends start each repeat from the same benchmark initial condition.

## Timing scope

The benchmark is deliberately **compute-only**. The timed region excludes:

- CUDA context creation,
- memory allocation,
- CPU and GPU initialization,
- initial host-to-device copies,
- CSV and VTK output,
- verification downloads,
- console output inside the timestep loop.

CUDA uses synchronized host wall-clock timing. A device synchronization occurs
immediately before the start time and immediately after the final timestep, so
the measurement includes kernel-launch overhead and cannot finish while GPU work
is still pending.

## Run

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe `
    --benchmark-speedup `
    --steps 10000 `
    --warmup 500 `
    --repeats 5
```

`--warmup 0` disables warm-up. Each measured repeat constructs a fresh CPU or
CUDA simulation outside the timed region. CPU-first and CUDA-first order is
alternated between repeats to reduce systematic thermal or background-load bias.

The same default benchmark is available through CMake:

```powershell
cmake --build build-cuda --target benchmark_speedup --config Release
```

## Results

The executable writes:

```text
results/performance/speedup_results.csv
results/performance/speedup_report.md
```

The CSV contains per-repeat runtime, milliseconds per timestep, timesteps per
second, and CPU/CUDA speedup. The report uses the ratio of median CPU and CUDA
runtimes as the headline result:

```text
median speedup = median CPU runtime / median CUDA runtime
```

Generate the SVG comparison plot with:

```powershell
python tools\plot_speedup.py
```

or:

```powershell
cmake --build build-cuda --target plot_speedup --config Release
```

The generated figure is:

```text
results/performance/speedup_comparison.svg
```

## Interpretation

This result measures acceleration of the numerical timestep pipeline. It is not
an output-inclusive or end-to-end workflow benchmark. CSV/VTK writing can be
measured separately because its cost depends strongly on the requested output
interval, filesystem, and storage device.
