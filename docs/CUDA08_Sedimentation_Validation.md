# CPU-CUDA sedimentation benchmark validation

Commit 023 extends the original single-trajectory validation utility so the
published N=100 reference, CPU result, and CUDA result are assessed together.

## Required inputs

Run the complete one-second benchmark with both backends:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe
.\build-cuda\Release\lbm_ibm_dem.exe --gpu-sedimentation
```

This generates:

```text
sedimentation.csv
sedimentation_gpu.csv
```

The reference trajectory is stored in:

```text
benchmark/sedimentation/reference_position_N100.csv
```

## Validation command

```powershell
python tools\validate_sedimentation.py
```

Equivalent explicit arguments are:

```powershell
python tools\validate_sedimentation.py `
    --cpu sedimentation.csv `
    --gpu sedimentation_gpu.csv `
    --reference benchmark\sedimentation\reference_position_N100.csv `
    --output-dir results\sedimentation_validation
```

The script uses only the Python standard library.

## Metrics

At each of the eleven published reference times, both simulated trajectories
are linearly interpolated. For CPU and CUDA separately, the script reports:

- root-mean-square error;
- relative root-mean-square error;
- mean absolute error;
- maximum absolute error.

The script also interpolates the CUDA trajectory at every CPU output time in
the shared time range and reports direct CPU-CUDA RMSE, relative L2 difference,
mean absolute difference, and maximum absolute difference.

## Generated files

```text
results/sedimentation_validation/
├── validation_comparison.csv
├── validation_report.md
├── position_comparison.svg
└── cpu_gpu_difference.svg
```

`position_comparison.svg` overlays the published reference points with the full
CPU and CUDA trajectories. `cpu_gpu_difference.svg` plots CUDA height minus CPU
height, making differences visible even when the two trajectory curves overlap.

The comparison CSV contains one row for each published reference time and the
columns:

```text
time_s
reference_y_cm
cpu_y_cm
cpu_error_cm
gpu_y_cm
gpu_error_cm
gpu_minus_cpu_cm
```

## CMake target

After both trajectory files exist, the same validation can be run through:

```powershell
cmake --build build-cuda --target validate_sedimentation --config Release
```
