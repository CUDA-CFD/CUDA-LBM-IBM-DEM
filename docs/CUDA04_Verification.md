# Full-domain CPU–GPU verification

The verification mode evolves the CPU and CUDA D2Q9 fluid solvers in
lockstep on the complete `200 x 600` sedimentation benchmark lattice.
It verifies the CUDA macroscopic, collision, and pull-streaming kernels
as one repeated LBM pipeline rather than checking only a single kernel
launch on a small synthetic domain.

## Execution

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe --verify
```

The number of timesteps and comparison interval are configurable:

```powershell
.\build-cuda\Release\lbm_ibm_dem.exe --verify --steps 5000 --interval 100
```

## Quantities compared

At each verification checkpoint the program compares:

- density,
- horizontal velocity,
- vertical velocity,
- all nine D2Q9 populations.

For every quantity it records:

- maximum absolute error,
- L2 error,
- relative L2 error.

The results are written to:

```text
results/gpu_verification.csv
```

## Why verification is repeated

A one-step comparison detects indexing and equation errors, but it does
not reveal whether small floating-point differences accumulate over time.
Periodic checks show whether CPU–GPU differences remain bounded during
repeated LBM evolution.

The CUDA compiler may use fused multiply-add instructions or a different
floating-point evaluation order from the CPU compiler. Therefore exact
bitwise agreement is not always expected. The verifier uses configurable
numerical tolerances and reports the location and values associated with
the largest discrepancy when a test fails.
