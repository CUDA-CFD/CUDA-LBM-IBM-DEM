# Single-particle sedimentation benchmark

The benchmark reproduces the N=100 grid case reported in Table III of:

D. Pu, M. Li, L. Shen, Z. Wang, and Z. Li, “The effects of channel width on particle sedimentation in fluids using a coupled lattice Boltzmann-discrete element model,” *Physics of Fluids* 35, 053307 (2023). DOI: 10.1063/5.0147826.

The reference file contains the particle-centre vertical coordinate from 0 to 1 s. After running the solver, execute:

```bash
python3 tools/validate_sedimentation.py
```

or, after CMake configuration:

```bash
cmake --build build --target validate_sedimentation
```

Generated files are placed under `results/sedimentation_validation/`:

- `validation_comparison.csv`
- `validation_report.md`
- `position_comparison.svg`
