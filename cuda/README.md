# CUDA backend

The CUDA backend is being ported one verified operation at a time.

Current kernels:

- macroscopic density and velocity reconstruction,
- BGK collision with Guo forcing.

The complete sedimentation time loop still runs on the CPU. Each CUDA
kernel is compared automatically with the production CPU implementation
before the benchmark starts.
