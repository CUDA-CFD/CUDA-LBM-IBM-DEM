#include "IBMHydrodynamicLoads.cuh"

#include "CUDAUtils.cuh"

#include <cuda_runtime.h>

namespace
{

// -----------------------------------------------------------------------------
// One CUDA thread owns one Lagrangian marker.
//
// Reads:
//   - marker position
//   - marker direct-forcing components
//
// Atomically accumulates:
//   - total hydrodynamic force in x
//   - total hydrodynamic force in y
//   - total hydrodynamic torque about the particle centre
//
// Every valid marker thread contributes to the same three output scalars, so
// double-precision atomic additions are required.
// -----------------------------------------------------------------------------
__global__ void accumulateIBMHydrodynamicLoadsKernel(
    const double* markerX,
    const double* markerY,
    const double* markerForceX,
    const double* markerForceY,
    int markerCount,
    double markerSpacing,
    double particleX,
    double particleY,
    double* particleForceX,
    double* particleForceY,
    double* particleTorque)
{
    const int marker =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (marker >= markerCount)
    {
        return;
    }

    const double hydrodynamicForceX =
        -markerForceX[marker] * markerSpacing;

    const double hydrodynamicForceY =
        -markerForceY[marker] * markerSpacing;

    const double relativeX = markerX[marker] - particleX;
    const double relativeY = markerY[marker] - particleY;

    const double hydrodynamicTorque =
        relativeX * hydrodynamicForceY
        - relativeY * hydrodynamicForceX;

    atomicAdd(particleForceX, hydrodynamicForceX);
    atomicAdd(particleForceY, hydrodynamicForceY);
    atomicAdd(particleTorque, hydrodynamicTorque);
}

} // namespace

namespace cuda_backend
{

void launchIBMHydrodynamicLoadAccumulation(
    const double* dMarkerX,
    const double* dMarkerY,
    const double* dMarkerForceX,
    const double* dMarkerForceY,
    int markerCount,
    double markerSpacing,
    double particleX,
    double particleY,
    double* dParticleForceX,
    double* dParticleForceY,
    double* dParticleTorque)
{
    CUDA_CHECK(cudaMemset(dParticleForceX, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(dParticleForceY, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(dParticleTorque, 0, sizeof(double)));

    if (markerCount <= 0)
    {
        return;
    }

    constexpr int threadsPerBlock = 256;
    const int blocks =
        (markerCount + threadsPerBlock - 1) / threadsPerBlock;

    accumulateIBMHydrodynamicLoadsKernel<<<blocks, threadsPerBlock>>>(
        dMarkerX,
        dMarkerY,
        dMarkerForceX,
        dMarkerForceY,
        markerCount,
        markerSpacing,
        particleX,
        particleY,
        dParticleForceX,
        dParticleForceY,
        dParticleTorque);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend
