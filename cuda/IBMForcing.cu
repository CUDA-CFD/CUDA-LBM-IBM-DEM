#include "IBMForcing.cuh"

#include "CUDAUtils.cuh"
#include "IBMParticleIntegration.cuh"

#include <cuda_runtime.h>

namespace
{

// -----------------------------------------------------------------------------
// One CUDA thread owns one Lagrangian marker.
//
// Reads:
//   - marker position
//   - interpolated fluid velocity at the marker
//   - persistent rigid-particle state
//
// Writes:
//   - direct-forcing components for that marker
// -----------------------------------------------------------------------------
__global__ void computeIBMDirectForcingKernel(
    const double* markerX,
    const double* markerY,
    const double* fluidUx,
    const double* fluidUy,
    double* markerForceX,
    double* markerForceY,
    const cuda_backend::IBMParticleStateCUDA* particle,
    int markerCount,
    double dt)
{
    const int marker =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (marker >= markerCount)
    {
        return;
    }

    const double relativeX = markerX[marker] - particle->x;
    const double relativeY = markerY[marker] - particle->y;

    const double boundaryVelocityX =
        particle->vx - particle->omega * relativeY;

    const double boundaryVelocityY =
        particle->vy + particle->omega * relativeX;

    markerForceX[marker] =
        (boundaryVelocityX - fluidUx[marker]) / dt;

    markerForceY[marker] =
        (boundaryVelocityY - fluidUy[marker]) / dt;
}

} // namespace

namespace cuda_backend
{

void launchIBMDirectForcing(
    const double* dMarkerX,
    const double* dMarkerY,
    const double* dFluidUx,
    const double* dFluidUy,
    double* dMarkerForceX,
    double* dMarkerForceY,
    const IBMParticleStateCUDA* dParticleState,
    int markerCount,
    double dt)
{
    if (markerCount <= 0)
    {
        return;
    }

    constexpr int threadsPerBlock = 256;
    const int blocks =
        (markerCount + threadsPerBlock - 1) / threadsPerBlock;

    computeIBMDirectForcingKernel<<<blocks, threadsPerBlock>>>(
        dMarkerX,
        dMarkerY,
        dFluidUx,
        dFluidUy,
        dMarkerForceX,
        dMarkerForceY,
        dParticleState,
        markerCount,
        dt);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend
