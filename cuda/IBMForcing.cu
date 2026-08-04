#include "IBMForcing.cuh"

#include "CUDAUtils.cuh"

#include <cuda_runtime.h>

namespace
{

// -----------------------------------------------------------------------------
// One CUDA thread owns one Lagrangian marker.
//
// Reads:
//   - marker position
//   - interpolated fluid velocity at the marker
//   - rigid-particle translational and angular velocity
//
// Writes:
//   - direct-forcing components for that marker
//
// Each thread writes to a unique marker entry, so no atomics or synchronization
// are required.
// -----------------------------------------------------------------------------
__global__ void computeIBMDirectForcingKernel(
    const double* markerX,
    const double* markerY,
    const double* fluidUx,
    const double* fluidUy,
    double* markerForceX,
    double* markerForceY,
    int markerCount,
    double particleX,
    double particleY,
    double particleVelocityX,
    double particleVelocityY,
    double particleAngularVelocity,
    double dt)
{
    const int marker =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (marker >= markerCount)
    {
        return;
    }

    const double relativeX = markerX[marker] - particleX;
    const double relativeY = markerY[marker] - particleY;

    const double boundaryVelocityX =
        particleVelocityX - particleAngularVelocity * relativeY;

    const double boundaryVelocityY =
        particleVelocityY + particleAngularVelocity * relativeX;

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
    int markerCount,
    double particleX,
    double particleY,
    double particleVelocityX,
    double particleVelocityY,
    double particleAngularVelocity,
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
        markerCount,
        particleX,
        particleY,
        particleVelocityX,
        particleVelocityY,
        particleAngularVelocity,
        dt);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend
