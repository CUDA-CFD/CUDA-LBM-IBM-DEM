#include "IBMInterpolation.cuh"

#include "CUDAUtils.cuh"
#include "DeltaKernel.hpp"

#include <cuda_runtime.h>

namespace
{

// -----------------------------------------------------------------------------
// One CUDA thread owns one Lagrangian marker.
//
// Reads:
//   - marker position
//   - Eulerian velocity field on a 4 x 4 stencil
//
// Writes:
//   - interpolated x- and y-velocity for that marker
//
// Each thread writes to a unique marker entry, so no atomics or synchronization
// are required.
// -----------------------------------------------------------------------------
__global__ void interpolateIBMKernel(
    const double* markerX,
    const double* markerY,
    double* markerFluidUx,
    double* markerFluidUy,
    const double* velocityX,
    const double* velocityY,
    int markerCount,
    int nx,
    int ny)
{
    const int marker =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (marker >= markerCount)
    {
        return;
    }

    const double xMarker = markerX[marker];
    const double yMarker = markerY[marker];

    const int baseX = static_cast<int>(floor(xMarker));
    const int baseY = static_cast<int>(floor(yMarker));

    double interpolatedUx = 0.0;
    double interpolatedUy = 0.0;

    for (int y = baseY - 1; y <= baseY + 2; ++y)
    {
        for (int x = baseX - 1; x <= baseX + 2; ++x)
        {
            if (x < 0 || x >= nx || y < 0 || y >= ny)
            {
                continue;
            }

            const double weight =
                ibm::deltaKernel(x - xMarker) *
                ibm::deltaKernel(y - yMarker);

            const int node = y * nx + x;

            interpolatedUx += velocityX[node] * weight;
            interpolatedUy += velocityY[node] * weight;
        }
    }

    markerFluidUx[marker] = interpolatedUx;
    markerFluidUy[marker] = interpolatedUy;
}

} // namespace

namespace cuda_backend
{

void launchIBMInterpolation(
    const double* dMarkerX,
    const double* dMarkerY,
    double* dFluidUx,
    double* dFluidUy,
    const double* dVelocityX,
    const double* dVelocityY,
    int markerCount,
    int nx,
    int ny)
{
    if (markerCount <= 0)
    {
        return;
    }

    constexpr int threadsPerBlock = 256;
    const int blocks =
        (markerCount + threadsPerBlock - 1) / threadsPerBlock;

    interpolateIBMKernel<<<blocks, threadsPerBlock>>>(
        dMarkerX,
        dMarkerY,
        dFluidUx,
        dFluidUy,
        dVelocityX,
        dVelocityY,
        markerCount,
        nx,
        ny);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend
