#include "IBMSpreading.cuh"

#include "CUDAUtils.cuh"
#include "DeltaKernel.hpp"

#include <cuda_runtime.h>

namespace
{

// -----------------------------------------------------------------------------
// One CUDA thread owns one Lagrangian marker.
//
// Reads:
//   - marker position and direct-forcing components
//
// Accumulates:
//   - force contributions on the marker's 4 x 4 Eulerian stencil
//
// Different markers can contribute to the same fluid node. Double-precision
// atomic additions are therefore required to prevent lost updates.
// -----------------------------------------------------------------------------
__global__ void spreadIBMForcesKernel(
    const double* markerX,
    const double* markerY,
    const double* markerForceX,
    const double* markerForceY,
    double* forceX,
    double* forceY,
    int markerCount,
    int nx,
    int ny,
    double markerSpacing)
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

    const double scaledForceX = markerForceX[marker] * markerSpacing;
    const double scaledForceY = markerForceY[marker] * markerSpacing;

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

            atomicAdd(&forceX[node], scaledForceX * weight);
            atomicAdd(&forceY[node], scaledForceY * weight);
        }
    }
}

} // namespace

namespace cuda_backend
{

void launchIBMSpreading(
    const double* dMarkerX,
    const double* dMarkerY,
    const double* dMarkerForceX,
    const double* dMarkerForceY,
    double* dForceX,
    double* dForceY,
    int markerCount,
    int nx,
    int ny,
    double markerSpacing)
{
    if (markerCount <= 0)
    {
        return;
    }

    constexpr int threadsPerBlock = 256;
    const int blocks =
        (markerCount + threadsPerBlock - 1) / threadsPerBlock;

    spreadIBMForcesKernel<<<blocks, threadsPerBlock>>>(
        dMarkerX,
        dMarkerY,
        dMarkerForceX,
        dMarkerForceY,
        dForceX,
        dForceY,
        markerCount,
        nx,
        ny,
        markerSpacing);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend
