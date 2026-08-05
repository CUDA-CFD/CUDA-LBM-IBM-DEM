#include "IBMHydrodynamicLoads.cuh"

#include "CUDAUtils.cuh"
#include "IBMParticleIntegration.cuh"

#include <cuda_runtime.h>

namespace
{

// -----------------------------------------------------------------------------
// One CUDA thread owns one Lagrangian marker and contributes to the particle's
// total hydrodynamic force and torque.
// -----------------------------------------------------------------------------
__global__ void accumulateIBMHydrodynamicLoadsKernel(
    const double* markerX,
    const double* markerY,
    const double* markerForceX,
    const double* markerForceY,
    const cuda_backend::IBMParticleStateCUDA* particle,
    int markerCount,
    double markerSpacing,
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

    const double relativeX = markerX[marker] - particle->x;
    const double relativeY = markerY[marker] - particle->y;

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
    const IBMParticleStateCUDA* dParticleState,
    int markerCount,
    double markerSpacing,
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
        dParticleState,
        markerCount,
        markerSpacing,
        dParticleForceX,
        dParticleForceY,
        dParticleTorque);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend
