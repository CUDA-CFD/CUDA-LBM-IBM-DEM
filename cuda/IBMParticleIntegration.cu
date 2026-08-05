#include "IBMParticleIntegration.cuh"

#include "CUDAUtils.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace
{

constexpr double pi = 3.14159265358979323846;

// -----------------------------------------------------------------------------
// One CUDA thread advances the single rigid particle.
//
// The update order matches IBMParticle::integrate():
//   1. acceleration from the current total load,
//   2. velocity update,
//   3. position/orientation update using the new velocities.
// -----------------------------------------------------------------------------
__global__ void integrateIBMParticleKernel(
    cuda_backend::IBMParticleStateCUDA* particle,
    const double* particleForceX,
    const double* particleForceY,
    const double* particleTorque,
    double dt)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    particle->ax = particleForceX[0] / particle->mass;
    particle->ay = particleForceY[0] / particle->mass;
    particle->alpha = particleTorque[0] / particle->inertia;

    particle->vx += particle->ax * dt;
    particle->vy += particle->ay * dt;
    particle->omega += particle->alpha * dt;

    particle->x += particle->vx * dt;
    particle->y += particle->vy * dt;
    particle->theta += particle->omega * dt;
}

// -----------------------------------------------------------------------------
// One CUDA thread owns one Lagrangian marker.
//
// Reads:
//   - updated particle centre, orientation, and radius
//
// Writes:
//   - one marker x coordinate
//   - one marker y coordinate
// -----------------------------------------------------------------------------
__global__ void updateIBMMarkerPositionsKernel(
    const cuda_backend::IBMParticleStateCUDA* particle,
    double* markerX,
    double* markerY,
    int markerCount)
{
    const int marker =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (marker >= markerCount)
    {
        return;
    }

    const double angle =
        2.0 * pi * static_cast<double>(marker)
            / static_cast<double>(markerCount)
        + particle->theta;

    markerX[marker] = particle->x + particle->radius * cos(angle);
    markerY[marker] = particle->y + particle->radius * sin(angle);
}

} // namespace

namespace cuda_backend
{

void launchIBMParticleIntegration(
    IBMParticleStateCUDA* dParticleState,
    const double* dParticleForceX,
    const double* dParticleForceY,
    const double* dParticleTorque,
    double* dMarkerX,
    double* dMarkerY,
    int markerCount,
    double dt)
{
    integrateIBMParticleKernel<<<1, 1>>>(
        dParticleState,
        dParticleForceX,
        dParticleForceY,
        dParticleTorque,
        dt);

    CUDA_CHECK(cudaGetLastError());

    if (markerCount <= 0)
    {
        return;
    }

    constexpr int threadsPerBlock = 256;
    const int blocks =
        (markerCount + threadsPerBlock - 1) / threadsPerBlock;

    updateIBMMarkerPositionsKernel<<<blocks, threadsPerBlock>>>(
        dParticleState,
        dMarkerX,
        dMarkerY,
        markerCount);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend
