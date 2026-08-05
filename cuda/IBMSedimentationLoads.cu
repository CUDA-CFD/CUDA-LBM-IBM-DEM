#include "IBMSedimentationLoads.cuh"

#include "CUDAUtils.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace
{

__device__ double maximum(double left, double right)
{
    return left > right ? left : right;
}

__device__ double compressiveForce(
    double overlap,
    double normalVelocityIntoWall,
    double stiffness,
    double damping)
{
    if (overlap <= 0.0)
    {
        return 0.0;
    }

    // Match the CPU clamp: contact is compressive only and cannot pull the
    // particle toward the wall while it is separating.
    return maximum(
        0.0,
        stiffness * overlap + damping * normalVelocityIntoWall
    );
}

// -----------------------------------------------------------------------------
// One CUDA thread owns the single rigid particle.
//
// Reads:
//   - current particle position, velocity, and radius
//   - hydrodynamic force already accumulated from IBM markers
//
// Writes:
//   - total particle force including gravity, buoyancy, and wall contact
//   - one compact diagnostics structure
// -----------------------------------------------------------------------------
__global__ void applyIBMSedimentationLoadsKernel(
    const cuda_backend::IBMParticleStateCUDA* particle,
    double* particleForceX,
    double* particleForceY,
    double gravityForceY,
    double buoyancyForceY,
    double domainWidth,
    double domainHeight,
    double wallNormalStiffness,
    double wallNormalDamping,
    cuda_backend::IBMSedimentationLoadsCUDA* diagnostics)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    const double hydrodynamicForceX = particleForceX[0];
    const double hydrodynamicForceY = particleForceY[0];
    const double externalForceY = gravityForceY + buoyancyForceY;

    // Bottom wall: inward normal is +y, and -vy is approach speed.
    const double bottomOverlap = particle->radius - particle->y;
    const double bottomForce = compressiveForce(
        bottomOverlap,
        -particle->vy,
        wallNormalStiffness,
        wallNormalDamping
    );

    // Top wall: inward normal is -y, and +vy is approach speed.
    const double topOverlap =
        particle->y + particle->radius - domainHeight;
    const double topForce = compressiveForce(
        topOverlap,
        particle->vy,
        wallNormalStiffness,
        wallNormalDamping
    );

    // Left wall: inward normal is +x, and -vx is approach speed.
    const double leftOverlap = particle->radius - particle->x;
    const double leftForce = compressiveForce(
        leftOverlap,
        -particle->vx,
        wallNormalStiffness,
        wallNormalDamping
    );

    // Right wall: inward normal is -x, and +vx is approach speed.
    const double rightOverlap =
        particle->x + particle->radius - domainWidth;
    const double rightForce = compressiveForce(
        rightOverlap,
        particle->vx,
        wallNormalStiffness,
        wallNormalDamping
    );

    const double wallForceX = leftForce - rightForce;
    const double wallForceY = bottomForce - topForce;

    // Preserve the CPU addition order:
    //   hydrodynamic -> gravity+buoyancy -> wall contact.
    double totalForceX = hydrodynamicForceX;
    double totalForceY = hydrodynamicForceY;
    totalForceY += externalForceY;
    totalForceX += wallForceX;
    totalForceY += wallForceY;

    particleForceX[0] = totalForceX;
    particleForceY[0] = totalForceY;

    const double maximumOverlap = maximum(
        0.0,
        maximum(
            maximum(bottomOverlap, topOverlap),
            maximum(leftOverlap, rightOverlap)
        )
    );

    diagnostics->hydrodynamicForceX = hydrodynamicForceX;
    diagnostics->hydrodynamicForceY = hydrodynamicForceY;
    diagnostics->gravityForceY = gravityForceY;
    diagnostics->buoyancyForceY = buoyancyForceY;
    diagnostics->externalForceY = externalForceY;
    diagnostics->wallForceX = wallForceX;
    diagnostics->wallForceY = wallForceY;
    diagnostics->maximumWallOverlap = maximumOverlap;
    diagnostics->totalForceX = totalForceX;
    diagnostics->totalForceY = totalForceY;
    diagnostics->wallContactActive = maximumOverlap > 0.0 ? 1 : 0;
}

} // namespace

namespace cuda_backend
{

void launchIBMSedimentationLoads(
    const IBMParticleStateCUDA* dParticleState,
    double* dParticleForceX,
    double* dParticleForceY,
    double gravityForceY,
    double buoyancyForceY,
    double domainWidth,
    double domainHeight,
    double wallNormalStiffness,
    double wallNormalDamping,
    IBMSedimentationLoadsCUDA* dDiagnostics)
{
    applyIBMSedimentationLoadsKernel<<<1, 1>>>(
        dParticleState,
        dParticleForceX,
        dParticleForceY,
        gravityForceY,
        buoyancyForceY,
        domainWidth,
        domainHeight,
        wallNormalStiffness,
        wallNormalDamping,
        dDiagnostics
    );

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend
