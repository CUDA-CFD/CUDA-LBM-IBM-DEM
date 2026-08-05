#pragma once

#include "IBMParticleIntegration.cuh"

namespace cuda_backend
{

/**
 * @brief Per-timestep external-load diagnostics for the sedimentation case.
 *
 * Hydrodynamic force is captured before gravity, buoyancy, and wall-contact
 * forces are added. The total force values are the loads subsequently used by
 * the particle integrator.
 */
struct IBMSedimentationLoadsCUDA
{
    double hydrodynamicForceX = 0.0;
    double hydrodynamicForceY = 0.0;
    double gravityForceY = 0.0;
    double buoyancyForceY = 0.0;
    double externalForceY = 0.0;
    double wallForceX = 0.0;
    double wallForceY = 0.0;
    double maximumWallOverlap = 0.0;
    double totalForceX = 0.0;
    double totalForceY = 0.0;
    int wallContactActive = 0;
};

/**
 * @brief Add gravity, buoyancy, and normal wall-contact loads on the GPU.
 *
 * One CUDA thread reads the current device-resident particle state and the
 * already accumulated hydrodynamic force. It reproduces the CPU spring-
 * dashpot wall model, adds all external loads to the particle force scalars,
 * and records diagnostics for occasional host output.
 */
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
    IBMSedimentationLoadsCUDA* dDiagnostics);

} // namespace cuda_backend
