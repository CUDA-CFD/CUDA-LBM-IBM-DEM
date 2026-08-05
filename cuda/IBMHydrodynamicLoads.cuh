#pragma once

namespace cuda_backend
{

struct IBMParticleStateCUDA;

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
    double* dParticleTorque);

} // namespace cuda_backend
