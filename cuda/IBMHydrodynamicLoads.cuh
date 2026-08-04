#pragma once

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
    double* dParticleTorque);

} // namespace cuda_backend
