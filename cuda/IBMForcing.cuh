#pragma once

namespace cuda_backend
{

struct IBMParticleStateCUDA;

void launchIBMDirectForcing(
    const double* dMarkerX,
    const double* dMarkerY,
    const double* dFluidUx,
    const double* dFluidUy,
    double* dMarkerForceX,
    double* dMarkerForceY,
    const IBMParticleStateCUDA* dParticleState,
    int markerCount,
    double dt);

} // namespace cuda_backend
