#pragma once

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
    double dt);

} // namespace cuda_backend
