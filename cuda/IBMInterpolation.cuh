#pragma once

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
    int ny);

}