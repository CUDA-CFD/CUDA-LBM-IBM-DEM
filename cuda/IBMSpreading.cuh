#pragma once

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
    double markerSpacing);

} // namespace cuda_backend
