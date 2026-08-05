#pragma once

namespace cuda_backend
{

struct IBMParticleStateCUDA {
    double x = 0.0;
    double y = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double ax = 0.0;
    double ay = 0.0;
    double theta = 0.0;
    double omega = 0.0;
    double alpha = 0.0;
    double radius = 0.0;
    double mass = 0.0;
    double inertia = 0.0;
};

/**
 * @brief Advance one rigid particle and regenerate its Lagrangian markers.
 *
 * The first CUDA kernel updates the particle state with the same semi-implicit
 * Euler sequence used by IBMParticle::integrate(). A second kernel then maps
 * one CUDA thread to one marker and updates the marker coordinates from the
 * new particle position and orientation.
 *
 * Both kernels are launched on the default stream, so the marker update begins
 * only after the scalar particle-state update has completed.
 */
void launchIBMParticleIntegration(
    IBMParticleStateCUDA* dParticleState,
    const double* dParticleForceX,
    const double* dParticleForceY,
    const double* dParticleTorque,
    double* dMarkerX,
    double* dMarkerY,
    int markerCount,
    double dt);

} // namespace cuda_backend
