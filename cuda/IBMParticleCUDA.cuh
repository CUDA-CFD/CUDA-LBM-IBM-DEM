#pragma once

#include "CUDAUtils.cuh"
#include "IBMParticleIntegration.cuh"

class IBMParticle;

namespace cuda_backend
{

class LBMSolverCUDA;

/**
 * @brief Persistent device-resident state for one immersed rigid particle.
 *
 * The class owns the particle state, marker coordinates, interpolated marker
 * velocities, marker forces, and the three accumulated particle loads.  The
 * individual verified IBM kernels remain separate; this class only connects
 * them without host-device transfers between operations or timesteps.
 */
class IBMParticleCUDA
{
public:
    explicit IBMParticleCUDA(const IBMParticle& particle);

    IBMParticleCUDA(const IBMParticleCUDA&) = delete;
    IBMParticleCUDA& operator=(const IBMParticleCUDA&) = delete;
    IBMParticleCUDA(IBMParticleCUDA&&) = delete;
    IBMParticleCUDA& operator=(IBMParticleCUDA&&) = delete;

    void initializeFromHost(const IBMParticle& particle);
    void downloadToHost(IBMParticle& particle) const;

    void interpolate(const LBMSolverCUDA& fluid);
    void computeDirectForcing(double dt);
    void spreadToFluid(LBMSolverCUDA& fluid);
    void accumulateHydrodynamicLoads();
    void integrate(double dt);
    void synchronize() const;

    int markerCount() const noexcept { return markerCount_; }
    double markerSpacing() const noexcept { return markerSpacing_; }

    const IBMParticleStateCUDA* deviceState() const noexcept
    {
        return particleState_.data();
    }

    double* deviceForceX() noexcept { return particleForceX_.data(); }
    double* deviceForceY() noexcept { return particleForceY_.data(); }
    double* deviceTorque() noexcept { return particleTorque_.data(); }

private:
    int markerCount_;
    double markerSpacing_;

    DeviceArray<IBMParticleStateCUDA> particleState_;
    DeviceArray<double> markerX_;
    DeviceArray<double> markerY_;
    DeviceArray<double> markerFluidUx_;
    DeviceArray<double> markerFluidUy_;
    DeviceArray<double> markerForceX_;
    DeviceArray<double> markerForceY_;
    DeviceArray<double> particleForceX_;
    DeviceArray<double> particleForceY_;
    DeviceArray<double> particleTorque_;
};

} // namespace cuda_backend
