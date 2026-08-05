#include "IBMParticleCUDA.cuh"

#include "IBMHydrodynamicLoads.cuh"
#include "IBMForcing.cuh"
#include "IBMInterpolation.cuh"
#include "IBMParticle.hpp"
#include "IBMSpreading.cuh"
#include "LBMSolverCUDA.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace
{

cuda_backend::IBMParticleStateCUDA makeDeviceState(
    const IBMParticle& particle)
{
    cuda_backend::IBMParticleStateCUDA state;
    state.x = particle.x;
    state.y = particle.y;
    state.vx = particle.vx;
    state.vy = particle.vy;
    state.ax = particle.ax;
    state.ay = particle.ay;
    state.theta = particle.theta;
    state.omega = particle.omega;
    state.alpha = particle.alpha;
    state.radius = particle.radius;
    state.mass = particle.mass;
    state.inertia = particle.inertia;
    return state;
}

void requireMatchingParticle(
    const IBMParticle& particle,
    int markerCount,
    double markerSpacing)
{
    if (particle.markerCount != markerCount)
    {
        throw std::invalid_argument(
            "Host and CUDA IBM marker counts do not match."
        );
    }

    if (particle.ds != markerSpacing)
    {
        throw std::invalid_argument(
            "Host and CUDA IBM marker spacing does not match."
        );
    }
}

} // namespace

namespace cuda_backend
{

IBMParticleCUDA::IBMParticleCUDA(const IBMParticle& particle)
    : markerCount_(particle.markerCount),
      markerSpacing_(particle.ds),
      particleState_(1),
      markerX_(static_cast<std::size_t>(particle.markerCount)),
      markerY_(static_cast<std::size_t>(particle.markerCount)),
      markerFluidUx_(static_cast<std::size_t>(particle.markerCount)),
      markerFluidUy_(static_cast<std::size_t>(particle.markerCount)),
      markerForceX_(static_cast<std::size_t>(particle.markerCount)),
      markerForceY_(static_cast<std::size_t>(particle.markerCount)),
      particleForceX_(1),
      particleForceY_(1),
      particleTorque_(1)
{
    if (markerCount_ <= 0 || markerSpacing_ <= 0.0)
    {
        throw std::invalid_argument("Invalid CUDA IBM particle geometry.");
    }

    initializeFromHost(particle);
}

void IBMParticleCUDA::initializeFromHost(const IBMParticle& particle)
{
    requireMatchingParticle(particle, markerCount_, markerSpacing_);

    particleState_.copyFromHost({makeDeviceState(particle)});
    markerX_.copyFromHost(particle.markerX);
    markerY_.copyFromHost(particle.markerY);
    markerFluidUx_.copyFromHost(particle.fluidUx);
    markerFluidUy_.copyFromHost(particle.fluidUy);
    markerForceX_.copyFromHost(particle.markerForceX);
    markerForceY_.copyFromHost(particle.markerForceY);
    particleForceX_.copyFromHost({particle.forceX});
    particleForceY_.copyFromHost({particle.forceY});
    particleTorque_.copyFromHost({particle.torque});
}

void IBMParticleCUDA::downloadToHost(IBMParticle& particle) const
{
    requireMatchingParticle(particle, markerCount_, markerSpacing_);

    std::vector<IBMParticleStateCUDA> stateValues;
    std::vector<double> forceXValues;
    std::vector<double> forceYValues;
    std::vector<double> torqueValues;

    particleState_.copyToHost(stateValues);
    markerX_.copyToHost(particle.markerX);
    markerY_.copyToHost(particle.markerY);
    markerFluidUx_.copyToHost(particle.fluidUx);
    markerFluidUy_.copyToHost(particle.fluidUy);
    markerForceX_.copyToHost(particle.markerForceX);
    markerForceY_.copyToHost(particle.markerForceY);
    particleForceX_.copyToHost(forceXValues);
    particleForceY_.copyToHost(forceYValues);
    particleTorque_.copyToHost(torqueValues);

    const IBMParticleStateCUDA& state = stateValues.front();
    particle.x = state.x;
    particle.y = state.y;
    particle.vx = state.vx;
    particle.vy = state.vy;
    particle.ax = state.ax;
    particle.ay = state.ay;
    particle.theta = state.theta;
    particle.omega = state.omega;
    particle.alpha = state.alpha;
    particle.radius = state.radius;
    particle.mass = state.mass;
    particle.inertia = state.inertia;
    particle.forceX = forceXValues.front();
    particle.forceY = forceYValues.front();
    particle.torque = torqueValues.front();
}

void IBMParticleCUDA::interpolate(const LBMSolverCUDA& fluid)
{
    launchIBMInterpolation(
        markerX_.data(),
        markerY_.data(),
        markerFluidUx_.data(),
        markerFluidUy_.data(),
        fluid.deviceVelocityX(),
        fluid.deviceVelocityY(),
        markerCount_,
        fluid.nx(),
        fluid.ny()
    );
}

void IBMParticleCUDA::computeDirectForcing(double dt)
{
    launchIBMDirectForcing(
        markerX_.data(),
        markerY_.data(),
        markerFluidUx_.data(),
        markerFluidUy_.data(),
        markerForceX_.data(),
        markerForceY_.data(),
        particleState_.data(),
        markerCount_,
        dt
    );
}

void IBMParticleCUDA::spreadToFluid(LBMSolverCUDA& fluid)
{
    launchIBMSpreading(
        markerX_.data(),
        markerY_.data(),
        markerForceX_.data(),
        markerForceY_.data(),
        fluid.deviceForceX(),
        fluid.deviceForceY(),
        markerCount_,
        fluid.nx(),
        fluid.ny(),
        markerSpacing_
    );
}

void IBMParticleCUDA::accumulateHydrodynamicLoads()
{
    launchIBMHydrodynamicLoadAccumulation(
        markerX_.data(),
        markerY_.data(),
        markerForceX_.data(),
        markerForceY_.data(),
        particleState_.data(),
        markerCount_,
        markerSpacing_,
        particleForceX_.data(),
        particleForceY_.data(),
        particleTorque_.data()
    );
}

void IBMParticleCUDA::integrate(double dt)
{
    launchIBMParticleIntegration(
        particleState_.data(),
        particleForceX_.data(),
        particleForceY_.data(),
        particleTorque_.data(),
        markerX_.data(),
        markerY_.data(),
        markerCount_,
        dt
    );
}

void IBMParticleCUDA::synchronize() const
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace cuda_backend
