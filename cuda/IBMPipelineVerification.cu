#include "IBMPipelineVerification.hpp"

#include "CUDAUtils.cuh"
#include "IBMCoupling.hpp"
#include "IBMParticle.hpp"
#include "IBMParticleCUDA.cuh"
#include "LBMSolver.hpp"
#include "LBMSolverCUDA.cuh"
#include "SedimentationBenchmark.hpp"
#include "VerificationMetrics.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct PipelineSnapshot
{
    ArrayErrorMetrics populations;
    ArrayErrorMetrics density;
    ArrayErrorMetrics velocityX;
    ArrayErrorMetrics velocityY;
    ArrayErrorMetrics forceX;
    ArrayErrorMetrics forceY;
    ArrayErrorMetrics particleState;
    ArrayErrorMetrics markerX;
    ArrayErrorMetrics markerY;
    ArrayErrorMetrics markerFluidUx;
    ArrayErrorMetrics markerFluidUy;
    ArrayErrorMetrics markerForceX;
    ArrayErrorMetrics markerForceY;
};

bool exceedsTolerance(
    const ArrayErrorMetrics& metrics,
    const IBMPipelineVerificationSettings& settings)
{
    return metrics.maxAbsolute > settings.maximumAbsoluteTolerance
        && metrics.relativeL2 > settings.relativeL2Tolerance;
}

std::vector<double> particleStateValues(const IBMParticle& particle)
{
    return {
        particle.x,
        particle.y,
        particle.vx,
        particle.vy,
        particle.ax,
        particle.ay,
        particle.theta,
        particle.omega,
        particle.alpha,
        particle.forceX,
        particle.forceY,
        particle.torque
    };
}

const char* particleStateName(std::size_t index)
{
    static constexpr const char* names[] = {
        "x", "y", "vx", "vy", "ax", "ay",
        "theta", "omega", "alpha", "forceX", "forceY", "torque"
    };

    return index < 12 ? names[index] : "unknown";
}

void advanceCpuPipeline(
    LBMSolver& fluid,
    IBMParticle& particle,
    double dt)
{
    fluid.clearForces();
    fluid.computeMacroscopic();

    ibm::interpolate(fluid, particle);
    ibm::computeDirectForcing(particle, dt);
    ibm::spreadToFluid(fluid, particle);
    ibm::accumulateHydrodynamicLoads(particle);

    fluid.computeMacroscopic();
    fluid.collideAndForce();
    fluid.stream();

    particle.integrate(dt);
}

void advanceGpuPipeline(
    cuda_backend::LBMSolverCUDA& fluid,
    cuda_backend::IBMParticleCUDA& particle,
    double dt)
{
    fluid.clearForces();
    fluid.computeMacroscopic();

    particle.interpolate(fluid);
    particle.computeDirectForcing(dt);
    particle.spreadToFluid(fluid);
    particle.accumulateHydrodynamicLoads();

    fluid.computeMacroscopic();
    fluid.collideAndForce();
    fluid.stream();

    particle.integrate(dt);
}

PipelineSnapshot collectSnapshot(
    const LBMSolver& cpuFluid,
    const IBMParticle& cpuParticle,
    const cuda_backend::LBMSolverCUDA& gpuFluid,
    const cuda_backend::IBMParticleCUDA& gpuParticle)
{
    const cuda_backend::LBMHostState gpuFluidState =
        gpuFluid.downloadState();

    std::vector<double> gpuForceX;
    std::vector<double> gpuForceY;
    gpuFluid.downloadForces(gpuForceX, gpuForceY);

    IBMParticle gpuParticleState(
        cpuParticle.x,
        cpuParticle.y,
        cpuParticle.radius,
        cpuParticle.density
    );
    gpuParticle.downloadToHost(gpuParticleState);

    PipelineSnapshot snapshot;
    snapshot.populations = calculateArrayErrorMetrics(
        cpuFluid.populations(), gpuFluidState.populations
    );
    snapshot.density = calculateArrayErrorMetrics(
        cpuFluid.rho(), gpuFluidState.density
    );
    snapshot.velocityX = calculateArrayErrorMetrics(
        cpuFluid.ux(), gpuFluidState.velocityX
    );
    snapshot.velocityY = calculateArrayErrorMetrics(
        cpuFluid.uy(), gpuFluidState.velocityY
    );
    snapshot.forceX = calculateArrayErrorMetrics(
        cpuFluid.forceX(), gpuForceX
    );
    snapshot.forceY = calculateArrayErrorMetrics(
        cpuFluid.forceY(), gpuForceY
    );
    snapshot.particleState = calculateArrayErrorMetrics(
        particleStateValues(cpuParticle),
        particleStateValues(gpuParticleState)
    );
    snapshot.markerX = calculateArrayErrorMetrics(
        cpuParticle.markerX, gpuParticleState.markerX
    );
    snapshot.markerY = calculateArrayErrorMetrics(
        cpuParticle.markerY, gpuParticleState.markerY
    );
    snapshot.markerFluidUx = calculateArrayErrorMetrics(
        cpuParticle.fluidUx, gpuParticleState.fluidUx
    );
    snapshot.markerFluidUy = calculateArrayErrorMetrics(
        cpuParticle.fluidUy, gpuParticleState.fluidUy
    );
    snapshot.markerForceX = calculateArrayErrorMetrics(
        cpuParticle.markerForceX, gpuParticleState.markerForceX
    );
    snapshot.markerForceY = calculateArrayErrorMetrics(
        cpuParticle.markerForceY, gpuParticleState.markerForceY
    );
    return snapshot;
}

void reportFailure(
    const char* quantity,
    const ArrayErrorMetrics& metrics,
    int step)
{
    std::cerr
        << "Persistent IBM pipeline verification failure\n"
        << "  step: " << step << '\n'
        << "  quantity: " << quantity << '\n'
        << "  index: " << metrics.maxIndex << '\n'
        << "  CPU value: " << metrics.referenceValue << '\n'
        << "  GPU value: " << metrics.candidateValue << '\n'
        << "  maximum absolute error: " << metrics.maxAbsolute << '\n'
        << "  relative L2 error: " << metrics.relativeL2 << '\n';
}

bool checkSnapshot(
    const PipelineSnapshot& snapshot,
    const IBMPipelineVerificationSettings& settings,
    int step)
{
    struct Quantity
    {
        const char* name;
        const ArrayErrorMetrics* metrics;
    };

    const Quantity quantities[] = {
        {"populations", &snapshot.populations},
        {"density", &snapshot.density},
        {"velocityX", &snapshot.velocityX},
        {"velocityY", &snapshot.velocityY},
        {"fluid forceX", &snapshot.forceX},
        {"fluid forceY", &snapshot.forceY},
        {"particle state", &snapshot.particleState},
        {"marker X", &snapshot.markerX},
        {"marker Y", &snapshot.markerY},
        {"marker fluid Ux", &snapshot.markerFluidUx},
        {"marker fluid Uy", &snapshot.markerFluidUy},
        {"marker force X", &snapshot.markerForceX},
        {"marker force Y", &snapshot.markerForceY}
    };

    for (const Quantity& quantity : quantities)
    {
        if (exceedsTolerance(*quantity.metrics, settings))
        {
            reportFailure(quantity.name, *quantity.metrics, step);
            if (quantity.metrics == &snapshot.particleState)
            {
                std::cerr
                    << "  particle component: "
                    << particleStateName(quantity.metrics->maxIndex)
                    << '\n';
            }
            return false;
        }
    }

    return true;
}

} // namespace

int runIBMPipelineVerification(
    const IBMPipelineVerificationSettings& settings)
{
    if (settings.steps <= 0)
    {
        throw std::invalid_argument(
            "IBM pipeline verification steps must be positive."
        );
    }
    if (settings.interval <= 0)
    {
        throw std::invalid_argument(
            "IBM pipeline verification interval must be positive."
        );
    }

    constexpr double dt = 1.0;

    LBMSolver cpuFluid(
        SedimentationBenchmark::nx,
        SedimentationBenchmark::ny,
        SedimentationBenchmark::latticeViscosity,
        0.0,
        true
    );
    cpuFluid.initialize();

    IBMParticle cpuParticle(
        SedimentationBenchmark::initialXLattice + 0.37,
        SedimentationBenchmark::initialYLattice - 0.29,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );
    cpuParticle.vx = 4.0e-4;
    cpuParticle.vy = -7.0e-4;
    cpuParticle.theta = 0.17;
    cpuParticle.omega = 2.5e-4;
    cpuParticle.updateMarkers();

    cuda_backend::LBMSolverCUDA gpuFluid(
        cpuFluid.nx(),
        cpuFluid.ny(),
        cpuFluid.viscosity(),
        cpuFluid.lidVelocity(),
        cpuFluid.periodicY()
    );
    gpuFluid.initializeFromHost(cpuFluid);

    cuda_backend::IBMParticleCUDA gpuParticle(cpuParticle);
    gpuFluid.synchronize();

    std::cout
        << std::scientific
        << std::setprecision(6)
        << "Persistent CUDA IBM pipeline verification\n"
        << "domain: " << cpuFluid.nx() << " x " << cpuFluid.ny() << '\n'
        << "markers: " << cpuParticle.markerCount << '\n'
        << "steps: " << settings.steps << '\n'
        << "comparison interval: " << settings.interval << '\n'
        << "maximum absolute tolerance: "
        << settings.maximumAbsoluteTolerance << '\n'
        << "relative L2 tolerance: "
        << settings.relativeL2Tolerance << "\n\n";

    for (int step = 1; step <= settings.steps; ++step)
    {
        advanceCpuPipeline(cpuFluid, cpuParticle, dt);
        advanceGpuPipeline(gpuFluid, gpuParticle, dt);

        if (step % settings.interval != 0 && step != settings.steps)
        {
            continue;
        }

        gpuFluid.synchronize();
        const PipelineSnapshot snapshot = collectSnapshot(
            cpuFluid,
            cpuParticle,
            gpuFluid,
            gpuParticle
        );

        std::cout
            << "step " << std::setw(6) << step
            << " | max |delta f|=" << snapshot.populations.maxAbsolute
            << " |delta Ffluid|="
            << (snapshot.forceX.maxAbsolute > snapshot.forceY.maxAbsolute
                ? snapshot.forceX.maxAbsolute
                : snapshot.forceY.maxAbsolute)
            << " |delta particle|="
            << snapshot.particleState.maxAbsolute
            << " (" << particleStateName(snapshot.particleState.maxIndex)
            << ")"
            << " |delta marker|="
            << (snapshot.markerX.maxAbsolute > snapshot.markerY.maxAbsolute
                ? snapshot.markerX.maxAbsolute
                : snapshot.markerY.maxAbsolute)
            << '\n';

        if (!checkSnapshot(snapshot, settings, step))
        {
            return 1;
        }
    }

    std::cout
        << "\nPersistent CUDA IBM pipeline verification: PASSED\n"
        << "The particle and all marker arrays remained device-resident "
        << "between operations and timesteps.\n";

    return 0;
}
