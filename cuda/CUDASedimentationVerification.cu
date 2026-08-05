#include "CUDASedimentationVerification.hpp"

#include "IBMCoupling.hpp"
#include "IBMParticle.hpp"
#include "IBMParticleCUDA.cuh"
#include "LBMSolver.hpp"
#include "LBMSolverCUDA.cuh"
#include "SedimentationBenchmark.hpp"
#include "VerificationMetrics.hpp"
#include "WallContact.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct HostSedimentationLoads
{
    double hydrodynamicForceX = 0.0;
    double hydrodynamicForceY = 0.0;
    double gravityForceY = 0.0;
    double buoyancyForceY = 0.0;
    double externalForceY = 0.0;
    double wallForceX = 0.0;
    double wallForceY = 0.0;
    double maximumWallOverlap = 0.0;
    double totalForceX = 0.0;
    double totalForceY = 0.0;
    int wallContactActive = 0;
};

struct SedimentationSnapshot
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
    ArrayErrorMetrics sedimentationLoads;
};

bool exceedsTolerance(
    const ArrayErrorMetrics& metrics,
    const CUDASedimentationVerificationSettings& settings)
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

std::vector<double> loadValues(const HostSedimentationLoads& loads)
{
    return {
        loads.hydrodynamicForceX,
        loads.hydrodynamicForceY,
        loads.gravityForceY,
        loads.buoyancyForceY,
        loads.externalForceY,
        loads.wallForceX,
        loads.wallForceY,
        loads.maximumWallOverlap,
        loads.totalForceX,
        loads.totalForceY,
        static_cast<double>(loads.wallContactActive)
    };
}

std::vector<double> loadValues(
    const cuda_backend::IBMSedimentationLoadsCUDA& loads)
{
    return {
        loads.hydrodynamicForceX,
        loads.hydrodynamicForceY,
        loads.gravityForceY,
        loads.buoyancyForceY,
        loads.externalForceY,
        loads.wallForceX,
        loads.wallForceY,
        loads.maximumWallOverlap,
        loads.totalForceX,
        loads.totalForceY,
        static_cast<double>(loads.wallContactActive)
    };
}

HostSedimentationLoads applyCpuExternalLoads(
    IBMParticle& particle,
    double gravityForceY,
    double buoyancyForceY)
{
    HostSedimentationLoads loads;
    loads.hydrodynamicForceX = particle.forceX;
    loads.hydrodynamicForceY = particle.forceY;
    loads.gravityForceY = gravityForceY;
    loads.buoyancyForceY = buoyancyForceY;
    loads.externalForceY = gravityForceY + buoyancyForceY;

    particle.addExternalForce(0.0, loads.externalForceY);

    const WallContactLoads wallContact = computeWallContact(
        particle,
        static_cast<double>(SedimentationBenchmark::nx),
        static_cast<double>(SedimentationBenchmark::ny),
        SedimentationBenchmark::wallNormalStiffness,
        SedimentationBenchmark::wallNormalDamping
    );
    particle.addExternalForce(wallContact.forceX, wallContact.forceY);

    loads.wallForceX = wallContact.forceX;
    loads.wallForceY = wallContact.forceY;
    loads.maximumWallOverlap = wallContact.maximumOverlap;
    loads.wallContactActive = wallContact.active ? 1 : 0;
    loads.totalForceX = particle.forceX;
    loads.totalForceY = particle.forceY;
    return loads;
}

HostSedimentationLoads advanceCpuSedimentation(
    LBMSolver& fluid,
    IBMParticle& particle,
    double gravityForceY,
    double buoyancyForceY,
    double dt)
{
    fluid.clearForces();
    fluid.computeMacroscopic();

    ibm::interpolate(fluid, particle);
    ibm::computeDirectForcing(particle, dt);
    ibm::spreadToFluid(fluid, particle);
    ibm::accumulateHydrodynamicLoads(particle);

    HostSedimentationLoads loads = applyCpuExternalLoads(
        particle,
        gravityForceY,
        buoyancyForceY
    );

    fluid.computeMacroscopic();
    fluid.collideAndForce();
    fluid.stream();

    particle.integrate(dt);
    return loads;
}

void advanceGpuSedimentation(
    cuda_backend::LBMSolverCUDA& fluid,
    cuda_backend::IBMParticleCUDA& particle,
    double gravityForceY,
    double buoyancyForceY,
    double dt)
{
    fluid.clearForces();
    fluid.computeMacroscopic();

    particle.interpolate(fluid);
    particle.computeDirectForcing(dt);
    particle.spreadToFluid(fluid);
    particle.accumulateHydrodynamicLoads();
    particle.applySedimentationLoads(
        gravityForceY,
        buoyancyForceY,
        static_cast<double>(SedimentationBenchmark::nx),
        static_cast<double>(SedimentationBenchmark::ny),
        SedimentationBenchmark::wallNormalStiffness,
        SedimentationBenchmark::wallNormalDamping
    );

    fluid.computeMacroscopic();
    fluid.collideAndForce();
    fluid.stream();

    particle.integrate(dt);
}

SedimentationSnapshot collectSnapshot(
    const LBMSolver& cpuFluid,
    const IBMParticle& cpuParticle,
    const HostSedimentationLoads& cpuLoads,
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

    const cuda_backend::IBMSedimentationLoadsCUDA gpuLoads =
        gpuParticle.downloadSedimentationLoads();

    SedimentationSnapshot snapshot;
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
    snapshot.sedimentationLoads = calculateArrayErrorMetrics(
        loadValues(cpuLoads), loadValues(gpuLoads)
    );
    return snapshot;
}

bool checkSnapshot(
    const SedimentationSnapshot& snapshot,
    const CUDASedimentationVerificationSettings& settings,
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
        {"marker force Y", &snapshot.markerForceY},
        {"sedimentation loads", &snapshot.sedimentationLoads}
    };

    for (const Quantity& quantity : quantities)
    {
        if (!exceedsTolerance(*quantity.metrics, settings))
        {
            continue;
        }

        std::cerr
            << "CUDA sedimentation verification failure\n"
            << "  step: " << step << '\n'
            << "  quantity: " << quantity.name << '\n'
            << "  index: " << quantity.metrics->maxIndex << '\n'
            << "  CPU value: " << quantity.metrics->referenceValue << '\n'
            << "  GPU value: " << quantity.metrics->candidateValue << '\n'
            << "  maximum absolute error: "
            << quantity.metrics->maxAbsolute << '\n'
            << "  relative L2 error: "
            << quantity.metrics->relativeL2 << '\n';
        return false;
    }

    return true;
}

bool verifyExternalLoadCases(
    const CUDASedimentationVerificationSettings& settings,
    double gravityForceY,
    double buoyancyForceY)
{
    struct ExternalLoadCase
    {
        const char* name;
        double x;
        double y;
        double vx;
        double vy;
        double hydrodynamicForceX;
        double hydrodynamicForceY;
    };

    const double radius = SedimentationBenchmark::particleRadiusLattice;
    const double width = static_cast<double>(SedimentationBenchmark::nx);
    const double height = static_cast<double>(SedimentationBenchmark::ny);

    const ExternalLoadCase cases[] = {
        {"interior particle", 100.0, 300.0, 0.01, -0.02, 0.003, -0.004},
        {"bottom wall approach", 100.0, radius - 0.25, 0.0, -0.004, 0.001, -0.002},
        {"bottom wall separation clamp", 100.0, radius - 0.25, 0.0, 1.0, -0.002, 0.003},
        {"top wall approach", 100.0, height - radius + 0.20, 0.0, 0.003, 0.004, -0.001},
        {"left wall approach", radius - 0.15, 300.0, -0.002, 0.0, -0.003, 0.002},
        {"right wall approach", width - radius + 0.15, 300.0, 0.002, 0.0, 0.002, -0.003},
        {"bottom-left corner", radius - 0.10, radius - 0.20, -0.002, -0.003, 0.001, -0.002}
    };

    std::cout << "External-load and wall-contact cases: "
              << std::size(cases) << '\n';

    for (std::size_t index = 0; index < std::size(cases); ++index)
    {
        const ExternalLoadCase& test = cases[index];

        IBMParticle cpuParticle(
            test.x,
            test.y,
            radius,
            SedimentationBenchmark::latticeParticleDensity
        );
        cpuParticle.vx = test.vx;
        cpuParticle.vy = test.vy;
        cpuParticle.forceX = test.hydrodynamicForceX;
        cpuParticle.forceY = test.hydrodynamicForceY;

        IBMParticle gpuHostParticle = cpuParticle;
        cuda_backend::IBMParticleCUDA gpuParticle(gpuHostParticle);

        const HostSedimentationLoads cpuLoads = applyCpuExternalLoads(
            cpuParticle,
            gravityForceY,
            buoyancyForceY
        );

        gpuParticle.applySedimentationLoads(
            gravityForceY,
            buoyancyForceY,
            width,
            height,
            SedimentationBenchmark::wallNormalStiffness,
            SedimentationBenchmark::wallNormalDamping
        );
        gpuParticle.synchronize();
        gpuParticle.downloadRigidStateToHost(gpuHostParticle);
        const cuda_backend::IBMSedimentationLoadsCUDA gpuLoads =
            gpuParticle.downloadSedimentationLoads();

        const ArrayErrorMetrics forceMetrics = calculateArrayErrorMetrics(
            {cpuParticle.forceX, cpuParticle.forceY},
            {gpuHostParticle.forceX, gpuHostParticle.forceY}
        );
        const ArrayErrorMetrics loadMetrics = calculateArrayErrorMetrics(
            loadValues(cpuLoads), loadValues(gpuLoads)
        );

        const bool passed =
            !exceedsTolerance(forceMetrics, settings)
            && !exceedsTolerance(loadMetrics, settings);

        std::cout
            << "  [" << index + 1 << '/' << std::size(cases) << "] "
            << test.name
            << " | max |delta load|="
            << std::max(forceMetrics.maxAbsolute, loadMetrics.maxAbsolute)
            << " | contact=" << cpuLoads.wallContactActive
            << " | result: " << (passed ? "PASSED" : "FAILED")
            << '\n';

        if (!passed)
        {
            return false;
        }
    }

    return true;
}

} // namespace

int runCUDASedimentationVerification(
    const CUDASedimentationVerificationSettings& settings)
{
    if (settings.steps <= 0)
    {
        throw std::invalid_argument(
            "CUDA sedimentation verification steps must be positive."
        );
    }
    if (settings.interval <= 0)
    {
        throw std::invalid_argument(
            "CUDA sedimentation verification interval must be positive."
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
        SedimentationBenchmark::initialXLattice,
        SedimentationBenchmark::initialYLattice,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );

    const double particleArea = cpuParticle.area();
    const double gravityForceY =
        -cpuParticle.density * particleArea
        * SedimentationBenchmark::latticeGravity;
    const double buoyancyForceY =
        SedimentationBenchmark::latticeFluidDensity * particleArea
        * SedimentationBenchmark::latticeGravity;

    std::cout
        << std::scientific
        << std::setprecision(6)
        << "Complete CUDA sedimentation verification\n"
        << "domain: " << cpuFluid.nx() << " x " << cpuFluid.ny() << '\n'
        << "markers: " << cpuParticle.markerCount << '\n'
        << "steps: " << settings.steps << '\n'
        << "comparison interval: " << settings.interval << '\n'
        << "maximum absolute tolerance: "
        << settings.maximumAbsoluteTolerance << '\n'
        << "relative L2 tolerance: "
        << settings.relativeL2Tolerance << "\n\n";

    if (!verifyExternalLoadCases(
            settings,
            gravityForceY,
            buoyancyForceY))
    {
        std::cerr << "External-load verification: FAILED\n";
        return 1;
    }

    std::cout << "\nRepeated full sedimentation pipeline:\n";

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

    HostSedimentationLoads cpuLoads;

    for (int step = 1; step <= settings.steps; ++step)
    {
        cpuLoads = advanceCpuSedimentation(
            cpuFluid,
            cpuParticle,
            gravityForceY,
            buoyancyForceY,
            dt
        );
        advanceGpuSedimentation(
            gpuFluid,
            gpuParticle,
            gravityForceY,
            buoyancyForceY,
            dt
        );

        if (step % settings.interval != 0 && step != settings.steps)
        {
            continue;
        }

        gpuFluid.synchronize();
        const SedimentationSnapshot snapshot = collectSnapshot(
            cpuFluid,
            cpuParticle,
            cpuLoads,
            gpuFluid,
            gpuParticle
        );

        const double maximumFluidForceError = std::max(
            snapshot.forceX.maxAbsolute,
            snapshot.forceY.maxAbsolute
        );
        const double maximumMarkerError = std::max(
            snapshot.markerX.maxAbsolute,
            snapshot.markerY.maxAbsolute
        );

        std::cout
            << "step " << std::setw(6) << step
            << " | max |delta f|=" << snapshot.populations.maxAbsolute
            << " |delta Ffluid|=" << maximumFluidForceError
            << " |delta particle|="
            << snapshot.particleState.maxAbsolute
            << " |delta marker|=" << maximumMarkerError
            << " |delta loads|="
            << snapshot.sedimentationLoads.maxAbsolute
            << '\n';

        if (!checkSnapshot(snapshot, settings, step))
        {
            return 1;
        }
    }

    std::cout
        << "\nComplete CUDA sedimentation verification: PASSED\n"
        << "Gravity, buoyancy, normal wall contact, LBM, IBM, particle "
        << "integration, and marker regeneration agree with the CPU path.\n";

    return 0;
}
