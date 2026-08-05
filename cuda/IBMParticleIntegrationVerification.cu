#include "IBMParticleIntegrationVerification.hpp"

#include "CUDAUtils.cuh"
#include "IBMParticle.hpp"
#include "IBMParticleIntegration.cuh"
#include "SedimentationBenchmark.hpp"
#include "VerificationMetrics.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{

struct ParticleIntegrationCase {
    std::string name;
    double x = 0.0;
    double y = 0.0;
    double radius = SedimentationBenchmark::particleRadiusLattice;
    double density = SedimentationBenchmark::latticeParticleDensity;
    double vx = 0.0;
    double vy = 0.0;
    double theta = 0.0;
    double omega = 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double torque = 0.0;
    double dt = 1.0;
    int steps = 1;
};

bool exceedsTolerance(
    const ArrayErrorMetrics& metrics,
    const IBMParticleIntegrationVerificationSettings& settings)
{
    return metrics.maxAbsolute > settings.maximumAbsoluteTolerance
        && metrics.relativeL2 > settings.relativeL2Tolerance;
}

cuda_backend::IBMParticleStateCUDA makeCudaState(
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

std::vector<double> stateValues(const IBMParticle& particle)
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
        particle.alpha
    };
}

std::vector<double> stateValues(
    const cuda_backend::IBMParticleStateCUDA& particle)
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
        particle.alpha
    };
}

const char* stateName(std::size_t index)
{
    static constexpr const char* names[] = {
        "x", "y", "vx", "vy", "ax", "ay",
        "theta", "omega", "alpha"
    };

    return index < 9 ? names[index] : "unknown";
}

void reportArrayFailure(
    const std::string& caseName,
    const char* quantity,
    const ArrayErrorMetrics& metrics)
{
    std::cerr
        << "Verification failure in case '" << caseName << "', "
        << quantity << " at index " << metrics.maxIndex << '\n'
        << "  CPU value: " << metrics.referenceValue << '\n'
        << "  GPU value: " << metrics.candidateValue << '\n'
        << "  absolute difference: " << metrics.maxAbsolute << '\n'
        << "  relative L2: " << metrics.relativeL2 << '\n';
}

std::vector<ParticleIntegrationCase> createVerificationCases()
{
    const double x = SedimentationBenchmark::initialXLattice;
    const double y = SedimentationBenchmark::initialYLattice;

    return {
        {
            "stationary particle with zero loads",
            x, y,
            SedimentationBenchmark::particleRadiusLattice,
            SedimentationBenchmark::latticeParticleDensity,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 1
        },
        {
            "single translational update",
            x + 0.37, y - 0.42,
            SedimentationBenchmark::particleRadiusLattice,
            SedimentationBenchmark::latticeParticleDensity,
            1.4e-3, -2.1e-3, 0.13, 0.0,
            2.5e-2, -3.7e-2, 0.0,
            0.40, 1
        },
        {
            "single angular update",
            x - 0.58, y + 0.26,
            SedimentationBenchmark::particleRadiusLattice,
            SedimentationBenchmark::latticeParticleDensity,
            0.0, 0.0, -0.27, 1.8e-3,
            0.0, 0.0, 1.25,
            0.25, 1
        },
        {
            "combined translation and rotation over 25 steps",
            x + 1.15, y - 0.93,
            9.75, 1.60,
            -2.2e-3, 1.7e-3, 0.41, -1.4e-3,
            -1.7e-2, 2.3e-2, -0.85,
            0.125, 25
        },
        {
            "large particle using multiple marker blocks",
            x, y,
            50.0, 1.10,
            8.0e-4, -6.0e-4, -0.19, 9.0e-4,
            8.0e-2, -5.0e-2, 3.0,
            0.20, 10
        },
        {
            "benchmark-sized particle over 10000 repeated steps",
            x, y,
            SedimentationBenchmark::particleRadiusLattice,
            SedimentationBenchmark::latticeParticleDensity,
            0.0, 0.0, 0.0, 0.0,
            3.0e-5, -1.0e-4, 2.0e-4,
            1.0, 10000
        }
    };
}

bool runVerificationCase(
    const ParticleIntegrationCase& testCase,
    const IBMParticleIntegrationVerificationSettings& settings,
    std::size_t caseIndex,
    std::size_t caseCount)
{
    IBMParticle cpuParticle(
        testCase.x,
        testCase.y,
        testCase.radius,
        testCase.density
    );

    cpuParticle.vx = testCase.vx;
    cpuParticle.vy = testCase.vy;
    cpuParticle.theta = testCase.theta;
    cpuParticle.omega = testCase.omega;
    cpuParticle.forceX = testCase.forceX;
    cpuParticle.forceY = testCase.forceY;
    cpuParticle.torque = testCase.torque;
    cpuParticle.updateMarkers();

    const cuda_backend::IBMParticleStateCUDA initialGpuState =
        makeCudaState(cpuParticle);

    const int markerCount = cpuParticle.markerCount;

    cuda_backend::DeviceArray<cuda_backend::IBMParticleStateCUDA>
        deviceParticleState(1);
    cuda_backend::DeviceArray<double> deviceForceX(1);
    cuda_backend::DeviceArray<double> deviceForceY(1);
    cuda_backend::DeviceArray<double> deviceTorque(1);
    cuda_backend::DeviceArray<double> deviceMarkerX(markerCount);
    cuda_backend::DeviceArray<double> deviceMarkerY(markerCount);

    deviceParticleState.copyFromHost({initialGpuState});
    deviceForceX.copyFromHost({testCase.forceX});
    deviceForceY.copyFromHost({testCase.forceY});
    deviceTorque.copyFromHost({testCase.torque});
    deviceMarkerX.copyFromHost(cpuParticle.markerX);
    deviceMarkerY.copyFromHost(cpuParticle.markerY);

    for (int step = 0; step < testCase.steps; ++step)
    {
        cpuParticle.integrate(testCase.dt);

        cuda_backend::launchIBMParticleIntegration(
            deviceParticleState.data(),
            deviceForceX.data(),
            deviceForceY.data(),
            deviceTorque.data(),
            deviceMarkerX.data(),
            deviceMarkerY.data(),
            markerCount,
            testCase.dt
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<cuda_backend::IBMParticleStateCUDA> gpuStateValues;
    std::vector<double> gpuMarkerX;
    std::vector<double> gpuMarkerY;
    deviceParticleState.copyToHost(gpuStateValues);
    deviceMarkerX.copyToHost(gpuMarkerX);
    deviceMarkerY.copyToHost(gpuMarkerY);

    const std::vector<double> cpuState = stateValues(cpuParticle);
    const std::vector<double> gpuState = stateValues(gpuStateValues.front());
    const ArrayErrorMetrics stateMetrics = calculateArrayErrorMetrics(
        cpuState,
        gpuState
    );
    const ArrayErrorMetrics markerXMetrics = calculateArrayErrorMetrics(
        cpuParticle.markerX,
        gpuMarkerX
    );
    const ArrayErrorMetrics markerYMetrics = calculateArrayErrorMetrics(
        cpuParticle.markerY,
        gpuMarkerY
    );

    bool statePassed = true;
    for (std::size_t stateIndex = 0;
         stateIndex < cpuState.size();
         ++stateIndex)
    {
        const double absoluteError =
            std::abs(gpuState[stateIndex] - cpuState[stateIndex]);
        const double relativeError =
            std::abs(cpuState[stateIndex]) > 0.0
            ? absoluteError / std::abs(cpuState[stateIndex])
            : absoluteError;

        if (absoluteError > settings.maximumAbsoluteTolerance
            && relativeError > settings.relativeL2Tolerance)
        {
            statePassed = false;
            std::cerr
                << "Verification failure in case '" << testCase.name
                << "', particle state " << stateName(stateIndex) << '\n'
                << "  CPU value: " << cpuState[stateIndex] << '\n'
                << "  GPU value: " << gpuState[stateIndex] << '\n'
                << "  absolute difference: " << absoluteError << '\n'
                << "  relative difference: " << relativeError << '\n';
        }
    }

    const bool markerXPassed = !exceedsTolerance(markerXMetrics, settings);
    const bool markerYPassed = !exceedsTolerance(markerYMetrics, settings);
    const bool passed = statePassed && markerXPassed && markerYPassed;

    std::cout
        << '[' << caseIndex + 1 << '/' << caseCount << "] "
        << testCase.name << '\n'
        << "  integration steps: " << testCase.steps << '\n'
        << "  markers: " << markerCount << '\n'
        << "  dt: " << testCase.dt << '\n'
        << "  max |delta particle state|=" << stateMetrics.maxAbsolute
        << "  relative L2=" << stateMetrics.relativeL2
        << "  component=" << stateName(stateMetrics.maxIndex) << '\n'
        << "  max |delta marker X|=" << markerXMetrics.maxAbsolute
        << "  relative L2=" << markerXMetrics.relativeL2 << '\n'
        << "  max |delta marker Y|=" << markerYMetrics.maxAbsolute
        << "  relative L2=" << markerYMetrics.relativeL2 << '\n'
        << "  result: " << (passed ? "PASSED" : "FAILED") << "\n\n";

    if (!markerXPassed)
    {
        reportArrayFailure(testCase.name, "marker X", markerXMetrics);
    }
    if (!markerYPassed)
    {
        reportArrayFailure(testCase.name, "marker Y", markerYMetrics);
    }

    return passed;
}

} // namespace

int runIBMParticleIntegrationVerification(
    const IBMParticleIntegrationVerificationSettings& settings)
{
    const std::vector<ParticleIntegrationCase> cases =
        createVerificationCases();

    std::cout
        << std::scientific
        << std::setprecision(6)
        << "CUDA IBM particle-integration verification\n"
        << "cases: " << cases.size() << '\n'
        << "maximum absolute tolerance: "
        << settings.maximumAbsoluteTolerance << '\n'
        << "relative L2 tolerance: "
        << settings.relativeL2Tolerance << "\n\n";

    bool allPassed = true;
    std::size_t totalIntegrationSteps = 0;
    std::size_t totalMarkerEvaluations = 0;

    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        IBMParticle markerCounter(
            cases[index].x,
            cases[index].y,
            cases[index].radius,
            cases[index].density
        );

        totalIntegrationSteps +=
            static_cast<std::size_t>(cases[index].steps);
        totalMarkerEvaluations +=
            static_cast<std::size_t>(cases[index].steps)
            * static_cast<std::size_t>(markerCounter.markerCount);

        if (!runVerificationCase(
                cases[index],
                settings,
                index,
                cases.size()))
        {
            allPassed = false;
        }
    }

    std::cout
        << "integration launcher calls: " << totalIntegrationSteps << '\n'
        << "CUDA kernel launches: " << 2 * totalIntegrationSteps << '\n'
        << "total marker-position evaluations: "
        << totalMarkerEvaluations << '\n'
        << "\nCUDA IBM particle-integration verification: "
        << (allPassed ? "PASSED" : "FAILED") << '\n';

    return allPassed ? 0 : 1;
}
