#include "IBMForcingVerification.hpp"

#include "CUDAUtils.cuh"
#include "IBMCoupling.hpp"
#include "IBMForcing.cuh"
#include "IBMParticle.hpp"
#include "SedimentationBenchmark.hpp"
#include "VerificationMetrics.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct ForcingCase {
    std::string name;
    double particleX = 0.0;
    double particleY = 0.0;
    double particleVelocityX = 0.0;
    double particleVelocityY = 0.0;
    double particleAngularVelocity = 0.0;
    double dt = 1.0;
    std::vector<double> markerX;
    std::vector<double> markerY;
    std::vector<double> fluidUx;
    std::vector<double> fluidUy;
};

bool exceedsTolerance(
    const ArrayErrorMetrics& metrics,
    const IBMForcingVerificationSettings& settings)
{
    return metrics.maxAbsolute > settings.maximumAbsoluteTolerance
        && metrics.relativeL2 > settings.relativeL2Tolerance;
}

void reportFailure(
    const std::string& caseName,
    const char* component,
    const ArrayErrorMetrics& metrics)
{
    std::cerr
        << "Verification failure in case '" << caseName << "', "
        << component << " at marker " << metrics.maxIndex << '\n'
        << "  CPU value: " << metrics.referenceValue << '\n'
        << "  GPU value: " << metrics.candidateValue << '\n'
        << "  absolute difference: " << metrics.maxAbsolute << '\n'
        << "  relative L2: " << metrics.relativeL2 << '\n';
}

IBMParticle createParticle(const ForcingCase& testCase)
{
    if (testCase.markerX.size() != testCase.markerY.size()
        || testCase.markerX.size() != testCase.fluidUx.size()
        || testCase.markerX.size() != testCase.fluidUy.size())
    {
        throw std::invalid_argument(
            "IBM forcing verification case has inconsistent marker arrays."
        );
    }

    IBMParticle particle(
        testCase.particleX,
        testCase.particleY,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );

    particle.vx = testCase.particleVelocityX;
    particle.vy = testCase.particleVelocityY;
    particle.omega = testCase.particleAngularVelocity;
    particle.markerCount = static_cast<int>(testCase.markerX.size());
    particle.markerX = testCase.markerX;
    particle.markerY = testCase.markerY;
    particle.fluidUx = testCase.fluidUx;
    particle.fluidUy = testCase.fluidUy;
    particle.markerForceX.assign(testCase.markerX.size(), 0.0);
    particle.markerForceY.assign(testCase.markerX.size(), 0.0);

    return particle;
}

ForcingCase createCircularCase(
    const std::string& name,
    double centreX,
    double centreY,
    double velocityX,
    double velocityY,
    double angularVelocity,
    double dt,
    double uniformFluidUx,
    double uniformFluidUy,
    bool spatiallyVaryingFluid)
{
    IBMParticle particle(
        centreX,
        centreY,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );

    ForcingCase testCase;
    testCase.name = name;
    testCase.particleX = centreX;
    testCase.particleY = centreY;
    testCase.particleVelocityX = velocityX;
    testCase.particleVelocityY = velocityY;
    testCase.particleAngularVelocity = angularVelocity;
    testCase.dt = dt;
    testCase.markerX = particle.markerX;
    testCase.markerY = particle.markerY;
    testCase.fluidUx.resize(testCase.markerX.size());
    testCase.fluidUy.resize(testCase.markerY.size());

    for (std::size_t marker = 0;
         marker < testCase.markerX.size();
         ++marker)
    {
        if (spatiallyVaryingFluid)
        {
            const double markerValue = static_cast<double>(marker);
            testCase.fluidUx[marker] =
                uniformFluidUx
                + 1.7e-5 * static_cast<double>(
                    (static_cast<int>(3 * marker + 5) % 17) - 8
                )
                + 2.0e-7 * markerValue;

            testCase.fluidUy[marker] =
                uniformFluidUy
                + 1.3e-5 * static_cast<double>(
                    (static_cast<int>(5 * marker + 7) % 19) - 9
                )
                - 1.0e-7 * markerValue;
        }
        else
        {
            testCase.fluidUx[marker] = uniformFluidUx;
            testCase.fluidUy[marker] = uniformFluidUy;
        }
    }

    return testCase;
}

ForcingCase createDeterministicRandomCase()
{
    constexpr std::size_t markerCount = 1000;
    constexpr double centreX = 97.35;
    constexpr double centreY = 302.65;

    ForcingCase testCase;
    testCase.name = "1000 deterministic pseudo-random markers";
    testCase.particleX = centreX;
    testCase.particleY = centreY;
    testCase.particleVelocityX = -2.7e-3;
    testCase.particleVelocityY = 1.9e-3;
    testCase.particleAngularVelocity = -3.1e-3;
    testCase.dt = 0.375;
    testCase.markerX.reserve(markerCount);
    testCase.markerY.reserve(markerCount);
    testCase.fluidUx.reserve(markerCount);
    testCase.fluidUy.reserve(markerCount);

    std::uint64_t state = 0x4d595df4d0f33173ULL;
    auto nextUnit = [&state]() {
        state = state * 6364136223846793005ULL
              + 1442695040888963407ULL;
        const std::uint64_t mantissa = state >> 11;
        return static_cast<double>(mantissa)
             * (1.0 / 9007199254740992.0);
    };

    for (std::size_t marker = 0; marker < markerCount; ++marker)
    {
        const double relativeX = 30.0 * (nextUnit() - 0.5);
        const double relativeY = 30.0 * (nextUnit() - 0.5);

        testCase.markerX.push_back(centreX + relativeX);
        testCase.markerY.push_back(centreY + relativeY);
        testCase.fluidUx.push_back(8.0e-3 * (nextUnit() - 0.5));
        testCase.fluidUy.push_back(8.0e-3 * (nextUnit() - 0.5));
    }

    return testCase;
}

std::vector<ForcingCase> createVerificationCases()
{
    const double x = SedimentationBenchmark::initialXLattice;
    const double y = SedimentationBenchmark::initialYLattice;

    std::vector<ForcingCase> cases;
    cases.reserve(6);
    cases.push_back(createCircularCase(
        "stationary particle in quiescent fluid",
        x, y,
        0.0, 0.0, 0.0,
        1.0,
        0.0, 0.0,
        false
    ));
    cases.push_back(createCircularCase(
        "translating particle in quiescent fluid",
        x, y,
        3.2e-3, -1.7e-3, 0.0,
        0.5,
        0.0, 0.0,
        false
    ));
    cases.push_back(createCircularCase(
        "rotating particle in quiescent fluid",
        x + 0.37, y + 0.61,
        0.0, 0.0, 2.3e-3,
        0.25,
        0.0, 0.0,
        false
    ));
    cases.push_back(createCircularCase(
        "combined rigid motion in a uniform fluid",
        x - 0.41, y + 0.29,
        -2.1e-3, 1.4e-3, -1.8e-3,
        1.25,
        4.0e-4, -6.0e-4,
        false
    ));
    cases.push_back(createCircularCase(
        "combined rigid motion in a varying marker field",
        x + 0.73, y - 0.46,
        1.6e-3, -2.8e-3, 2.5e-3,
        0.625,
        -3.0e-4, 2.0e-4,
        true
    ));
    cases.push_back(createDeterministicRandomCase());

    return cases;
}

bool runVerificationCase(
    const ForcingCase& testCase,
    const IBMForcingVerificationSettings& settings,
    std::size_t caseIndex,
    std::size_t caseCount)
{
    IBMParticle particle = createParticle(testCase);
    ibm::computeDirectForcing(particle, testCase.dt);

    const std::vector<double> cpuForceX = particle.markerForceX;
    const std::vector<double> cpuForceY = particle.markerForceY;
    const std::size_t markerCount = testCase.markerX.size();

    cuda_backend::DeviceArray<double> deviceMarkerX(markerCount);
    cuda_backend::DeviceArray<double> deviceMarkerY(markerCount);
    cuda_backend::DeviceArray<double> deviceFluidUx(markerCount);
    cuda_backend::DeviceArray<double> deviceFluidUy(markerCount);
    cuda_backend::DeviceArray<double> deviceForceX(markerCount);
    cuda_backend::DeviceArray<double> deviceForceY(markerCount);

    deviceMarkerX.copyFromHost(testCase.markerX);
    deviceMarkerY.copyFromHost(testCase.markerY);
    deviceFluidUx.copyFromHost(testCase.fluidUx);
    deviceFluidUy.copyFromHost(testCase.fluidUy);

    cuda_backend::launchIBMDirectForcing(
        deviceMarkerX.data(),
        deviceMarkerY.data(),
        deviceFluidUx.data(),
        deviceFluidUy.data(),
        deviceForceX.data(),
        deviceForceY.data(),
        static_cast<int>(markerCount),
        testCase.particleX,
        testCase.particleY,
        testCase.particleVelocityX,
        testCase.particleVelocityY,
        testCase.particleAngularVelocity,
        testCase.dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpuForceX;
    std::vector<double> gpuForceY;
    deviceForceX.copyToHost(gpuForceX);
    deviceForceY.copyToHost(gpuForceY);

    const ArrayErrorMetrics forceXMetrics =
        calculateArrayErrorMetrics(cpuForceX, gpuForceX);
    const ArrayErrorMetrics forceYMetrics =
        calculateArrayErrorMetrics(cpuForceY, gpuForceY);

    const bool forceXPassed = !exceedsTolerance(forceXMetrics, settings);
    const bool forceYPassed = !exceedsTolerance(forceYMetrics, settings);
    const bool passed = forceXPassed && forceYPassed;

    std::cout
        << '[' << caseIndex + 1 << '/' << caseCount << "] "
        << testCase.name << '\n'
        << "  markers: " << markerCount << '\n'
        << "  dt: " << testCase.dt << '\n'
        << "  max |delta marker Fx|=" << forceXMetrics.maxAbsolute
        << "  relative L2=" << forceXMetrics.relativeL2 << '\n'
        << "  max |delta marker Fy|=" << forceYMetrics.maxAbsolute
        << "  relative L2=" << forceYMetrics.relativeL2 << '\n'
        << "  result: " << (passed ? "PASSED" : "FAILED") << "\n\n";

    if (!forceXPassed)
    {
        reportFailure(testCase.name, "markerForceX", forceXMetrics);
    }
    if (!forceYPassed)
    {
        reportFailure(testCase.name, "markerForceY", forceYMetrics);
    }

    return passed;
}

} // namespace

int runIBMForcingVerification(
    const IBMForcingVerificationSettings& settings)
{
    const std::vector<ForcingCase> cases = createVerificationCases();

    std::cout
        << std::scientific
        << std::setprecision(6)
        << "CUDA IBM direct-forcing multi-case verification\n"
        << "cases: " << cases.size() << '\n'
        << "maximum absolute tolerance: "
        << settings.maximumAbsoluteTolerance << '\n'
        << "relative L2 tolerance: "
        << settings.relativeL2Tolerance << "\n\n";

    bool allPassed = true;
    std::size_t totalMarkers = 0;

    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        totalMarkers += cases[index].markerX.size();
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
        << "kernel launches: " << cases.size() << '\n'
        << "total marker evaluations: " << totalMarkers << '\n'
        << "\nCUDA IBM direct-forcing multi-case verification: "
        << (allPassed ? "PASSED" : "FAILED") << '\n';

    return allPassed ? 0 : 1;
}
