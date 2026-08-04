#include "IBMHydrodynamicLoadsVerification.hpp"

#include "CUDAUtils.cuh"
#include "IBMCoupling.hpp"
#include "IBMHydrodynamicLoads.cuh"
#include "IBMParticle.hpp"
#include "SedimentationBenchmark.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct HydrodynamicLoadCase {
    std::string name;
    double particleX = 0.0;
    double particleY = 0.0;
    double markerSpacing = 1.0;
    std::vector<double> markerX;
    std::vector<double> markerY;
    std::vector<double> markerForceX;
    std::vector<double> markerForceY;
};

struct ScalarErrorMetrics {
    double absolute = 0.0;
    double relative = 0.0;
};

ScalarErrorMetrics calculateScalarError(
    double reference,
    double candidate)
{
    ScalarErrorMetrics metrics;
    metrics.absolute = std::abs(candidate - reference);
    metrics.relative =
        std::abs(reference) > 0.0
        ? metrics.absolute / std::abs(reference)
        : metrics.absolute;
    return metrics;
}

bool exceedsTolerance(
    const ScalarErrorMetrics& metrics,
    const IBMHydrodynamicLoadsVerificationSettings& settings)
{
    return metrics.absolute > settings.maximumAbsoluteTolerance
        && metrics.relative > settings.relativeTolerance;
}

void reportFailure(
    const std::string& caseName,
    const char* quantity,
    double reference,
    double candidate,
    const ScalarErrorMetrics& metrics)
{
    std::cerr
        << "Verification failure in case '" << caseName << "', "
        << quantity << '\n'
        << "  CPU value: " << reference << '\n'
        << "  GPU value: " << candidate << '\n'
        << "  absolute difference: " << metrics.absolute << '\n'
        << "  relative difference: " << metrics.relative << '\n';
}

void validateCaseArrays(const HydrodynamicLoadCase& testCase)
{
    const std::size_t markerCount = testCase.markerX.size();
    if (testCase.markerY.size() != markerCount
        || testCase.markerForceX.size() != markerCount
        || testCase.markerForceY.size() != markerCount)
    {
        throw std::invalid_argument(
            "IBM hydrodynamic-load verification case has inconsistent arrays."
        );
    }
}

IBMParticle createParticle(const HydrodynamicLoadCase& testCase)
{
    validateCaseArrays(testCase);

    IBMParticle particle(
        testCase.particleX,
        testCase.particleY,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );

    particle.markerCount = static_cast<int>(testCase.markerX.size());
    particle.ds = testCase.markerSpacing;
    particle.markerX = testCase.markerX;
    particle.markerY = testCase.markerY;
    particle.markerForceX = testCase.markerForceX;
    particle.markerForceY = testCase.markerForceY;

    return particle;
}

HydrodynamicLoadCase createCircularCase(
    const std::string& name,
    double centreX,
    double centreY)
{
    IBMParticle particle(
        centreX,
        centreY,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );

    HydrodynamicLoadCase testCase;
    testCase.name = name;
    testCase.particleX = centreX;
    testCase.particleY = centreY;
    testCase.markerSpacing = particle.ds;
    testCase.markerX = particle.markerX;
    testCase.markerY = particle.markerY;
    testCase.markerForceX.resize(particle.markerX.size());
    testCase.markerForceY.resize(particle.markerY.size());
    return testCase;
}

HydrodynamicLoadCase createSingleMarkerCase()
{
    HydrodynamicLoadCase testCase;
    testCase.name = "single off-centre marker";
    testCase.particleX = 100.25;
    testCase.particleY = 300.75;
    testCase.markerSpacing = 0.83;
    testCase.markerX = {102.10};
    testCase.markerY = {297.80};
    testCase.markerForceX = {3.4e-4};
    testCase.markerForceY = {-7.1e-4};
    return testCase;
}

HydrodynamicLoadCase createUniformCircularCase()
{
    HydrodynamicLoadCase testCase = createCircularCase(
        "circular particle with uniform marker forcing",
        SedimentationBenchmark::initialXLattice,
        SedimentationBenchmark::initialYLattice
    );

    for (std::size_t marker = 0;
         marker < testCase.markerX.size();
         ++marker)
    {
        testCase.markerForceX[marker] = 1.25e-4;
        testCase.markerForceY[marker] = -2.75e-4;
    }

    return testCase;
}

HydrodynamicLoadCase createTorqueDominatedCase()
{
    HydrodynamicLoadCase testCase = createCircularCase(
        "circular particle with tangential marker forcing",
        SedimentationBenchmark::initialXLattice + 0.37,
        SedimentationBenchmark::initialYLattice - 0.61
    );

    constexpr double scale = 2.2e-5;
    for (std::size_t marker = 0;
         marker < testCase.markerX.size();
         ++marker)
    {
        const double relativeX =
            testCase.markerX[marker] - testCase.particleX;
        const double relativeY =
            testCase.markerY[marker] - testCase.particleY;

        testCase.markerForceX[marker] = scale * relativeY;
        testCase.markerForceY[marker] = -scale * relativeX;
    }

    return testCase;
}

HydrodynamicLoadCase createVaryingCircularCase()
{
    HydrodynamicLoadCase testCase = createCircularCase(
        "fractionally shifted circle with varying marker forces",
        SedimentationBenchmark::initialXLattice - 0.43,
        SedimentationBenchmark::initialYLattice + 0.29
    );

    for (std::size_t marker = 0;
         marker < testCase.markerX.size();
         ++marker)
    {
        const double markerValue = static_cast<double>(marker);
        testCase.markerForceX[marker] =
            2.5e-4
            + 1.3e-5 * static_cast<double>(
                (static_cast<int>(7 * marker + 3) % 19) - 9
            )
            + 1.0e-7 * markerValue;

        testCase.markerForceY[marker] =
            -1.7e-4
            + 1.1e-5 * static_cast<double>(
                (static_cast<int>(5 * marker + 11) % 17) - 8
            )
            - 2.0e-7 * markerValue;
    }

    return testCase;
}

HydrodynamicLoadCase createHighContentionCase()
{
    constexpr std::size_t markerCount = 1024;

    HydrodynamicLoadCase testCase;
    testCase.name = "1024 coincident markers with atomic contention";
    testCase.particleX = 98.40;
    testCase.particleY = 301.20;
    testCase.markerSpacing = 0.025;
    testCase.markerX.assign(markerCount, testCase.particleX + 4.75);
    testCase.markerY.assign(markerCount, testCase.particleY - 3.25);
    testCase.markerForceX.resize(markerCount);
    testCase.markerForceY.resize(markerCount);

    for (std::size_t marker = 0; marker < markerCount; ++marker)
    {
        const double sign = marker % 2 == 0 ? 1.0 : -1.0;
        testCase.markerForceX[marker] =
            sign * 4.0e-4 + 3.0e-8 * static_cast<double>(marker);
        testCase.markerForceY[marker] =
            -sign * 2.5e-4 - 2.0e-8 * static_cast<double>(marker);
    }

    return testCase;
}

HydrodynamicLoadCase createDeterministicRandomCase()
{
    constexpr std::size_t markerCount = 4096;

    HydrodynamicLoadCase testCase;
    testCase.name = "4096 deterministic pseudo-random markers";
    testCase.particleX = 97.35;
    testCase.particleY = 302.65;
    testCase.markerSpacing = 0.019;
    testCase.markerX.reserve(markerCount);
    testCase.markerY.reserve(markerCount);
    testCase.markerForceX.reserve(markerCount);
    testCase.markerForceY.reserve(markerCount);

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
        testCase.markerX.push_back(
            testCase.particleX + 30.0 * (nextUnit() - 0.5)
        );
        testCase.markerY.push_back(
            testCase.particleY + 30.0 * (nextUnit() - 0.5)
        );
        testCase.markerForceX.push_back(
            1.0e-3 * (nextUnit() - 0.5)
        );
        testCase.markerForceY.push_back(
            1.0e-3 * (nextUnit() - 0.5)
        );
    }

    return testCase;
}

std::vector<HydrodynamicLoadCase> createVerificationCases()
{
    std::vector<HydrodynamicLoadCase> cases;
    cases.reserve(6);
    cases.push_back(createSingleMarkerCase());
    cases.push_back(createUniformCircularCase());
    cases.push_back(createTorqueDominatedCase());
    cases.push_back(createVaryingCircularCase());
    cases.push_back(createHighContentionCase());
    cases.push_back(createDeterministicRandomCase());
    return cases;
}

bool runVerificationCase(
    const HydrodynamicLoadCase& testCase,
    const IBMHydrodynamicLoadsVerificationSettings& settings,
    std::size_t caseIndex,
    std::size_t caseCount)
{
    IBMParticle particle = createParticle(testCase);
    ibm::accumulateHydrodynamicLoads(particle);

    const double cpuForceX = particle.forceX;
    const double cpuForceY = particle.forceY;
    const double cpuTorque = particle.torque;
    const std::size_t markerCount = testCase.markerX.size();

    cuda_backend::DeviceArray<double> deviceMarkerX(markerCount);
    cuda_backend::DeviceArray<double> deviceMarkerY(markerCount);
    cuda_backend::DeviceArray<double> deviceMarkerForceX(markerCount);
    cuda_backend::DeviceArray<double> deviceMarkerForceY(markerCount);
    cuda_backend::DeviceArray<double> deviceParticleForceX(1);
    cuda_backend::DeviceArray<double> deviceParticleForceY(1);
    cuda_backend::DeviceArray<double> deviceParticleTorque(1);

    deviceMarkerX.copyFromHost(testCase.markerX);
    deviceMarkerY.copyFromHost(testCase.markerY);
    deviceMarkerForceX.copyFromHost(testCase.markerForceX);
    deviceMarkerForceY.copyFromHost(testCase.markerForceY);

    // Non-zero initial values verify that the launcher mirrors clearLoads()
    // before accumulating the hydrodynamic contributions.
    deviceParticleForceX.copyFromHost(std::vector<double>{3.25});
    deviceParticleForceY.copyFromHost(std::vector<double>{-4.50});
    deviceParticleTorque.copyFromHost(std::vector<double>{8.75});

    cuda_backend::launchIBMHydrodynamicLoadAccumulation(
        deviceMarkerX.data(),
        deviceMarkerY.data(),
        deviceMarkerForceX.data(),
        deviceMarkerForceY.data(),
        static_cast<int>(markerCount),
        testCase.markerSpacing,
        testCase.particleX,
        testCase.particleY,
        deviceParticleForceX.data(),
        deviceParticleForceY.data(),
        deviceParticleTorque.data()
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpuForceXValues;
    std::vector<double> gpuForceYValues;
    std::vector<double> gpuTorqueValues;
    deviceParticleForceX.copyToHost(gpuForceXValues);
    deviceParticleForceY.copyToHost(gpuForceYValues);
    deviceParticleTorque.copyToHost(gpuTorqueValues);

    const double gpuForceX = gpuForceXValues.front();
    const double gpuForceY = gpuForceYValues.front();
    const double gpuTorque = gpuTorqueValues.front();

    const ScalarErrorMetrics forceXMetrics =
        calculateScalarError(cpuForceX, gpuForceX);
    const ScalarErrorMetrics forceYMetrics =
        calculateScalarError(cpuForceY, gpuForceY);
    const ScalarErrorMetrics torqueMetrics =
        calculateScalarError(cpuTorque, gpuTorque);

    const bool forceXPassed = !exceedsTolerance(forceXMetrics, settings);
    const bool forceYPassed = !exceedsTolerance(forceYMetrics, settings);
    const bool torquePassed = !exceedsTolerance(torqueMetrics, settings);
    const bool passed = forceXPassed && forceYPassed && torquePassed;

    std::cout
        << '[' << caseIndex + 1 << '/' << caseCount << "] "
        << testCase.name << '\n'
        << "  markers: " << markerCount << '\n'
        << "  marker spacing: " << testCase.markerSpacing << '\n'
        << "  CPU hydrodynamic load: ("
        << cpuForceX << ", " << cpuForceY << ", " << cpuTorque << ")\n"
        << "  max |delta particle Fx|=" << forceXMetrics.absolute
        << "  relative=" << forceXMetrics.relative << '\n'
        << "  max |delta particle Fy|=" << forceYMetrics.absolute
        << "  relative=" << forceYMetrics.relative << '\n'
        << "  max |delta particle torque|=" << torqueMetrics.absolute
        << "  relative=" << torqueMetrics.relative << '\n'
        << "  result: " << (passed ? "PASSED" : "FAILED") << "\n\n";

    if (!forceXPassed)
    {
        reportFailure(
            testCase.name,
            "particle forceX",
            cpuForceX,
            gpuForceX,
            forceXMetrics
        );
    }
    if (!forceYPassed)
    {
        reportFailure(
            testCase.name,
            "particle forceY",
            cpuForceY,
            gpuForceY,
            forceYMetrics
        );
    }
    if (!torquePassed)
    {
        reportFailure(
            testCase.name,
            "particle torque",
            cpuTorque,
            gpuTorque,
            torqueMetrics
        );
    }

    return passed;
}

} // namespace

int runIBMHydrodynamicLoadsVerification(
    const IBMHydrodynamicLoadsVerificationSettings& settings)
{
    const std::vector<HydrodynamicLoadCase> cases =
        createVerificationCases();

    std::cout
        << std::scientific
        << std::setprecision(6)
        << "CUDA IBM hydrodynamic-load accumulation verification\n"
        << "cases: " << cases.size() << '\n'
        << "maximum absolute tolerance: "
        << settings.maximumAbsoluteTolerance << '\n'
        << "relative tolerance: "
        << settings.relativeTolerance << "\n\n";

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
        << "atomic additions requested per marker: 3\n"
        << "\nCUDA IBM hydrodynamic-load accumulation verification: "
        << (allPassed ? "PASSED" : "FAILED") << '\n';

    return allPassed ? 0 : 1;
}
