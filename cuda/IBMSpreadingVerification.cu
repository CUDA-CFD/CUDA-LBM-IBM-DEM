#include "IBMSpreadingVerification.hpp"

#include "CUDAUtils.cuh"
#include "IBMCoupling.hpp"
#include "IBMParticle.hpp"
#include "IBMSpreading.cuh"
#include "LBMSolver.hpp"
#include "SedimentationBenchmark.hpp"
#include "VerificationMetrics.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct SpreadingCase {
    std::string name;
    double markerSpacing = 1.0;
    bool useBackgroundForce = false;
    std::vector<double> markerX;
    std::vector<double> markerY;
    std::vector<double> markerForceX;
    std::vector<double> markerForceY;
};

bool exceedsTolerance(
    const ArrayErrorMetrics& metrics,
    const IBMSpreadingVerificationSettings& settings)
{
    return metrics.maxAbsolute > settings.maximumAbsoluteTolerance
        && metrics.relativeL2 > settings.relativeL2Tolerance;
}

void reportFailure(
    const std::string& caseName,
    const char* component,
    const ArrayErrorMetrics& metrics,
    int nx)
{
    const int node = static_cast<int>(metrics.maxIndex);
    const int x = node % nx;
    const int y = node / nx;

    std::cerr
        << "Verification failure in case '" << caseName << "', "
        << component << " at node " << node
        << " (x=" << x << ", y=" << y << ")\n"
        << "  CPU value: " << metrics.referenceValue << '\n'
        << "  GPU value: " << metrics.candidateValue << '\n'
        << "  absolute difference: " << metrics.maxAbsolute << '\n'
        << "  relative L2: " << metrics.relativeL2 << '\n';
}

void validateCase(const SpreadingCase& testCase)
{
    const std::size_t markerCount = testCase.markerX.size();
    if (testCase.markerY.size() != markerCount
        || testCase.markerForceX.size() != markerCount
        || testCase.markerForceY.size() != markerCount)
    {
        throw std::invalid_argument(
            "IBM spreading verification case has inconsistent marker arrays."
        );
    }

    if (markerCount == 0 || testCase.markerSpacing <= 0.0)
    {
        throw std::invalid_argument(
            "IBM spreading verification requires markers and positive spacing."
        );
    }
}

IBMParticle createParticle(const SpreadingCase& testCase)
{
    validateCase(testCase);

    IBMParticle particle(
        SedimentationBenchmark::initialXLattice,
        SedimentationBenchmark::initialYLattice,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );

    particle.markerCount = static_cast<int>(testCase.markerX.size());
    particle.ds = testCase.markerSpacing;
    particle.markerX = testCase.markerX;
    particle.markerY = testCase.markerY;
    particle.markerForceX = testCase.markerForceX;
    particle.markerForceY = testCase.markerForceY;
    particle.fluidUx.assign(testCase.markerX.size(), 0.0);
    particle.fluidUy.assign(testCase.markerX.size(), 0.0);

    return particle;
}

void assignDeterministicMarkerForces(SpreadingCase& testCase)
{
    const std::size_t markerCount = testCase.markerX.size();
    testCase.markerForceX.resize(markerCount);
    testCase.markerForceY.resize(markerCount);

    for (std::size_t marker = 0; marker < markerCount; ++marker)
    {
        const double index = static_cast<double>(marker);
        testCase.markerForceX[marker] =
            1.5e-3
            + 2.5e-5 * static_cast<double>(
                (static_cast<int>(7 * marker + 3) % 23) - 11
            )
            + 1.0e-7 * index;

        testCase.markerForceY[marker] =
            -1.1e-3
            + 2.0e-5 * static_cast<double>(
                (static_cast<int>(5 * marker + 9) % 19) - 9
            )
            - 8.0e-8 * index;
    }
}

SpreadingCase createCircularCase(
    const std::string& name,
    double centreX,
    double centreY,
    bool useBackgroundForce)
{
    IBMParticle particle(
        centreX,
        centreY,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );

    SpreadingCase testCase;
    testCase.name = name;
    testCase.markerSpacing = particle.ds;
    testCase.useBackgroundForce = useBackgroundForce;
    testCase.markerX = particle.markerX;
    testCase.markerY = particle.markerY;
    assignDeterministicMarkerForces(testCase);
    return testCase;
}

SpreadingCase createSingleMarkerCase()
{
    SpreadingCase testCase;
    testCase.name = "single interior marker";
    testCase.markerSpacing = 0.83;
    testCase.markerX = {91.37};
    testCase.markerY = {277.64};
    testCase.markerForceX = {2.4e-3};
    testCase.markerForceY = {-1.7e-3};
    return testCase;
}

SpreadingCase createBoundaryCase(int nx, int ny)
{
    SpreadingCase testCase;
    testCase.name = "markers near all four boundaries";
    testCase.markerSpacing = 0.71;
    testCase.markerX = {
        -1.35, -0.20, 0.35, 1.60,
        static_cast<double>(nx) - 1.65,
        static_cast<double>(nx) - 0.25,
        static_cast<double>(nx) + 0.40,
        47.25, 153.70, 0.45, static_cast<double>(nx) - 0.40
    };
    testCase.markerY = {
        75.30, 211.75, -0.35, static_cast<double>(ny) - 0.20,
        88.45, static_cast<double>(ny) - 1.55,
        403.10, -1.20, static_cast<double>(ny) + 0.35,
        static_cast<double>(ny) - 0.30, 0.55
    };
    assignDeterministicMarkerForces(testCase);
    return testCase;
}

SpreadingCase createHighContentionCase()
{
    constexpr std::size_t markerCount = 1024;

    SpreadingCase testCase;
    testCase.name = "1024 overlapping markers with atomic contention";
    testCase.markerSpacing = 0.025;
    testCase.markerX.assign(markerCount, 100.375);
    testCase.markerY.assign(markerCount, 300.625);
    testCase.markerForceX.resize(markerCount);
    testCase.markerForceY.resize(markerCount);

    for (std::size_t marker = 0; marker < markerCount; ++marker)
    {
        testCase.markerForceX[marker] =
            8.0e-4 + 1.0e-7 * static_cast<double>(marker % 31);
        testCase.markerForceY[marker] =
            -6.0e-4 - 8.0e-8 * static_cast<double>(marker % 29);
    }

    return testCase;
}

SpreadingCase createDeterministicRandomCase(int nx, int ny)
{
    constexpr std::size_t markerCount = 1000;

    SpreadingCase testCase;
    testCase.name = "1000 deterministic pseudo-random markers";
    testCase.markerSpacing = 0.19;
    testCase.markerX.reserve(markerCount);
    testCase.markerY.reserve(markerCount);
    testCase.markerForceX.reserve(markerCount);
    testCase.markerForceY.reserve(markerCount);

    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
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
            -0.75 + (static_cast<double>(nx) + 0.50) * nextUnit()
        );
        testCase.markerY.push_back(
            -0.75 + (static_cast<double>(ny) + 0.50) * nextUnit()
        );
        testCase.markerForceX.push_back(4.0e-3 * (nextUnit() - 0.5));
        testCase.markerForceY.push_back(4.0e-3 * (nextUnit() - 0.5));
    }

    return testCase;
}

std::vector<SpreadingCase> createVerificationCases(int nx, int ny)
{
    const double x = SedimentationBenchmark::initialXLattice;
    const double y = SedimentationBenchmark::initialYLattice;

    std::vector<SpreadingCase> cases;
    cases.reserve(7);
    cases.push_back(createSingleMarkerCase());
    cases.push_back(createCircularCase(
        "centred circular particle",
        x,
        y,
        false
    ));
    cases.push_back(createCircularCase(
        "fractionally shifted circular particle",
        x + 0.37,
        y + 0.61,
        false
    ));
    cases.push_back(createBoundaryCase(nx, ny));
    cases.push_back(createHighContentionCase());
    cases.push_back(createDeterministicRandomCase(nx, ny));
    cases.push_back(createCircularCase(
        "accumulation onto a preloaded fluid-force field",
        x - 0.43,
        y + 0.28,
        true
    ));
    return cases;
}

void initializeForceField(LBMSolver& fluid, bool useBackgroundForce)
{
    fluid.clearForces();

    if (!useBackgroundForce)
    {
        return;
    }

    for (int y = 0; y < fluid.ny(); ++y)
    {
        for (int x = 0; x < fluid.nx(); ++x)
        {
            const int node = fluid.nodeIndex(x, y);
            fluid.forceX(node) =
                2.0e-7 * static_cast<double>((3 * x + 5 * y) % 17 - 8);
            fluid.forceY(node) =
                1.5e-7 * static_cast<double>((7 * x + 2 * y) % 19 - 9);
        }
    }
}

std::size_t countChangedNodes(
    const std::vector<double>& initialX,
    const std::vector<double>& initialY,
    const std::vector<double>& finalX,
    const std::vector<double>& finalY)
{
    std::size_t count = 0;
    for (std::size_t node = 0; node < finalX.size(); ++node)
    {
        if (finalX[node] != initialX[node]
            || finalY[node] != initialY[node])
        {
            ++count;
        }
    }
    return count;
}

bool runVerificationCase(
    const SpreadingCase& testCase,
    const IBMSpreadingVerificationSettings& settings,
    std::size_t caseIndex,
    std::size_t caseCount)
{
    validateCase(testCase);

    LBMSolver fluid(
        SedimentationBenchmark::nx,
        SedimentationBenchmark::ny,
        SedimentationBenchmark::latticeViscosity,
        0.0,
        true
    );
    fluid.initialize();
    initializeForceField(fluid, testCase.useBackgroundForce);

    const std::vector<double> initialForceX = fluid.forceX();
    const std::vector<double> initialForceY = fluid.forceY();

    IBMParticle particle = createParticle(testCase);
    ibm::spreadToFluid(fluid, particle);

    const std::vector<double> cpuForceX = fluid.forceX();
    const std::vector<double> cpuForceY = fluid.forceY();
    const std::size_t markerCount = testCase.markerX.size();
    const std::size_t nodeCount = static_cast<std::size_t>(fluid.nodes());

    cuda_backend::DeviceArray<double> deviceMarkerX(markerCount);
    cuda_backend::DeviceArray<double> deviceMarkerY(markerCount);
    cuda_backend::DeviceArray<double> deviceMarkerForceX(markerCount);
    cuda_backend::DeviceArray<double> deviceMarkerForceY(markerCount);
    cuda_backend::DeviceArray<double> deviceForceX(nodeCount);
    cuda_backend::DeviceArray<double> deviceForceY(nodeCount);

    deviceMarkerX.copyFromHost(testCase.markerX);
    deviceMarkerY.copyFromHost(testCase.markerY);
    deviceMarkerForceX.copyFromHost(testCase.markerForceX);
    deviceMarkerForceY.copyFromHost(testCase.markerForceY);
    deviceForceX.copyFromHost(initialForceX);
    deviceForceY.copyFromHost(initialForceY);

    cuda_backend::launchIBMSpreading(
        deviceMarkerX.data(),
        deviceMarkerY.data(),
        deviceMarkerForceX.data(),
        deviceMarkerForceY.data(),
        deviceForceX.data(),
        deviceForceY.data(),
        static_cast<int>(markerCount),
        fluid.nx(),
        fluid.ny(),
        testCase.markerSpacing
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

    const std::size_t changedNodes = countChangedNodes(
        initialForceX,
        initialForceY,
        cpuForceX,
        cpuForceY
    );

    std::cout
        << '[' << caseIndex + 1 << '/' << caseCount << "] "
        << testCase.name << '\n'
        << "  markers: " << markerCount << '\n'
        << "  marker spacing: " << testCase.markerSpacing << '\n'
        << "  affected fluid nodes: " << changedNodes << '\n'
        << "  max |delta fluid Fx|=" << forceXMetrics.maxAbsolute
        << "  relative L2=" << forceXMetrics.relativeL2 << '\n'
        << "  max |delta fluid Fy|=" << forceYMetrics.maxAbsolute
        << "  relative L2=" << forceYMetrics.relativeL2 << '\n'
        << "  result: " << (passed ? "PASSED" : "FAILED") << "\n\n";

    if (!forceXPassed)
    {
        reportFailure(testCase.name, "fluid force X", forceXMetrics, fluid.nx());
    }
    if (!forceYPassed)
    {
        reportFailure(testCase.name, "fluid force Y", forceYMetrics, fluid.nx());
    }

    return passed;
}

} // namespace

int runIBMSpreadingVerification(
    const IBMSpreadingVerificationSettings& settings)
{
    const std::vector<SpreadingCase> cases = createVerificationCases(
        SedimentationBenchmark::nx,
        SedimentationBenchmark::ny
    );

    std::cout
        << std::scientific
        << std::setprecision(6)
        << "CUDA IBM force-spreading multi-case verification\n"
        << "domain: " << SedimentationBenchmark::nx
        << " x " << SedimentationBenchmark::ny << '\n'
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
        if (!runVerificationCase(cases[index], settings, index, cases.size()))
        {
            allPassed = false;
        }
    }

    std::cout
        << "kernel launches: " << cases.size() << '\n'
        << "total marker evaluations: " << totalMarkers << '\n'
        << "atomic additions requested per marker: up to 32\n"
        << "\nCUDA IBM force-spreading multi-case verification: "
        << (allPassed ? "PASSED" : "FAILED") << '\n';

    return allPassed ? 0 : 1;
}
