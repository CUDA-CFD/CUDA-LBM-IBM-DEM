#include "IBMInterpolationVerification.hpp"

#include "CUDAUtils.cuh"
#include "IBMInterpolation.cuh"
#include "IBMCoupling.hpp"
#include "IBMParticle.hpp"
#include "LBMSolver.hpp"
#include "SedimentationBenchmark.hpp"
#include "VerificationMetrics.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

enum class FluidState {
    uniform,
    spatiallyVarying
};

struct MarkerSet {
    std::vector<double> x;
    std::vector<double> y;
};

struct VerificationCase {
    std::string name;
    FluidState fluidState;
    MarkerSet markers;
};

void createUniformFluidState(LBMSolver& fluid)
{
    constexpr double forceX = 1.6e-4;
    constexpr double forceY = -1.2e-4;

    for (int node = 0; node < fluid.nodes(); ++node) {
        fluid.forceX(node) = forceX;
        fluid.forceY(node) = forceY;
    }

    fluid.computeMacroscopic();
}

void createSpatiallyVaryingFluidState(LBMSolver& fluid)
{
    // Produce a non-zero, spatially varying velocity field using the same
    // macroscopic reconstruction routine used by the solver.
    for (int node = 0; node < fluid.nodes(); ++node) {
        const int x = node % fluid.nx();
        const int y = node / fluid.nx();

        fluid.forceX(node) =
            2.0e-5 * static_cast<double>(((3 * x + 2 * y) % 17) - 8);
        fluid.forceY(node) =
            2.0e-5 * static_cast<double>(((5 * x + 7 * y) % 19) - 9);
    }

    fluid.computeMacroscopic();
}

void createFluidState(LBMSolver& fluid, FluidState state)
{
    if (state == FluidState::uniform) {
        createUniformFluidState(fluid);
    }
    else {
        createSpatiallyVaryingFluidState(fluid);
    }
}

const char* fluidStateName(FluidState state)
{
    return state == FluidState::uniform
        ? "uniform"
        : "spatially varying";
}

MarkerSet createCircularMarkers(double centreX, double centreY)
{
    const IBMParticle particle(
        centreX,
        centreY,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );

    return {particle.markerX, particle.markerY};
}

MarkerSet createLeftBottomBoundaryMarkers()
{
    return {
        {
            -1.25, -0.25, 0.05, 0.75,
             1.50,  2.00, 8.25, 24.50
        },
        {
            18.25, 42.50, 0.05, 0.75,
             1.50,  2.00, -0.25, -1.25
        }
    };
}

MarkerSet createRightTopBoundaryMarkers(int nx, int ny)
{
    return {
        {
            nx + 1.25, nx + 0.25, nx - 0.05, nx - 0.75,
            nx - 1.50, nx - 2.00, 48.25, 72.50
        },
        {
            24.25, 58.50, ny - 0.05, ny - 0.75,
            ny - 1.50, ny - 2.00, ny + 0.25, ny + 1.25
        }
    };
}

MarkerSet createInteriorGridMarkers(int nx, int ny)
{
    constexpr int countX = 16;
    constexpr int countY = 16;
    constexpr double margin = 2.25;

    MarkerSet markers;
    markers.x.reserve(countX * countY);
    markers.y.reserve(countX * countY);

    for (int j = 0; j < countY; ++j) {
        const double y = margin +
            (static_cast<double>(j) + 0.37) *
            (static_cast<double>(ny) - 2.0 * margin) /
            static_cast<double>(countY);

        for (int i = 0; i < countX; ++i) {
            const double x = margin +
                (static_cast<double>(i) + 0.61) *
                (static_cast<double>(nx) - 2.0 * margin) /
                static_cast<double>(countX);

            markers.x.push_back(x);
            markers.y.push_back(y);
        }
    }

    return markers;
}

MarkerSet createDeterministicRandomMarkers(
    int nx,
    int ny,
    int markerCount)
{
    MarkerSet markers;
    markers.x.reserve(static_cast<std::size_t>(markerCount));
    markers.y.reserve(static_cast<std::size_t>(markerCount));

    std::uint32_t state = 0x00C0FFEEu;
    auto nextUnitValue = [&state]() {
        state = 1664525u * state + 1013904223u;
        return static_cast<double>(state) /
            static_cast<double>(std::numeric_limits<std::uint32_t>::max());
    };

    // Include the two-cell interpolation support beyond each boundary so that
    // clipping of partially outside stencils is exercised as well.
    constexpr double supportMargin = 1.75;
    const double xSpan = static_cast<double>(nx) + 2.0 * supportMargin;
    const double ySpan = static_cast<double>(ny) + 2.0 * supportMargin;

    for (int marker = 0; marker < markerCount; ++marker) {
        markers.x.push_back(-supportMargin + nextUnitValue() * xSpan);
        markers.y.push_back(-supportMargin + nextUnitValue() * ySpan);
    }

    return markers;
}

void configureParticleMarkers(
    IBMParticle& particle,
    const MarkerSet& markers)
{
    if (markers.x.empty() || markers.x.size() != markers.y.size()) {
        throw std::invalid_argument(
            "IBM interpolation verification requires matching, non-empty "
            "marker-coordinate arrays."
        );
    }

    particle.markerCount = static_cast<int>(markers.x.size());
    particle.markerX = markers.x;
    particle.markerY = markers.y;
    particle.fluidUx.assign(markers.x.size(), 0.0);
    particle.fluidUy.assign(markers.x.size(), 0.0);
    particle.markerForceX.assign(markers.x.size(), 0.0);
    particle.markerForceY.assign(markers.x.size(), 0.0);
}

bool exceedsTolerance(
    const ArrayErrorMetrics& metrics,
    const IBMInterpolationVerificationSettings& settings)
{
    return metrics.maxAbsolute > settings.maximumAbsoluteTolerance
        && metrics.relativeL2 > settings.relativeL2Tolerance;
}

void reportFailure(
    const std::string& caseName,
    const char* quantity,
    const ArrayErrorMetrics& metrics)
{
    std::cerr
        << "  failure in case: " << caseName << '\n'
        << "  quantity: " << quantity << '\n'
        << "  marker: " << metrics.maxIndex << '\n'
        << "  maximum absolute error: " << metrics.maxAbsolute << '\n'
        << "  relative L2 error: " << metrics.relativeL2 << '\n'
        << "  CPU value: " << metrics.referenceValue << '\n'
        << "  GPU value: " << metrics.candidateValue << '\n';
}

bool runVerificationCase(
    LBMSolver& fluid,
    const VerificationCase& testCase,
    const IBMInterpolationVerificationSettings& settings,
    std::size_t caseIndex,
    std::size_t caseCount)
{
    createFluidState(fluid, testCase.fluidState);

    IBMParticle particle(
        SedimentationBenchmark::initialXLattice,
        SedimentationBenchmark::initialYLattice,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );
    configureParticleMarkers(particle, testCase.markers);

    // Independent CPU reference result.
    ibm::interpolate(fluid, particle);
    const std::vector<double> cpuMarkerUx = particle.fluidUx;
    const std::vector<double> cpuMarkerUy = particle.fluidUy;

    const std::size_t markerCount =
        static_cast<std::size_t>(particle.markerCount);

    cuda_backend::DeviceArray<double> deviceMarkerX(markerCount);
    cuda_backend::DeviceArray<double> deviceMarkerY(markerCount);
    cuda_backend::DeviceArray<double> deviceMarkerUx(markerCount);
    cuda_backend::DeviceArray<double> deviceMarkerUy(markerCount);
    cuda_backend::DeviceArray<double> deviceVelocityX(
        static_cast<std::size_t>(fluid.nodes())
    );
    cuda_backend::DeviceArray<double> deviceVelocityY(
        static_cast<std::size_t>(fluid.nodes())
    );

    deviceMarkerX.copyFromHost(particle.markerX);
    deviceMarkerY.copyFromHost(particle.markerY);
    deviceVelocityX.copyFromHost(fluid.ux());
    deviceVelocityY.copyFromHost(fluid.uy());

    cuda_backend::launchIBMInterpolation(
        deviceMarkerX.data(),
        deviceMarkerY.data(),
        deviceMarkerUx.data(),
        deviceMarkerUy.data(),
        deviceVelocityX.data(),
        deviceVelocityY.data(),
        particle.markerCount,
        fluid.nx(),
        fluid.ny()
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpuMarkerUx;
    std::vector<double> gpuMarkerUy;
    deviceMarkerUx.copyToHost(gpuMarkerUx);
    deviceMarkerUy.copyToHost(gpuMarkerUy);

    const ArrayErrorMetrics uxMetrics =
        calculateArrayErrorMetrics(cpuMarkerUx, gpuMarkerUx);
    const ArrayErrorMetrics uyMetrics =
        calculateArrayErrorMetrics(cpuMarkerUy, gpuMarkerUy);

    const bool uxPassed = !exceedsTolerance(uxMetrics, settings);
    const bool uyPassed = !exceedsTolerance(uyMetrics, settings);
    const bool passed = uxPassed && uyPassed;

    std::cout
        << '[' << caseIndex + 1 << '/' << caseCount << "] "
        << testCase.name << '\n'
        << "  field: " << fluidStateName(testCase.fluidState) << '\n'
        << "  markers: " << particle.markerCount << '\n'
        << "  max |delta marker Ux|=" << uxMetrics.maxAbsolute
        << "  relative L2=" << uxMetrics.relativeL2 << '\n'
        << "  max |delta marker Uy|=" << uyMetrics.maxAbsolute
        << "  relative L2=" << uyMetrics.relativeL2 << '\n'
        << "  result: " << (passed ? "PASSED" : "FAILED") << "\n\n";

    if (!uxPassed) {
        reportFailure(testCase.name, "markerVelocityX", uxMetrics);
    }
    if (!uyPassed) {
        reportFailure(testCase.name, "markerVelocityY", uyMetrics);
    }

    return passed;
}

std::vector<VerificationCase> createVerificationCases(int nx, int ny)
{
    constexpr int randomMarkerCount = 1000;

    MarkerSet interiorGrid = createInteriorGridMarkers(nx, ny);

    std::vector<VerificationCase> cases;
    cases.reserve(7);
    cases.push_back({
        "centred circular particle",
        FluidState::spatiallyVarying,
        createCircularMarkers(
            SedimentationBenchmark::initialXLattice,
            SedimentationBenchmark::initialYLattice
        )
    });
    cases.push_back({
        "fractionally shifted circular particle",
        FluidState::spatiallyVarying,
        createCircularMarkers(
            SedimentationBenchmark::initialXLattice + 0.37,
            SedimentationBenchmark::initialYLattice + 0.61
        )
    });
    cases.push_back({
        "left and bottom boundary markers",
        FluidState::spatiallyVarying,
        createLeftBottomBoundaryMarkers()
    });
    cases.push_back({
        "right and top boundary markers",
        FluidState::spatiallyVarying,
        createRightTopBoundaryMarkers(nx, ny)
    });
    cases.push_back({
        "interior grid in a uniform field",
        FluidState::uniform,
        interiorGrid
    });
    cases.push_back({
        "interior grid in a spatially varying field",
        FluidState::spatiallyVarying,
        std::move(interiorGrid)
    });
    cases.push_back({
        "deterministic pseudo-random markers",
        FluidState::spatiallyVarying,
        createDeterministicRandomMarkers(nx, ny, randomMarkerCount)
    });

    return cases;
}

} // namespace

int runIBMInterpolationVerification(
    const IBMInterpolationVerificationSettings& settings)
{
    LBMSolver fluid(
        SedimentationBenchmark::nx,
        SedimentationBenchmark::ny,
        SedimentationBenchmark::latticeViscosity,
        0.0,
        true
    );
    fluid.initialize();

    const std::vector<VerificationCase> cases =
        createVerificationCases(fluid.nx(), fluid.ny());

    std::cout
        << std::scientific
        << std::setprecision(6)
        << "CUDA IBM interpolation multi-case verification\n"
        << "domain: " << fluid.nx() << " x " << fluid.ny() << '\n'
        << "cases: " << cases.size() << '\n'
        << "maximum absolute tolerance: "
        << settings.maximumAbsoluteTolerance << '\n'
        << "relative L2 tolerance: "
        << settings.relativeL2Tolerance << "\n\n";

    bool allPassed = true;
    std::size_t totalMarkers = 0;

    for (std::size_t index = 0; index < cases.size(); ++index) {
        totalMarkers += cases[index].markers.x.size();
        if (!runVerificationCase(
                fluid,
                cases[index],
                settings,
                index,
                cases.size())) {
            allPassed = false;
        }
    }

    std::cout
        << "kernel launches: " << cases.size() << '\n'
        << "total marker evaluations: " << totalMarkers << '\n'
        << "\nCUDA IBM interpolation multi-case verification: "
        << (allPassed ? "PASSED" : "FAILED") << '\n';

    return allPassed ? 0 : 1;
}
