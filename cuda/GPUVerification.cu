#include "GPUVerification.hpp"

#include "CUDAUtils.cuh"
#include "LBMCollision.cuh"
#include "LBMMacroscopic.cuh"
#include "LBMStreaming.cuh"
#include "LBMSolverCUDA.cuh"
#include "LBMSolver.hpp"
#include "SedimentationBenchmark.hpp"
#include "VerificationMetrics.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int Q = 9;

void applyDeterministicVerificationForces(LBMSolver& solver)
{
    for (int node = 0; node < solver.nodes(); ++node) {
        const int x = node % solver.nx();
        const int y = node / solver.nx();

        solver.forceX(node) =
            1.0e-7 * static_cast<double>(((3 * x + 2 * y) % 11) - 5);
        solver.forceY(node) =
            1.0e-7 * static_cast<double>(((2 * x + 5 * y) % 13) - 6);
    }
}

bool exceedsTolerance(
    const ArrayErrorMetrics& metrics,
    const GPUVerificationSettings& settings)
{
    return metrics.maxAbsolute > settings.maximumAbsoluteTolerance
        && metrics.relativeL2 > settings.relativeL2Tolerance;
}

void reportFailure(
    const char* quantity,
    const ArrayErrorMetrics& metrics,
    int step,
    int nx,
    bool populations)
{
    std::cerr
        << "CPU-GPU verification FAILED\n"
        << "step: " << step << '\n'
        << "quantity: " << quantity << '\n'
        << "max absolute error: " << metrics.maxAbsolute << '\n'
        << "relative L2 error: " << metrics.relativeL2 << '\n';

    if (populations) {
        const std::size_t node = metrics.maxIndex / Q;
        const std::size_t direction = metrics.maxIndex % Q;
        std::cerr
            << "node: " << node
            << " (x=" << (node % static_cast<std::size_t>(nx))
            << ", y=" << (node / static_cast<std::size_t>(nx)) << ")\n"
            << "direction: " << direction << '\n';
    }
    else {
        const std::size_t node = metrics.maxIndex;
        std::cerr
            << "node: " << node
            << " (x=" << (node % static_cast<std::size_t>(nx))
            << ", y=" << (node / static_cast<std::size_t>(nx)) << ")\n";
    }

    std::cerr
        << "CPU value: " << metrics.referenceValue << '\n'
        << "GPU value: " << metrics.candidateValue << '\n';
}

VerificationSnapshot collectSnapshot(
    int step,
    const LBMSolver& cpuSolver,
    const cuda_backend::LBMSolverCUDA& gpuSolver)
{
    const cuda_backend::LBMHostState gpuState =
        gpuSolver.downloadState();

    VerificationSnapshot snapshot;
    snapshot.step = step;
    snapshot.density = calculateArrayErrorMetrics(
        cpuSolver.rho(), gpuState.density
    );
    snapshot.velocityX = calculateArrayErrorMetrics(
        cpuSolver.ux(), gpuState.velocityX
    );
    snapshot.velocityY = calculateArrayErrorMetrics(
        cpuSolver.uy(), gpuState.velocityY
    );
    snapshot.populations = calculateArrayErrorMetrics(
        cpuSolver.populations(), gpuState.populations
    );
    return snapshot;
}

} // namespace

int runFullDomainGPUVerification(const GPUVerificationSettings& settings)
{
    if (settings.steps <= 0) {
        throw std::invalid_argument("Verification steps must be positive.");
    }
    if (settings.interval <= 0) {
        throw std::invalid_argument("Verification interval must be positive.");
    }

    std::cout
        << std::scientific
        << std::setprecision(6)
        << "Full-domain CPU-GPU LBM verification\n"
        << "domain: " << SedimentationBenchmark::nx
        << " x " << SedimentationBenchmark::ny << '\n'
        << "steps: " << settings.steps << '\n'
        << "comparison interval: " << settings.interval << '\n'
        << "maximum absolute tolerance: "
        << settings.maximumAbsoluteTolerance << '\n'
        << "relative L2 tolerance: "
        << settings.relativeL2Tolerance << "\n\n";

    LBMSolver cpuSolver(
        SedimentationBenchmark::nx,
        SedimentationBenchmark::ny,
        SedimentationBenchmark::latticeViscosity,
        0.0,
        true
    );
    cpuSolver.initialize();
    applyDeterministicVerificationForces(cpuSolver);
    cpuSolver.computeMacroscopic();

    cuda_backend::LBMSolverCUDA gpuSolver(
        cpuSolver.nx(),
        cpuSolver.ny(),
        cpuSolver.viscosity(),
        cpuSolver.lidVelocity(),
        cpuSolver.periodicY()
    );
    gpuSolver.initializeFromHost(cpuSolver);
    gpuSolver.synchronize();

    writeVerificationCsvHeader(settings.csvFilename);

    auto initialSnapshot = collectSnapshot(
        0,
        cpuSolver,
        gpuSolver
    );
    appendVerificationCsv(settings.csvFilename, initialSnapshot);

    bool passed = true;

    for (int step = 1; step <= settings.steps; ++step) {
        cpuSolver.computeMacroscopic();
        cpuSolver.collideAndForce();
        cpuSolver.stream();

        gpuSolver.step();

        if (step % settings.interval != 0 && step != settings.steps) {
            continue;
        }

        gpuSolver.synchronize();

        const VerificationSnapshot snapshot = collectSnapshot(
            step,
            cpuSolver,
            gpuSolver
        );
        appendVerificationCsv(settings.csvFilename, snapshot);

        std::cout
            << "step " << std::setw(6) << step
            << " | max |delta rho|=" << snapshot.density.maxAbsolute
            << " |delta ux|=" << snapshot.velocityX.maxAbsolute
            << " |delta uy|=" << snapshot.velocityY.maxAbsolute
            << " |delta f|=" << snapshot.populations.maxAbsolute << '\n';

        if (exceedsTolerance(snapshot.density, settings)) {
            reportFailure(
                "density", snapshot.density, step, cpuSolver.nx(), false
            );
            passed = false;
            break;
        }
        if (exceedsTolerance(snapshot.velocityX, settings)) {
            reportFailure(
                "velocityX", snapshot.velocityX, step, cpuSolver.nx(), false
            );
            passed = false;
            break;
        }
        if (exceedsTolerance(snapshot.velocityY, settings)) {
            reportFailure(
                "velocityY", snapshot.velocityY, step, cpuSolver.nx(), false
            );
            passed = false;
            break;
        }
        if (exceedsTolerance(snapshot.populations, settings)) {
            reportFailure(
                "populations", snapshot.populations,
                step, cpuSolver.nx(), true
            );
            passed = false;
            break;
        }
    }

    if (!passed) {
        return 1;
    }

    std::cout
        << "\nFull-domain CPU-GPU verification: PASSED\n"
        << "Results: " << settings.csvFilename << "\n\n";
    return 0;
}
