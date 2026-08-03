#include "GPULBMRunner.cuh"

#include "CUDAUtils.cuh"
#include "LBMSolverCUDA.cuh"
#include "LBMSolver.hpp"
#include "SedimentationBenchmark.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

void applyDeterministicForces(LBMSolver& solver)
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

class CudaEvent {
public:
    CudaEvent()
    {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent()
    {
        cudaEventDestroy(event_);
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

} // namespace

namespace cuda_backend {

int runPersistentGpuLbm(int steps)
{
    if (steps <= 0) {
        throw std::invalid_argument("GPU LBM steps must be positive.");
    }

    LBMSolver hostSeed(
        SedimentationBenchmark::nx,
        SedimentationBenchmark::ny,
        SedimentationBenchmark::latticeViscosity,
        0.0,
        true
    );
    hostSeed.initialize();
    applyDeterministicForces(hostSeed);
    hostSeed.computeMacroscopic();

    LBMSolverCUDA gpuSolver(
        hostSeed.nx(),
        hostSeed.ny(),
        hostSeed.viscosity(),
        hostSeed.lidVelocity(),
        hostSeed.periodicY()
    );
    gpuSolver.initializeFromHost(hostSeed);
    gpuSolver.synchronize();

    CudaEvent start;
    CudaEvent stop;

    CUDA_CHECK(cudaEventRecord(start.get()));

    for (int step = 0; step < steps; ++step) {
        gpuSolver.step();
    }

    // Reconstruct macroscopic fields from the final streamed populations.
    gpuSolver.computeMacroscopic();

    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(
        &elapsedMilliseconds,
        start.get(),
        stop.get()
    ));

    const LBMHostState finalState = gpuSolver.downloadState();

    double densitySum = 0.0;
    double maximumSpeed = 0.0;

    for (std::size_t node = 0; node < finalState.density.size(); ++node) {
        densitySum += finalState.density[node];
        maximumSpeed = std::max(
            maximumSpeed,
            std::hypot(
                finalState.velocityX[node],
                finalState.velocityY[node]
            )
        );
    }

    const double elapsedSeconds =
        static_cast<double>(elapsedMilliseconds) * 1.0e-3;
    const double latticeUpdates =
        static_cast<double>(gpuSolver.nodes()) * steps;
    const double mlups =
        latticeUpdates / elapsedSeconds / 1.0e6;
    const double meanDensity =
        densitySum / static_cast<double>(finalState.density.size());

    std::cout
        << std::scientific
        << std::setprecision(6)
        << "Persistent CUDA LBM run\n"
        << "domain: " << gpuSolver.nx() << " x " << gpuSolver.ny() << '\n'
        << "steps: " << steps << '\n'
        << "device-resident timesteps: yes\n"
        << "elapsed GPU time: " << elapsedSeconds << " s\n"
        << "throughput: " << mlups << " MLUPS\n"
        << "mean density: " << meanDensity << '\n'
        << "maximum speed: " << maximumSpeed << "\n\n";

    return 0;
}

} // namespace cuda_backend
