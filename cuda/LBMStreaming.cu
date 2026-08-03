#include "LBMStreaming.cuh"

#include "CUDAUtils.cuh"
#include "LBMSolver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int Q = 9;

__constant__ int deviceCx[Q] = {
     0,  1,  0, -1,  0,  1, -1, -1,  1
};

__constant__ int deviceCy[Q] = {
     0,  0,  1,  0, -1,  1,  1, -1, -1
};

__constant__ int deviceOpposite[Q] = {
    0, 3, 4, 1, 2, 7, 8, 5, 6
};

__constant__ double deviceWeight[Q] = {
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0
};

__global__ void pullStreamingKernel(
    const double* sourcePopulations,
    double* destinationPopulations,
    const double* density,
    int nx,
    int ny,
    double lidVelocity,
    bool periodicY)
{
    const int node =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    const int numberOfNodes = nx * ny;

    if (node >= numberOfNodes) {
        return;
    }

    const int x = node % nx;
    const int y = node / nx;
    const int destinationOffset = node * Q;

    #pragma unroll
    for (int direction = 0; direction < Q; ++direction) {
        int sourceX = x - deviceCx[direction];
        int sourceY = y - deviceCy[direction];

        if (periodicY) {
            // The two vertical boundaries are periodic, while the side
            // walls use halfway bounce-back. In pull form, an incoming
            // population whose source lies outside a side wall is obtained
            // from the opposite, outgoing population at the same node.
            if (sourceX < 0 || sourceX >= nx) {
                const int oppositeDirection = deviceOpposite[direction];
                destinationPopulations[destinationOffset + direction] =
                    sourcePopulations[destinationOffset + oppositeDirection];
                continue;
            }

            if (sourceY < 0) {
                sourceY += ny;
            }
            else if (sourceY >= ny) {
                sourceY -= ny;
            }

            const int sourceNode = sourceY * nx + sourceX;
            destinationPopulations[destinationOffset + direction] =
                sourcePopulations[sourceNode * Q + direction];
            continue;
        }

        const bool sourceInside =
            sourceX >= 0 && sourceX < nx
            && sourceY >= 0 && sourceY < ny;

        if (sourceInside) {
            const int sourceNode = sourceY * nx + sourceX;
            destinationPopulations[destinationOffset + direction] =
                sourcePopulations[sourceNode * Q + direction];
            continue;
        }

        const int oppositeDirection = deviceOpposite[direction];
        double reflectedPopulation =
            sourcePopulations[destinationOffset + oppositeDirection];

        // For an incoming population at the top wall, the corresponding
        // outgoing direction is oppositeDirection. This is the pull-form
        // equivalent of the moving-wall correction in the CPU push scheme.
        if (sourceY >= ny && lidVelocity != 0.0) {
            reflectedPopulation -=
                6.0
                * deviceWeight[oppositeDirection]
                * density[node]
                * static_cast<double>(deviceCx[oppositeDirection])
                * lidVelocity;
        }

        destinationPopulations[destinationOffset + direction] =
            reflectedPopulation;
    }
}

double maximumAbsoluteDifference(
    const std::vector<double>& first,
    const std::vector<double>& second)
{
    if (first.size() != second.size()) {
        throw std::invalid_argument("Vector size mismatch during verification.");
    }

    double maximumError = 0.0;

    for (std::size_t index = 0; index < first.size(); ++index) {
        maximumError = std::max(
            maximumError,
            std::abs(first[index] - second[index])
        );
    }

    return maximumError;
}

} // namespace

namespace cuda_backend {

StreamingVerificationResult verifyStreamingCUDA(
    const LBMSolver& solver,
    double tolerance)
{
    // Create the exact pre-stream state using the production CPU collision
    // implementation. Streaming is then compared independently on CPU/GPU.
    LBMSolver preStreamState = solver;
    preStreamState.collideAndForce();

    LBMSolver cpuReference = preStreamState;
    cpuReference.stream();

    const std::size_t nodeCount =
        static_cast<std::size_t>(preStreamState.nodes());

    const std::size_t populationCount =
        nodeCount * static_cast<std::size_t>(Q);

    DeviceArray<double> deviceSourcePopulations(populationCount);
    DeviceArray<double> deviceDestinationPopulations(populationCount);
    DeviceArray<double> deviceDensity(nodeCount);

    deviceSourcePopulations.copyFromHost(preStreamState.populations());
    deviceDensity.copyFromHost(preStreamState.rho());

    constexpr int threadsPerBlock = 256;
    const int blocks =
        (preStreamState.nodes() + threadsPerBlock - 1)
        / threadsPerBlock;

    pullStreamingKernel<<<blocks, threadsPerBlock>>>(
        deviceSourcePopulations.data(),
        deviceDestinationPopulations.data(),
        deviceDensity.data(),
        preStreamState.nx(),
        preStreamState.ny(),
        preStreamState.lidVelocity(),
        preStreamState.periodicY()
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpuPopulations;
    deviceDestinationPopulations.copyToHost(gpuPopulations);

    StreamingVerificationResult result;
    result.maxPopulationError = maximumAbsoluteDifference(
        cpuReference.populations(),
        gpuPopulations
    );

    std::cout
        << "CUDA pull-streaming kernel verification\n"
        << "max |f_CPU - f_GPU| = "
        << result.maxPopulationError << '\n';

    if (result.maxPopulationError > tolerance) {
        throw std::runtime_error(
            "CUDA pull-streaming kernel verification failed."
        );
    }

    std::cout << "CUDA pull-streaming kernel verification: PASSED\n\n";

    return result;
}

} // namespace cuda_backend
