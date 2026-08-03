#include "LBMMacroscopic.cuh"

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

__global__ void computeMacroscopicKernel(
    const double* populations,
    const double* forceX,
    const double* forceY,
    double* density,
    double* velocityX,
    double* velocityY,
    int numberOfNodes)
{
    const int node =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (node >= numberOfNodes) {
        return;
    }

    double localDensity = 0.0;
    double momentumX = 0.0;
    double momentumY = 0.0;

    const int populationOffset = node * Q;

    #pragma unroll
    for (int direction = 0; direction < Q; ++direction) {
        const double population =
            populations[populationOffset + direction];

        localDensity += population;
        momentumX += population * deviceCx[direction];
        momentumY += population * deviceCy[direction];
    }

    density[node] = localDensity;
    velocityX[node] =
        (momentumX + 0.5 * forceX[node]) / localDensity;
    velocityY[node] =
        (momentumY + 0.5 * forceY[node]) / localDensity;
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

void launchComputeMacroscopic(
    const double* populations,
    const double* forceX,
    const double* forceY,
    double* density,
    double* velocityX,
    double* velocityY,
    int numberOfNodes)
{
    constexpr int threadsPerBlock = 256;
    const int blocks =
        (numberOfNodes + threadsPerBlock - 1) / threadsPerBlock;

    computeMacroscopicKernel<<<blocks, threadsPerBlock>>>(
        populations, forceX, forceY, density, velocityX, velocityY,
        numberOfNodes
    );

    CUDA_CHECK(cudaGetLastError());
}

MacroscopicVerificationResult verifyMacroscopicCUDA(
    const LBMSolver& solver,
    double tolerance)
{
    const std::size_t nodeCount =
        static_cast<std::size_t>(solver.nodes());

    const std::size_t populationCount =
        nodeCount * static_cast<std::size_t>(Q);

    DeviceArray<double> devicePopulations(populationCount);
    DeviceArray<double> deviceForceX(nodeCount);
    DeviceArray<double> deviceForceY(nodeCount);
    DeviceArray<double> deviceDensity(nodeCount);
    DeviceArray<double> deviceVelocityX(nodeCount);
    DeviceArray<double> deviceVelocityY(nodeCount);

    devicePopulations.copyFromHost(solver.populations());
    deviceForceX.copyFromHost(solver.forceX());
    deviceForceY.copyFromHost(solver.forceY());

    launchComputeMacroscopic(
        devicePopulations.data(),
        deviceForceX.data(),
        deviceForceY.data(),
        deviceDensity.data(),
        deviceVelocityX.data(),
        deviceVelocityY.data(),
        solver.nodes()
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpuDensity;
    std::vector<double> gpuVelocityX;
    std::vector<double> gpuVelocityY;

    deviceDensity.copyToHost(gpuDensity);
    deviceVelocityX.copyToHost(gpuVelocityX);
    deviceVelocityY.copyToHost(gpuVelocityY);

    MacroscopicVerificationResult result;
    result.maxDensityError =
        maximumAbsoluteDifference(solver.rho(), gpuDensity);
    result.maxVelocityXError =
        maximumAbsoluteDifference(solver.ux(), gpuVelocityX);
    result.maxVelocityYError =
        maximumAbsoluteDifference(solver.uy(), gpuVelocityY);

    std::cout
        << "CUDA macroscopic kernel verification\n"
        << "max |rho_CPU - rho_GPU| = "
        << result.maxDensityError << '\n'
        << "max |ux_CPU - ux_GPU|   = "
        << result.maxVelocityXError << '\n'
        << "max |uy_CPU - uy_GPU|   = "
        << result.maxVelocityYError << '\n';

    if (result.maxDensityError > tolerance
        || result.maxVelocityXError > tolerance
        || result.maxVelocityYError > tolerance) {
        throw std::runtime_error(
            "CUDA macroscopic kernel verification failed."
        );
    }

    std::cout << "CUDA macroscopic kernel verification: PASSED\n\n";

    return result;
}

} // namespace cuda_backend
