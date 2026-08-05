#include "LBMCollision.cuh"

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

__global__ void collideAndForceKernel(
    double* populations,
    const double* density,
    const double* velocityX,
    const double* velocityY,
    const double* forceX,
    const double* forceY,
    double relaxationFrequency,
    int numberOfNodes)
{
    const int node =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (node >= numberOfNodes) {
        return;
    }

    const double ux = velocityX[node];
    const double uy = velocityY[node];
    const double localDensity = density[node];
    const double localForceX = forceX[node];
    const double localForceY = forceY[node];
    const double velocitySquared = ux * ux + uy * uy;
    const int populationOffset = node * Q;

    #pragma unroll
    for (int direction = 0; direction < Q; ++direction) {
        const double cx = static_cast<double>(deviceCx[direction]);
        const double cy = static_cast<double>(deviceCy[direction]);
        const double velocityProjection = cx * ux + cy * uy;

        const double equilibrium =
            deviceWeight[direction]
            * localDensity
            * (
                1.0
                + 3.0 * velocityProjection
                + 4.5 * velocityProjection * velocityProjection
                - 1.5 * velocitySquared
            );

        const double directionForce =
            cx * localForceX + cy * localForceY;

        const double shiftedDirectionForce =
            (cx - ux) * localForceX
            + (cy - uy) * localForceY;

        const double forcingTerm =
            deviceWeight[direction]
            * (1.0 - 0.5 * relaxationFrequency)
            * (
                3.0 * shiftedDirectionForce
                + 9.0 * velocityProjection * directionForce
            );

        const int populationIndex = populationOffset + direction;
        const double population = populations[populationIndex];

        populations[populationIndex] =
            population
            - relaxationFrequency * (population - equilibrium)
            + forcingTerm;
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

void launchCollideAndForce(
    double* populations,
    const double* density,
    const double* velocityX,
    const double* velocityY,
    const double* forceX,
    const double* forceY,
    double relaxationFrequency,
    int numberOfNodes)
{
    constexpr int threadsPerBlock = 256;
    const int blocks =
        (numberOfNodes + threadsPerBlock - 1) / threadsPerBlock;

    collideAndForceKernel<<<blocks, threadsPerBlock>>>(
        populations, density, velocityX, velocityY, forceX, forceY,
        relaxationFrequency, numberOfNodes
    );

    CUDA_CHECK(cudaGetLastError());
}

CollisionVerificationResult verifyCollisionCUDA(
    const LBMSolver& solver,
    double tolerance)
{
    // Preserve the supplied pre-collision state and calculate the CPU
    // reference using exactly the production CPU implementation.
    LBMSolver cpuReference = solver;
    cpuReference.collideAndForce();

    const std::size_t nodeCount =
        static_cast<std::size_t>(solver.nodes());

    const std::size_t populationCount =
        nodeCount * static_cast<std::size_t>(Q);

    DeviceArray<double> devicePopulations(populationCount);
    DeviceArray<double> deviceDensity(nodeCount);
    DeviceArray<double> deviceVelocityX(nodeCount);
    DeviceArray<double> deviceVelocityY(nodeCount);
    DeviceArray<double> deviceForceX(nodeCount);
    DeviceArray<double> deviceForceY(nodeCount);

    devicePopulations.copyFromHost(solver.populations());
    deviceDensity.copyFromHost(solver.rho());
    deviceVelocityX.copyFromHost(solver.ux());
    deviceVelocityY.copyFromHost(solver.uy());
    deviceForceX.copyFromHost(solver.forceX());
    deviceForceY.copyFromHost(solver.forceY());

    launchCollideAndForce(
        devicePopulations.data(),
        deviceDensity.data(),
        deviceVelocityX.data(),
        deviceVelocityY.data(),
        deviceForceX.data(),
        deviceForceY.data(),
        solver.relaxationFrequency(),
        solver.nodes()
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gpuPopulations;
    devicePopulations.copyToHost(gpuPopulations);

    CollisionVerificationResult result;
    result.maxPopulationError = maximumAbsoluteDifference(
        cpuReference.populations(),
        gpuPopulations
    );

    std::cout
        << "CUDA collision kernel verification\n"
        << "max |f_CPU - f_GPU| = "
        << result.maxPopulationError << '\n';

    if (result.maxPopulationError > tolerance) {
        throw std::runtime_error(
            "CUDA collision kernel verification failed."
        );
    }

    std::cout << "CUDA collision kernel verification: PASSED\n\n";

    return result;
}

} // namespace cuda_backend
