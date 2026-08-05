#include "CUDAUtils.cuh"

#include <iomanip>
#include <iostream>

namespace cuda_backend {

void printDeviceInformation()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA-capable GPU was detected.");
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const double memoryGiB =
        static_cast<double>(properties.totalGlobalMem)
        / (1024.0 * 1024.0 * 1024.0);

    std::cout
        << "CUDA backend enabled\n"
        << "Device: " << properties.name << '\n'
        << "Compute capability: "
        << properties.major << '.' << properties.minor << '\n'
        << "Global memory: " << std::fixed << std::setprecision(2)
        << memoryGiB << " GiB\n"
        << "Multiprocessors: " << properties.multiProcessorCount << '\n'
        << "Warp size: " << properties.warpSize << "\n\n";
}

} // namespace cuda_backend
