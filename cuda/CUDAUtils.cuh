#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cuda_backend {

inline void checkCuda(
    cudaError_t error,
    const char* expression,
    const char* file,
    int line)
{
    if (error == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string("CUDA error: ") + cudaGetErrorString(error)
        + "\nExpression: " + expression
        + "\nLocation: " + file + ":" + std::to_string(line)
    );
}

void printDeviceInformation();

} // namespace cuda_backend

#define CUDA_CHECK(expression) \
    ::cuda_backend::checkCuda((expression), #expression, __FILE__, __LINE__)
