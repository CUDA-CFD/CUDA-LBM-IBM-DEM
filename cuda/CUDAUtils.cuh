#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

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

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(std::size_t count)
        : count_(count)
    {
        if (count_ == 0) {
            return;
        }

        checkCuda(
            cudaMalloc(
                reinterpret_cast<void**>(&data_),
                count_ * sizeof(T)
            ),
            "cudaMalloc",
            __FILE__,
            __LINE__
        );
    }

    ~DeviceArray()
    {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& other) noexcept
        : data_(other.data_),
          count_(other.count_)
    {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    DeviceArray& operator=(DeviceArray&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }

        if (data_ != nullptr) {
            cudaFree(data_);
        }

        data_ = other.data_;
        count_ = other.count_;
        other.data_ = nullptr;
        other.count_ = 0;
        return *this;
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }

    void copyFromHost(const std::vector<T>& host)
    {
        if (host.size() != count_) {
            throw std::invalid_argument("CUDA copy size mismatch.");
        }

        checkCuda(
            cudaMemcpy(
                data_,
                host.data(),
                count_ * sizeof(T),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy HostToDevice",
            __FILE__,
            __LINE__
        );
    }

    void copyToHost(std::vector<T>& host) const
    {
        host.resize(count_);

        checkCuda(
            cudaMemcpy(
                host.data(),
                data_,
                count_ * sizeof(T),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy DeviceToHost",
            __FILE__,
            __LINE__
        );
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace cuda_backend

#define CUDA_CHECK(expression) \
    ::cuda_backend::checkCuda((expression), #expression, __FILE__, __LINE__)
