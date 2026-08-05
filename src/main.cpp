#include "Simulation.hpp"

#ifdef USE_CUDA
#include "CUDAUtils.cuh"
#include "GPUVerification.hpp"
#include "GPULBMRunner.cuh"
#include "IBMInterpolationVerification.hpp"
#include "IBMForcingVerification.hpp"
#include "IBMSpreadingVerification.hpp"
#include "IBMHydrodynamicLoadsVerification.hpp"
#include "IBMParticleIntegrationVerification.hpp"
#include "IBMPipelineVerification.hpp"
#include "CUDASedimentationSimulation.cuh"
#include "CUDASedimentationVerification.hpp"
#include "PerformanceBenchmark.hpp"
#endif

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CommandLineOptions {
    bool verify = false;
    bool gpuLbm = false;
    bool verifyIbmInterpolation = false;
    bool verifyIbmForcing = false;
    bool verifyIbmSpreading = false;
    bool verifyIbmHydrodynamicLoads = false;
    bool verifyIbmParticleIntegration = false;
    bool verifyIbmPipeline = false;
    bool gpuSedimentation = false;
    bool verifyCudaSedimentation = false;
    bool benchmarkSpeedup = false;
    bool stepsProvided = false;
    bool intervalProvided = false;
    int steps = 1000;
    int verificationInterval = 100;
    int warmupSteps = 500;
    int repeats = 5;
};

int parseNonNegativeInteger(const char* text, const char* optionName)
{
    try {
        const int value = std::stoi(text);
        if (value < 0) {
            throw std::invalid_argument("negative");
        }
        return value;
    }
    catch (const std::exception&) {
        throw std::invalid_argument(
            std::string(optionName) + " requires a non-negative integer."
        );
    }
}

int parsePositiveInteger(const char* text, const char* optionName)
{
    try {
        const int value = std::stoi(text);
        if (value <= 0) {
            throw std::invalid_argument("not positive");
        }
        return value;
    }
    catch (const std::exception&) {
        throw std::invalid_argument(
            std::string(optionName) + " requires a positive integer."
        );
    }
}

CommandLineOptions parseCommandLine(int argc, char** argv)
{
    CommandLineOptions options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];

        if (argument == "--verify") {
            options.verify = true;
        }
        else if (argument == "--gpu-lbm") {
            options.gpuLbm = true;
        }
        else if (argument == "--verify-ibm-interpolation") {
            options.verifyIbmInterpolation = true;
        }
        else if (argument == "--verify-ibm-forcing") {
            options.verifyIbmForcing = true;
        }
        else if (argument == "--verify-ibm-spreading") {
            options.verifyIbmSpreading = true;
        }
        else if (argument == "--verify-ibm-hydrodynamic-loads") {
            options.verifyIbmHydrodynamicLoads = true;
        }
        else if (argument == "--verify-ibm-particle-integration") {
            options.verifyIbmParticleIntegration = true;
        }
        else if (argument == "--verify-ibm-pipeline") {
            options.verifyIbmPipeline = true;
        }
        else if (argument == "--gpu-sedimentation") {
            options.gpuSedimentation = true;
        }
        else if (argument == "--verify-cuda-sedimentation") {
            options.verifyCudaSedimentation = true;
        }
        else if (argument == "--benchmark-speedup") {
            options.benchmarkSpeedup = true;
        }
        else if (argument == "--steps") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--steps requires a value.");
            }
            options.steps =
                parsePositiveInteger(argv[++index], "--steps");
            options.stepsProvided = true;
        }
        else if (argument == "--warmup") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--warmup requires a value.");
            }
            options.warmupSteps =
                parseNonNegativeInteger(argv[++index], "--warmup");
        }
        else if (argument == "--repeats") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--repeats requires a value.");
            }
            options.repeats =
                parsePositiveInteger(argv[++index], "--repeats");
        }
        else if (argument == "--interval") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--interval requires a value.");
            }
            options.verificationInterval =
                parsePositiveInteger(argv[++index], "--interval");
            options.intervalProvided = true;
        }
        else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage:\n"
                << "  lbm_ibm_dem\n"
                << "  lbm_ibm_dem --gpu-lbm [--steps N]\n"
                << "  lbm_ibm_dem --verify [--steps N] [--interval N]\n"
                << "  lbm_ibm_dem --verify-ibm-interpolation\n"
                << "  lbm_ibm_dem --verify-ibm-forcing\n"
                << "  lbm_ibm_dem --verify-ibm-spreading\n"
                << "  lbm_ibm_dem --verify-ibm-hydrodynamic-loads\n"
                << "  lbm_ibm_dem --verify-ibm-particle-integration\n"
                << "  lbm_ibm_dem --verify-ibm-pipeline "
                   "[--steps N] [--interval N]\n"
                << "  lbm_ibm_dem --verify-cuda-sedimentation "
                   "[--steps N] [--interval N]\n"
                << "  lbm_ibm_dem --gpu-sedimentation "
                   "[--steps N] [--interval N]\n"
                << "  lbm_ibm_dem --benchmark-speedup "
                   "[--steps N] [--warmup N] [--repeats N]\n";
            std::exit(0);
        }
        else {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }

    return options;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const CommandLineOptions options = parseCommandLine(argc, argv);

        const int selectedCudaModes =
            static_cast<int>(options.verify) +
            static_cast<int>(options.gpuLbm) +
            static_cast<int>(options.verifyIbmInterpolation) +
            static_cast<int>(options.verifyIbmForcing) +
            static_cast<int>(options.verifyIbmSpreading) +
            static_cast<int>(options.verifyIbmHydrodynamicLoads) +
            static_cast<int>(options.verifyIbmParticleIntegration) +
            static_cast<int>(options.verifyIbmPipeline) +
            static_cast<int>(options.gpuSedimentation) +
            static_cast<int>(options.verifyCudaSedimentation) +
            static_cast<int>(options.benchmarkSpeedup);

        if (selectedCudaModes > 1) {
            throw std::invalid_argument(
                "Choose only one CUDA execution or verification mode."
            );
        }

        if (options.verify) {
#ifdef USE_CUDA
            cuda_backend::printDeviceInformation();

            GPUVerificationSettings settings;
            settings.steps = options.steps;
            settings.interval = options.verificationInterval;
            return runFullDomainGPUVerification(settings);
#else
            throw std::runtime_error(
                "GPU verification requires a CUDA-enabled build."
            );
#endif
        }

        if (options.verifyIbmInterpolation) {
#ifdef USE_CUDA
            cuda_backend::printDeviceInformation();
            return runIBMInterpolationVerification();
#else
            throw std::runtime_error(
                "IBM interpolation verification requires a CUDA-enabled build."
            );
#endif
        }

        if (options.verifyIbmForcing) {
#ifdef USE_CUDA
            cuda_backend::printDeviceInformation();
            return runIBMForcingVerification();
#else
            throw std::runtime_error(
                "IBM direct-forcing verification requires a CUDA-enabled build."
            );
#endif
        }

        if (options.verifyIbmSpreading) {
#ifdef USE_CUDA
            cuda_backend::printDeviceInformation();
            return runIBMSpreadingVerification();
#else
            throw std::runtime_error(
                "IBM force-spreading verification requires a CUDA-enabled build."
            );
#endif
        }

        if (options.verifyIbmHydrodynamicLoads) {
#ifdef USE_CUDA
            cuda_backend::printDeviceInformation();
            return runIBMHydrodynamicLoadsVerification();
#else
            throw std::runtime_error(
                "IBM hydrodynamic-load verification requires a CUDA-enabled build."
            );
#endif
        }

        if (options.verifyIbmParticleIntegration) {
#ifdef USE_CUDA
            cuda_backend::printDeviceInformation();
            return runIBMParticleIntegrationVerification();
#else
            throw std::runtime_error(
                "IBM particle-integration verification requires a CUDA-enabled build."
            );
#endif
        }

        if (options.verifyIbmPipeline) {
#ifdef USE_CUDA
            cuda_backend::printDeviceInformation();
            IBMPipelineVerificationSettings settings;
            settings.steps = options.stepsProvided ? options.steps : 100;
            settings.interval =
                options.intervalProvided ? options.verificationInterval : 10;
            return runIBMPipelineVerification(settings);
#else
            throw std::runtime_error(
                "Persistent IBM pipeline verification requires a CUDA-enabled build."
            );
#endif
        }


        if (options.verifyCudaSedimentation) {
#ifdef USE_CUDA
            cuda_backend::printDeviceInformation();
            CUDASedimentationVerificationSettings settings;
            settings.steps = options.stepsProvided ? options.steps : 100;
            settings.interval =
                options.intervalProvided ? options.verificationInterval : 10;
            return runCUDASedimentationVerification(settings);
#else
            throw std::runtime_error(
                "CUDA sedimentation verification requires a CUDA-enabled build."
            );
#endif
        }


        if (options.benchmarkSpeedup) {
#ifdef USE_CUDA
            cuda_backend::printDeviceInformation();
            PerformanceBenchmarkSettings settings;
            settings.steps = options.stepsProvided ? options.steps : 10000;
            settings.warmupSteps = options.warmupSteps;
            settings.repeats = options.repeats;
            return runPerformanceBenchmark(settings);
#else
            throw std::runtime_error(
                "Speedup benchmarking requires a CUDA-enabled build."
            );
#endif
        }

        if (options.gpuSedimentation) {
#ifdef USE_CUDA
            cuda_backend::printDeviceInformation();
            const int steps = options.stepsProvided
                ? options.steps
                : Simulation::defaultSteps();
            const int outputInterval = options.intervalProvided
                ? options.verificationInterval
                : Simulation::defaultOutputInterval();
            return cuda_backend::runCudaSedimentation(
                steps,
                outputInterval
            );
#else
            throw std::runtime_error(
                "CUDA sedimentation requires a CUDA-enabled build."
            );
#endif
        }

        if (options.gpuLbm) {
#ifdef USE_CUDA
            cuda_backend::printDeviceInformation();
            return cuda_backend::runPersistentGpuLbm(options.steps);
#else
            throw std::runtime_error(
                "The GPU LBM mode requires a CUDA-enabled build."
            );
#endif
        }

        // No CUDA mode selected: run the validated CPU sedimentation path.
        Simulation simulation;
        simulation.run(
            Simulation::defaultSteps(),
            Simulation::defaultOutputInterval()
        );
    }
    catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}
