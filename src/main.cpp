#include "Simulation.hpp"

#ifdef USE_CUDA
#include "CUDAUtils.cuh"
#include "GPUVerification.hpp"
#include "GPULBMRunner.cuh"
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
    int steps = 1000;
    int verificationInterval = 100;
};

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
        else if (argument == "--steps") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--steps requires a value.");
            }
            options.steps =
                parsePositiveInteger(argv[++index], "--steps");
        }
        else if (argument == "--interval") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--interval requires a value.");
            }
            options.verificationInterval =
                parsePositiveInteger(argv[++index], "--interval");
        }
        else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage:\n"
                << "  lbm_ibm_dem\n"
                << "  lbm_ibm_dem --gpu-lbm [--steps N]\n"
                << "  lbm_ibm_dem --verify [--steps N] [--interval N]\n";
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

        if (options.verify && options.gpuLbm) {
            throw std::invalid_argument(
                "Choose either --verify or --gpu-lbm, not both."
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

        // The complete sedimentation simulation remains on the CPU.
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
