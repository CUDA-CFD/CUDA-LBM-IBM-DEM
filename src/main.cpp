#include "Simulation.hpp"

#ifdef USE_CUDA
#include "CUDAUtils.cuh"
#include "LBMMacroscopic.cuh"
#include "LBMSolver.hpp"
#endif

#include <exception>
#include <iostream>

namespace {

#ifdef USE_CUDA
void runCudaMacroscopicVerification()
{
    cuda_backend::printDeviceInformation();

    LBMSolver verificationSolver(
        32,
        24,
        0.1,
        0.0,
        false
    );

    verificationSolver.initialize();

    // Add a deterministic, spatially varying force so the
    // half-force velocity correction is also verified.
    for (int node = 0; node < verificationSolver.nodes(); ++node) {
        verificationSolver.forceX(node) =
            1.0e-6 * static_cast<double>((node % 7) - 3);
        verificationSolver.forceY(node) =
            1.0e-6 * static_cast<double>((node % 5) - 2);
    }

    verificationSolver.computeMacroscopic();
    cuda_backend::verifyMacroscopicCUDA(verificationSolver);
}
#endif

} // namespace

int main()
{
    try {
#ifdef USE_CUDA
        runCudaMacroscopicVerification();
#endif

        // The complete sedimentation simulation remains on the CPU
        // in this commit. Only computeMacroscopic() is verified on CUDA.
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
