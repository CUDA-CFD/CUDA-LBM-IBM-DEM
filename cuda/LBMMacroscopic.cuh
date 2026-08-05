#pragma once

class LBMSolver;

namespace cuda_backend {

struct MacroscopicVerificationResult {
    double maxDensityError = 0.0;
    double maxVelocityXError = 0.0;
    double maxVelocityYError = 0.0;
};

void launchComputeMacroscopic(
    const double* populations,
    const double* forceX,
    const double* forceY,
    double* density,
    double* velocityX,
    double* velocityY,
    int numberOfNodes
);

MacroscopicVerificationResult verifyMacroscopicCUDA(
    const LBMSolver& solver,
    double tolerance = 1.0e-13
);

} // namespace cuda_backend
