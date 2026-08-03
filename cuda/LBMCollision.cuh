#pragma once

class LBMSolver;

namespace cuda_backend {

struct CollisionVerificationResult {
    double maxPopulationError = 0.0;
};

void launchCollideAndForce(
    double* populations,
    const double* density,
    const double* velocityX,
    const double* velocityY,
    const double* forceX,
    const double* forceY,
    double relaxationFrequency,
    int numberOfNodes
);

CollisionVerificationResult verifyCollisionCUDA(
    const LBMSolver& solver,
    double tolerance = 1.0e-13
);

} // namespace cuda_backend
