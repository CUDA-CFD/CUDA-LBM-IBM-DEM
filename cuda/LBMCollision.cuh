#pragma once

class LBMSolver;

namespace cuda_backend {

struct CollisionVerificationResult {
    double maxPopulationError = 0.0;
};

CollisionVerificationResult verifyCollisionCUDA(
    const LBMSolver& solver,
    double tolerance = 1.0e-13
);

} // namespace cuda_backend
