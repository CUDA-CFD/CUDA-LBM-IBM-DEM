#pragma once

class LBMSolver;

namespace cuda_backend {

struct StreamingVerificationResult {
    double maxPopulationError = 0.0;
};

StreamingVerificationResult verifyStreamingCUDA(
    const LBMSolver& solver,
    double tolerance = 1.0e-13
);

} // namespace cuda_backend
