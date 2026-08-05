#pragma once

struct IBMParticleIntegrationVerificationSettings {
    double maximumAbsoluteTolerance = 1.0e-11;
    double relativeL2Tolerance = 1.0e-9;
};

int runIBMParticleIntegrationVerification(
    const IBMParticleIntegrationVerificationSettings& settings = {});
