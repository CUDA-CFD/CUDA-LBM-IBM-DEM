#pragma once

struct IBMForcingVerificationSettings {
    double maximumAbsoluteTolerance = 1.0e-12;
    double relativeL2Tolerance = 1.0e-9;
};

int runIBMForcingVerification(
    const IBMForcingVerificationSettings& settings = {}
);
