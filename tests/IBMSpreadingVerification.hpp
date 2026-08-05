#pragma once

struct IBMSpreadingVerificationSettings {
    double maximumAbsoluteTolerance = 1.0e-12;
    double relativeL2Tolerance = 1.0e-9;
};

int runIBMSpreadingVerification(
    const IBMSpreadingVerificationSettings& settings = {});
