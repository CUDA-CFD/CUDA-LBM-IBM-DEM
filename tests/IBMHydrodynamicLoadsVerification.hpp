#pragma once

struct IBMHydrodynamicLoadsVerificationSettings {
    double maximumAbsoluteTolerance = 1.0e-12;
    double relativeTolerance = 1.0e-9;
};

int runIBMHydrodynamicLoadsVerification(
    const IBMHydrodynamicLoadsVerificationSettings& settings = {});
