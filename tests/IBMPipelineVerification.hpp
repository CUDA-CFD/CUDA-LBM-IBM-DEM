#pragma once

struct IBMPipelineVerificationSettings
{
    int steps = 100;
    int interval = 10;
    double maximumAbsoluteTolerance = 1.0e-10;
    double relativeL2Tolerance = 1.0e-8;
};

int runIBMPipelineVerification(
    const IBMPipelineVerificationSettings& settings = {}
);
