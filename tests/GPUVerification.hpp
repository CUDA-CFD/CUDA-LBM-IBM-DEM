#pragma once

#include <string>

struct GPUVerificationSettings {
    int steps = 1000;
    int interval = 100;
    double maximumAbsoluteTolerance = 1.0e-12;
    double relativeL2Tolerance = 1.0e-9;
    std::string csvFilename = "results/gpu_verification.csv";
};

int runFullDomainGPUVerification(
    const GPUVerificationSettings& settings = {}
);
