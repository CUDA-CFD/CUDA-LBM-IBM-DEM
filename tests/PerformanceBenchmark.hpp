#pragma once

struct PerformanceBenchmarkSettings
{
    int steps = 10000;
    int warmupSteps = 500;
    int repeats = 5;
};

int runPerformanceBenchmark(const PerformanceBenchmarkSettings& settings);
