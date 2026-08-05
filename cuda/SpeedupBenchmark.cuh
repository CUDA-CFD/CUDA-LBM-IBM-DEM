#pragma once

namespace cuda_backend
{

struct SpeedupBenchmarkSettings
{
    int steps = 10000;
    int warmupSteps = 500;
    int repeats = 5;
};

/**
 * @brief Benchmark the complete CPU and CUDA sedimentation timestep pipelines.
 *
 * The measured region includes only the physics timestep loop. Initialization,
 * allocation, seed copies, trajectory output, VTK output, verification copies,
 * and console output inside the loop are excluded.
 */
int runSpeedupBenchmark(const SpeedupBenchmarkSettings& settings);

} // namespace cuda_backend
