#include "SpeedupBenchmark.cuh"

#include "CUDASedimentationSimulation.cuh"
#include "CUDAUtils.cuh"
#include "IBMParticle.hpp"
#include "SedimentationBenchmark.hpp"
#include "Simulation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct TimingStatistics
{
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double standardDeviation = 0.0;
};

TimingStatistics calculateStatistics(const std::vector<double>& values)
{
    if (values.empty())
    {
        throw std::invalid_argument("Timing statistics require data.");
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    const double sum =
        std::accumulate(values.begin(), values.end(), 0.0);
    const double mean = sum / static_cast<double>(values.size());

    double squaredDeviationSum = 0.0;
    for (const double value : values)
    {
        const double deviation = value - mean;
        squaredDeviationSum += deviation * deviation;
    }

    const std::size_t middle = sorted.size() / 2;
    const double median = sorted.size() % 2 == 0
        ? 0.5 * (sorted[middle - 1] + sorted[middle])
        : sorted[middle];

    TimingStatistics statistics;
    statistics.minimum = sorted.front();
    statistics.maximum = sorted.back();
    statistics.mean = mean;
    statistics.median = median;
    statistics.standardDeviation = std::sqrt(
        squaredDeviationSum / static_cast<double>(values.size())
    );
    return statistics;
}

double measureCpuSedimentationComputeSeconds(int steps)
{
    Simulation simulation(false);

    const auto start = std::chrono::steady_clock::now();
    simulation.runComputeOnly(steps);
    const auto stop = std::chrono::steady_clock::now();

    return std::chrono::duration<double>(stop - start).count();
}

const char* buildConfiguration()
{
#ifdef NDEBUG
    return "Release";
#else
    return "Debug";
#endif
}

void writeResultsCsv(
    const std::filesystem::path& path,
    int steps,
    const std::vector<double>& cpuTimes,
    const std::vector<double>& gpuTimes)
{
    std::ofstream csv(path);
    if (!csv)
    {
        throw std::runtime_error(
            "Unable to open performance CSV: " + path.string()
        );
    }

    csv << "repeat,steps,cpu_seconds,gpu_seconds,"
           "cpu_ms_per_step,gpu_ms_per_step,"
           "cpu_steps_per_second,gpu_steps_per_second,speedup\n";

    csv << std::setprecision(17);
    for (std::size_t repeat = 0; repeat < cpuTimes.size(); ++repeat)
    {
        const double cpuSeconds = cpuTimes[repeat];
        const double gpuSeconds = gpuTimes[repeat];
        const double stepCount = static_cast<double>(steps);

        csv << repeat + 1 << ','
            << steps << ','
            << cpuSeconds << ','
            << gpuSeconds << ','
            << 1000.0 * cpuSeconds / stepCount << ','
            << 1000.0 * gpuSeconds / stepCount << ','
            << stepCount / cpuSeconds << ','
            << stepCount / gpuSeconds << ','
            << cpuSeconds / gpuSeconds << '\n';
    }
}

void writeReport(
    const std::filesystem::path& path,
    const cudaDeviceProp& deviceProperties,
    int cudaRuntimeVersion,
    int markerCount,
    const cuda_backend::SpeedupBenchmarkSettings& settings,
    const TimingStatistics& cpu,
    const TimingStatistics& gpu,
    double medianSpeedup)
{
    std::ofstream report(path);
    if (!report)
    {
        throw std::runtime_error(
            "Unable to open performance report: " + path.string()
        );
    }

    report << std::fixed << std::setprecision(6)
           << "# CPU-CUDA sedimentation performance\n\n"
           << "## Benchmark configuration\n\n"
           << "- Domain: " << SedimentationBenchmark::nx << " x "
           << SedimentationBenchmark::ny << "\n"
           << "- LBM model: D2Q9 BGK with Guo forcing\n"
           << "- Particle markers: " << markerCount << "\n"
           << "- Precision: double\n"
           << "- Measured timesteps per repeat: " << settings.steps << "\n"
           << "- Warm-up timesteps: " << settings.warmupSteps << "\n"
           << "- Repeats: " << settings.repeats << "\n"
           << "- Build configuration: " << buildConfiguration() << "\n"
           << "- CPU solver threads: 1\n"
           << "- Host hardware concurrency reported by C++: "
           << std::thread::hardware_concurrency() << "\n"
           << "- GPU: " << deviceProperties.name << "\n"
           << "- Compute capability: " << deviceProperties.major << '.'
           << deviceProperties.minor << "\n"
           << "- CUDA runtime version: "
           << cudaRuntimeVersion / 1000 << '.'
           << (cudaRuntimeVersion % 1000) / 10 << "\n\n"
           << "## Timing scope\n\n"
           << "The measured region contains the complete sedimentation physics "
              "timestep only. It excludes allocation, initialization, initial "
              "host-device copies, CSV/VTK output, verification downloads, and "
              "console output. CUDA timing uses synchronized host wall-clock "
              "time, so kernel-launch overhead is included.\n\n"
           << "## Summary\n\n"
           << "| Metric | CPU | CUDA |\n"
           << "|---|---:|---:|\n"
           << "| Minimum time (s) | " << cpu.minimum << " | "
           << gpu.minimum << " |\n"
           << "| Mean time (s) | " << cpu.mean << " | "
           << gpu.mean << " |\n"
           << "| Median time (s) | " << cpu.median << " | "
           << gpu.median << " |\n"
           << "| Maximum time (s) | " << cpu.maximum << " | "
           << gpu.maximum << " |\n"
           << "| Standard deviation (s) | " << cpu.standardDeviation
           << " | " << gpu.standardDeviation << " |\n"
           << "| Median time per step (ms) | "
           << 1000.0 * cpu.median / settings.steps << " | "
           << 1000.0 * gpu.median / settings.steps << " |\n"
           << "| Median steps per second | "
           << settings.steps / cpu.median << " | "
           << settings.steps / gpu.median << " |\n\n"
           << "**Median compute-only speedup: " << medianSpeedup
           << "x**\n\n"
           << "Raw repeat measurements are stored in `speedup_results.csv`.\n";
}

} // namespace

namespace cuda_backend
{

int runSpeedupBenchmark(const SpeedupBenchmarkSettings& settings)
{
    if (settings.steps <= 0)
    {
        throw std::invalid_argument(
            "Speedup benchmark steps must be positive."
        );
    }
    if (settings.warmupSteps < 0)
    {
        throw std::invalid_argument(
            "Speedup benchmark warm-up steps cannot be negative."
        );
    }
    if (settings.repeats <= 0)
    {
        throw std::invalid_argument(
            "Speedup benchmark repeats must be positive."
        );
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    int cudaRuntimeVersion = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&cudaRuntimeVersion));

    // Force CUDA context creation before either warm-up or measured timing.
    CUDA_CHECK(cudaFree(nullptr));

    std::cout
        << "Complete sedimentation compute-only speedup benchmark\n"
        << "domain: " << SedimentationBenchmark::nx << " x "
        << SedimentationBenchmark::ny << '\n'
        << "steps per repeat: " << settings.steps << '\n'
        << "warm-up steps: " << settings.warmupSteps << '\n'
        << "repeats: " << settings.repeats << '\n'
        << "CPU solver threads: 1\n"
        << "timed output or verification copies: no\n\n";

    if (settings.warmupSteps > 0)
    {
        std::cout << "Warm-up: CPU..." << std::flush;
        {
            Simulation cpuWarmup(false);
            cpuWarmup.runComputeOnly(settings.warmupSteps);
        }
        std::cout << " done\nWarm-up: CUDA..." << std::flush;
        (void)measureCudaSedimentationComputeSeconds(
            settings.warmupSteps
        );
        std::cout << " done\n\n";
    }

    std::vector<double> cpuTimes(
        static_cast<std::size_t>(settings.repeats)
    );
    std::vector<double> gpuTimes(
        static_cast<std::size_t>(settings.repeats)
    );

    std::cout << std::fixed << std::setprecision(6);

    for (int repeat = 0; repeat < settings.repeats; ++repeat)
    {
        // Alternate measurement order to reduce a systematic thermal or
        // background-load bias between the CPU and GPU measurements.
        if (repeat % 2 == 0)
        {
            cpuTimes[static_cast<std::size_t>(repeat)] =
                measureCpuSedimentationComputeSeconds(settings.steps);
            gpuTimes[static_cast<std::size_t>(repeat)] =
                measureCudaSedimentationComputeSeconds(settings.steps);
        }
        else
        {
            gpuTimes[static_cast<std::size_t>(repeat)] =
                measureCudaSedimentationComputeSeconds(settings.steps);
            cpuTimes[static_cast<std::size_t>(repeat)] =
                measureCpuSedimentationComputeSeconds(settings.steps);
        }

        const double speedup =
            cpuTimes[static_cast<std::size_t>(repeat)]
            / gpuTimes[static_cast<std::size_t>(repeat)];

        std::cout
            << "repeat " << repeat + 1 << '/' << settings.repeats
            << ": CPU="
            << cpuTimes[static_cast<std::size_t>(repeat)] << " s"
            << "  CUDA="
            << gpuTimes[static_cast<std::size_t>(repeat)] << " s"
            << "  speedup=" << speedup << "x\n";
    }

    const TimingStatistics cpuStatistics =
        calculateStatistics(cpuTimes);
    const TimingStatistics gpuStatistics =
        calculateStatistics(gpuTimes);
    const double medianSpeedup =
        cpuStatistics.median / gpuStatistics.median;

    const IBMParticle metadataParticle(
        SedimentationBenchmark::initialXLattice,
        SedimentationBenchmark::initialYLattice,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );

    const std::filesystem::path outputDirectory =
        std::filesystem::path("results") / "performance";
    std::filesystem::create_directories(outputDirectory);

    const std::filesystem::path csvPath =
        outputDirectory / "speedup_results.csv";
    const std::filesystem::path reportPath =
        outputDirectory / "speedup_report.md";

    writeResultsCsv(
        csvPath,
        settings.steps,
        cpuTimes,
        gpuTimes
    );
    writeReport(
        reportPath,
        deviceProperties,
        cudaRuntimeVersion,
        metadataParticle.markerCount,
        settings,
        cpuStatistics,
        gpuStatistics,
        medianSpeedup
    );

    std::cout
        << "\nMedian CPU time: " << cpuStatistics.median << " s\n"
        << "Median CUDA time: " << gpuStatistics.median << " s\n"
        << "Median CPU time per step: "
        << 1000.0 * cpuStatistics.median / settings.steps << " ms\n"
        << "Median CUDA time per step: "
        << 1000.0 * gpuStatistics.median / settings.steps << " ms\n"
        << "Median compute-only speedup: " << medianSpeedup << "x\n\n"
        << "Results: " << csvPath.string() << '\n'
        << "Report: " << reportPath.string() << '\n'
        << "Plot with: python tools/plot_speedup.py\n";

    return 0;
}

} // namespace cuda_backend
