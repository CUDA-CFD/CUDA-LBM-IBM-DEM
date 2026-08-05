#include "PerformanceBenchmark.hpp"

#include "CUDAUtils.cuh"
#include "IBMCoupling.hpp"
#include "IBMParticle.hpp"
#include "IBMParticleCUDA.cuh"
#include "LBMSolver.hpp"
#include "LBMSolverCUDA.cuh"
#include "SedimentationBenchmark.hpp"
#include "WallContact.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct TimingSummary
{
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double standardDeviation = 0.0;
};

TimingSummary summarize(const std::vector<double>& samples)
{
    if (samples.empty())
    {
        throw std::invalid_argument("Cannot summarize an empty sample set.");
    }

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    TimingSummary summary;
    summary.minimum = sorted.front();
    summary.maximum = sorted.back();
    summary.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0)
                 / static_cast<double>(sorted.size());

    const std::size_t middle = sorted.size() / 2;
    summary.median = (sorted.size() % 2 == 0)
        ? 0.5 * (sorted[middle - 1] + sorted[middle])
        : sorted[middle];

    double squaredDeviationSum = 0.0;
    for (const double value : sorted)
    {
        const double deviation = value - summary.mean;
        squaredDeviationSum += deviation * deviation;
    }
    summary.standardDeviation = std::sqrt(
        squaredDeviationSum / static_cast<double>(sorted.size())
    );

    return summary;
}

void advanceCpuStep(
    LBMSolver& fluid,
    IBMParticle& particle,
    double gravityForceY,
    double buoyancyForceY)
{
    constexpr double dt = 1.0;

    fluid.clearForces();
    fluid.computeMacroscopic();

    ibm::interpolate(fluid, particle);
    ibm::computeDirectForcing(particle, dt);
    ibm::spreadToFluid(fluid, particle);
    ibm::accumulateHydrodynamicLoads(particle);

    particle.addExternalForce(0.0, gravityForceY + buoyancyForceY);

    const WallContactLoads wallContact = computeWallContact(
        particle,
        static_cast<double>(SedimentationBenchmark::nx),
        static_cast<double>(SedimentationBenchmark::ny),
        SedimentationBenchmark::wallNormalStiffness,
        SedimentationBenchmark::wallNormalDamping
    );
    particle.addExternalForce(wallContact.forceX, wallContact.forceY);

    fluid.computeMacroscopic();
    fluid.collideAndForce();
    fluid.stream();
    particle.integrate(dt);
}

void advanceGpuStep(
    cuda_backend::LBMSolverCUDA& fluid,
    cuda_backend::IBMParticleCUDA& particle,
    double gravityForceY,
    double buoyancyForceY)
{
    constexpr double dt = 1.0;

    fluid.clearForces();
    fluid.computeMacroscopic();

    particle.interpolate(fluid);
    particle.computeDirectForcing(dt);
    particle.spreadToFluid(fluid);
    particle.accumulateHydrodynamicLoads();
    particle.applySedimentationLoads(
        gravityForceY,
        buoyancyForceY,
        static_cast<double>(SedimentationBenchmark::nx),
        static_cast<double>(SedimentationBenchmark::ny),
        SedimentationBenchmark::wallNormalStiffness,
        SedimentationBenchmark::wallNormalDamping
    );

    fluid.computeMacroscopic();
    fluid.collideAndForce();
    fluid.stream();
    particle.integrate(dt);
}

double benchmarkCpu(int warmupSteps, int measuredSteps)
{
    LBMSolver fluid(
        SedimentationBenchmark::nx,
        SedimentationBenchmark::ny,
        SedimentationBenchmark::latticeViscosity,
        0.0,
        true
    );
    fluid.initialize();

    IBMParticle particle(
        SedimentationBenchmark::initialXLattice,
        SedimentationBenchmark::initialYLattice,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );

    const double area = particle.area();
    const double gravityForceY =
        -particle.density * area * SedimentationBenchmark::latticeGravity;
    const double buoyancyForceY =
        SedimentationBenchmark::latticeFluidDensity * area
        * SedimentationBenchmark::latticeGravity;

    for (int step = 0; step < warmupSteps; ++step)
    {
        advanceCpuStep(fluid, particle, gravityForceY, buoyancyForceY);
    }

    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < measuredSteps; ++step)
    {
        advanceCpuStep(fluid, particle, gravityForceY, buoyancyForceY);
    }
    const auto stop = std::chrono::steady_clock::now();

    // Keep the final state observably used without including file output.
    volatile double checksum = particle.y + particle.vy + particle.theta;
    (void)checksum;

    return std::chrono::duration<double>(stop - start).count();
}

double benchmarkGpu(int warmupSteps, int measuredSteps)
{
    LBMSolver hostFluid(
        SedimentationBenchmark::nx,
        SedimentationBenchmark::ny,
        SedimentationBenchmark::latticeViscosity,
        0.0,
        true
    );
    hostFluid.initialize();

    IBMParticle hostParticle(
        SedimentationBenchmark::initialXLattice,
        SedimentationBenchmark::initialYLattice,
        SedimentationBenchmark::particleRadiusLattice,
        SedimentationBenchmark::latticeParticleDensity
    );

    cuda_backend::LBMSolverCUDA fluid(
        hostFluid.nx(),
        hostFluid.ny(),
        hostFluid.viscosity(),
        hostFluid.lidVelocity(),
        hostFluid.periodicY()
    );
    fluid.initializeFromHost(hostFluid);

    cuda_backend::IBMParticleCUDA particle(hostParticle);

    const double area = hostParticle.area();
    const double gravityForceY =
        -hostParticle.density * area * SedimentationBenchmark::latticeGravity;
    const double buoyancyForceY =
        SedimentationBenchmark::latticeFluidDensity * area
        * SedimentationBenchmark::latticeGravity;

    // Warm-up includes the first CUDA context/kernel activity but is excluded
    // from the reported runtime.
    for (int step = 0; step < warmupSteps; ++step)
    {
        advanceGpuStep(fluid, particle, gravityForceY, buoyancyForceY);
    }
    fluid.synchronize();
    particle.synchronize();

    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < measuredSteps; ++step)
    {
        advanceGpuStep(fluid, particle, gravityForceY, buoyancyForceY);
    }
    fluid.synchronize();
    particle.synchronize();
    const auto stop = std::chrono::steady_clock::now();

    return std::chrono::duration<double>(stop - start).count();
}

void writeResultsCsv(
    const PerformanceBenchmarkSettings& settings,
    const std::vector<double>& cpuTimes,
    const std::vector<double>& gpuTimes)
{
    std::ofstream csv("results/performance/speedup_results.csv");
    if (!csv)
    {
        throw std::runtime_error(
            "Unable to open results/performance/speedup_results.csv"
        );
    }

    csv << "backend,repeat,steps,warmup_steps,runtime_seconds,"
           "milliseconds_per_step,steps_per_second,speedup_vs_gpu\n";
    csv << std::setprecision(12);

    for (int repeat = 0; repeat < settings.repeats; ++repeat)
    {
        const double cpu = cpuTimes[static_cast<std::size_t>(repeat)];
        const double gpu = gpuTimes[static_cast<std::size_t>(repeat)];
        const double speedup = cpu / gpu;

        csv << "CPU," << repeat + 1 << ',' << settings.steps << ','
            << settings.warmupSteps << ',' << cpu << ','
            << 1000.0 * cpu / settings.steps << ','
            << settings.steps / cpu << ',' << speedup << '\n';

        csv << "CUDA," << repeat + 1 << ',' << settings.steps << ','
            << settings.warmupSteps << ',' << gpu << ','
            << 1000.0 * gpu / settings.steps << ','
            << settings.steps / gpu << ',' << speedup << '\n';
    }
}

void writeReport(
    const PerformanceBenchmarkSettings& settings,
    const TimingSummary& cpu,
    const TimingSummary& gpu,
    const std::string& gpuName)
{
    std::ofstream report("results/performance/speedup_report.md");
    if (!report)
    {
        throw std::runtime_error(
            "Unable to open results/performance/speedup_report.md"
        );
    }

    const double speedup = cpu.median / gpu.median;

    report << "# CPU-CUDA performance benchmark\n\n"
           << "- Domain: " << SedimentationBenchmark::nx << " x "
           << SedimentationBenchmark::ny << "\n"
           << "- Markers: 79\n"
           << "- Precision: double\n"
           << "- Measured steps per repeat: " << settings.steps << "\n"
           << "- Warm-up steps per repeat: " << settings.warmupSteps << "\n"
           << "- Repeats: " << settings.repeats << "\n"
           << "- CPU hardware concurrency: "
           << std::thread::hardware_concurrency() << "\n"
           << "- GPU: " << gpuName << "\n\n"
           << "| Backend | Median runtime (s) | ms/step | steps/s | std. dev. (s) |\n"
           << "|---|---:|---:|---:|---:|\n"
           << "| CPU | " << cpu.median << " | "
           << 1000.0 * cpu.median / settings.steps << " | "
           << settings.steps / cpu.median << " | "
           << cpu.standardDeviation << " |\n"
           << "| CUDA | " << gpu.median << " | "
           << 1000.0 * gpu.median / settings.steps << " | "
           << settings.steps / gpu.median << " | "
           << gpu.standardDeviation << " |\n\n"
           << "**Median speedup: " << speedup << "x**\n\n"
           << "The timed region excludes allocation, initialization, initial "
              "host-device transfer, CSV output, VTK output, and verification "
              "copies. CUDA is synchronized immediately before and after the "
              "measured timestep loop.\n";
}

} // namespace

int runPerformanceBenchmark(const PerformanceBenchmarkSettings& settings)
{
    if (settings.steps <= 0 || settings.warmupSteps < 0 || settings.repeats <= 0)
    {
        throw std::invalid_argument(
            "Performance benchmark settings must be positive (warm-up may be zero)."
        );
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    // Initialize the CUDA context before any measured repetition.
    CUDA_CHECK(cudaFree(nullptr));

    std::filesystem::create_directories("results/performance");

    std::vector<double> cpuTimes;
    std::vector<double> gpuTimes;
    cpuTimes.reserve(static_cast<std::size_t>(settings.repeats));
    gpuTimes.reserve(static_cast<std::size_t>(settings.repeats));

    std::cout << "Complete sedimentation speedup benchmark\n"
              << "domain: " << SedimentationBenchmark::nx << " x "
              << SedimentationBenchmark::ny << '\n'
              << "steps per repeat: " << settings.steps << '\n'
              << "warm-up steps: " << settings.warmupSteps << '\n'
              << "repeats: " << settings.repeats << "\n\n";

    for (int repeat = 0; repeat < settings.repeats; ++repeat)
    {
        double cpu = 0.0;
        double gpu = 0.0;

        std::cout << "repeat " << repeat + 1 << '/' << settings.repeats << ": ";
        if (repeat % 2 == 0)
        {
            std::cout << "CPU..." << std::flush;
            cpu = benchmarkCpu(settings.warmupSteps, settings.steps);
            std::cout << " " << cpu << " s, CUDA..." << std::flush;
            gpu = benchmarkGpu(settings.warmupSteps, settings.steps);
            std::cout << " " << gpu << " s";
        }
        else
        {
            std::cout << "CUDA..." << std::flush;
            gpu = benchmarkGpu(settings.warmupSteps, settings.steps);
            std::cout << " " << gpu << " s, CPU..." << std::flush;
            cpu = benchmarkCpu(settings.warmupSteps, settings.steps);
            std::cout << " " << cpu << " s";
        }
        std::cout << ", speedup=" << cpu / gpu << "x\n";

        cpuTimes.push_back(cpu);
        gpuTimes.push_back(gpu);
    }

    const TimingSummary cpuSummary = summarize(cpuTimes);
    const TimingSummary gpuSummary = summarize(gpuTimes);
    const double medianSpeedup = cpuSummary.median / gpuSummary.median;

    writeResultsCsv(settings, cpuTimes, gpuTimes);
    writeReport(settings, cpuSummary, gpuSummary, properties.name);

    std::cout << std::fixed << std::setprecision(6)
              << "\nCPU median: " << cpuSummary.median << " s\n"
              << "CUDA median: " << gpuSummary.median << " s\n"
              << "Median speedup: " << medianSpeedup << "x\n"
              << "Results: results/performance/speedup_results.csv\n"
              << "Report: results/performance/speedup_report.md\n";

    return 0;
}
