#pragma once

#include "IBMParticle.hpp"
#include "LBMSolver.hpp"
#include "SedimentationBenchmark.hpp"

#include <fstream>

class Simulation {
public:
    Simulation();
    void run(int steps, int outputInterval);

    static constexpr int defaultSteps() noexcept {
        return SedimentationBenchmark::timeSteps;
    }

    static constexpr int defaultOutputInterval() noexcept {
        return SedimentationBenchmark::outputInterval;
    }

private:
    LBMSolver fluid_;
    IBMParticle particle_;

    double dt_{1.0};
    double fluidDensity_{SedimentationBenchmark::latticeFluidDensity};
    double gravity_{SedimentationBenchmark::latticeGravity};
    double wallNormalStiffness_{SedimentationBenchmark::wallNormalStiffness};
    double wallNormalDamping_{SedimentationBenchmark::wallNormalDamping};

    std::ofstream csv_;
};
