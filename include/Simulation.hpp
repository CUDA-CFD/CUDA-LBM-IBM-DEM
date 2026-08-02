#pragma once

#include "IBMParticle.hpp"
#include "LBMSolver.hpp"

#include <fstream>

class Simulation {
public:
    Simulation();
    void run(int steps, int outputInterval);

private:
    LBMSolver fluid_;
    IBMParticle particle_;

    double dt_{1.0};
    double fluidDensity_{1.0};
    double gravity_{2.0e-5};

    std::ofstream csv_;
};
