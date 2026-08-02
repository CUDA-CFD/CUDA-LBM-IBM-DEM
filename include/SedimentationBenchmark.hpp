#pragma once

#include <string_view>

/**
 * @brief Published single-particle sedimentation benchmark configuration.
 *
 * Physical case reproduced by Pu et al. (Physics of Fluids, 2023):
 * a circular particle settling in a 2 cm x 6 cm channel.
 * The lattice conversion uses 100 cells per centimetre and tau = 0.8.
 */
struct SedimentationBenchmark {
    static constexpr std::string_view name =
        "Single circular particle sedimentation";
    static constexpr std::string_view reference =
        "Pu et al., Physics of Fluids 35, 053307 (2023)";

    // Physical parameters (SI units).
    static constexpr double channelWidth = 0.02;       // m
    static constexpr double channelHeight = 0.06;      // m
    static constexpr double particleDiameter = 0.0025; // m
    static constexpr double initialX = 0.01;            // m, channel centreline
    static constexpr double initialY = 0.04;            // m
    static constexpr double fluidDensity = 1000.0;      // kg/m^3
    static constexpr double particleDensity = 1250.0;   // kg/m^3
    static constexpr double physicalViscosity = 1.0e-5; // m^2/s
    static constexpr double physicalGravity = 9.81;     // m/s^2

    // Discretisation reported for the reference N = 100 case.
    static constexpr int cellsPerCentimetre = 100;
    static constexpr double dx = 1.0e-4; // m
    static constexpr int nx = 200;
    static constexpr int ny = 600;

    static constexpr double relaxationTime = 0.8;
    static constexpr double latticeViscosity =
        (relaxationTime - 0.5) / 3.0; // 0.1

    // nu_phys = nu_lattice * dx^2 / dt_phys
    static constexpr double physicalTimeStep =
        latticeViscosity * dx * dx / physicalViscosity; // 1e-4 s

    // g_lattice = g_phys * dt_phys^2 / dx
    static constexpr double latticeGravity =
        physicalGravity * physicalTimeStep * physicalTimeStep / dx;

    static constexpr double particleRadiusLattice =
        0.5 * particleDiameter / dx; // 12.5 lu
    static constexpr double initialXLattice = initialX / dx; // 100 lu
    static constexpr double initialYLattice = initialY / dx; // 400 lu

    // LBM uses rho_f = 1. Particle density is represented by its ratio.
    static constexpr double latticeFluidDensity = 1.0;
    static constexpr double latticeParticleDensity =
        particleDensity / fluidDensity; // 1.25

    static constexpr double simulatedPhysicalTime = 1.0; // s
    static constexpr int timeSteps =
        static_cast<int>(simulatedPhysicalTime / physicalTimeStep); // 10000
    static constexpr int outputInterval = 500;
};
