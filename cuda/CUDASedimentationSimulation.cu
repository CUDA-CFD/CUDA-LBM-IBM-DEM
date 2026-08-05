#include "CUDASedimentationSimulation.cuh"

#include "IBMParticle.hpp"
#include "IBMParticleCUDA.cuh"
#include "LBMSolver.hpp"
#include "LBMSolverCUDA.cuh"
#include "SedimentationBenchmark.hpp"
#include "VTKWriter.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{

void writeCsvHeader(std::ofstream& csv)
{
    csv << "step,time_s,x_lu,y_lu,x_m,y_m,vx_lu,vy_lu,ax_lu,ay_lu,"
           "theta,omega,alpha,FhydroX,FhydroY,FgravityY,FbuoyancyY,"
           "FexternalY,FwallX,FwallY,wallOverlap,wallContactActive,"
           "FtotalX,FtotalY,translationalKE,rotationalKE,totalKE,"
           "particleReynolds,verticalForceResidual\n";
}

std::string numberedFilename(
    const char* prefix,
    int step)
{
    std::ostringstream filename;
    filename << prefix
             << std::setw(6)
             << std::setfill('0')
             << step
             << ".vtk";
    return filename.str();
}

} // namespace

namespace cuda_backend
{

int runCudaSedimentation(int steps, int outputInterval)
{
    if (steps <= 0)
    {
        throw std::invalid_argument(
            "CUDA sedimentation steps must be positive."
        );
    }
    if (outputInterval <= 0)
    {
        throw std::invalid_argument(
            "CUDA sedimentation output interval must be positive."
        );
    }

    constexpr double dt = 1.0;
    constexpr double fluidDensity =
        SedimentationBenchmark::latticeFluidDensity;
    constexpr double gravity = SedimentationBenchmark::latticeGravity;
    constexpr double wallNormalStiffness =
        SedimentationBenchmark::wallNormalStiffness;
    constexpr double wallNormalDamping =
        SedimentationBenchmark::wallNormalDamping;

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

    LBMSolverCUDA fluid(
        hostFluid.nx(),
        hostFluid.ny(),
        hostFluid.viscosity(),
        hostFluid.lidVelocity(),
        hostFluid.periodicY()
    );
    fluid.initializeFromHost(hostFluid);

    IBMParticleCUDA particle(hostParticle);

    const double particleArea = hostParticle.area();
    const double gravityForceY =
        -hostParticle.density * particleArea * gravity;
    const double buoyancyForceY =
        fluidDensity * particleArea * gravity;

    std::ofstream csv("sedimentation_gpu.csv");
    if (!csv)
    {
        throw std::runtime_error("Unable to open sedimentation_gpu.csv");
    }
    writeCsvHeader(csv);

    std::cout
        << "CUDA benchmark: " << SedimentationBenchmark::name << '\n'
        << "Reference: " << SedimentationBenchmark::reference << '\n'
        << "Grid: " << SedimentationBenchmark::nx << " x "
        << SedimentationBenchmark::ny << '\n'
        << "Particle diameter: "
        << 2.0 * SedimentationBenchmark::particleRadiusLattice
        << " lattice units\n"
        << "tau: " << SedimentationBenchmark::relaxationTime << '\n'
        << "physical dt: "
        << SedimentationBenchmark::physicalTimeStep << " s\n"
        << "steps: " << steps << '\n'
        << "VTK output interval: " << outputInterval << "\n\n";

    std::vector<double> density;
    std::vector<double> velocityX;
    std::vector<double> velocityY;

    for (int step = 0; step < steps; ++step)
    {
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
            wallNormalStiffness,
            wallNormalDamping
        );

        fluid.computeMacroscopic();
        fluid.collideAndForce();
        fluid.stream();

        particle.integrate(dt);

        // Only the compact rigid state and load diagnostics are copied every
        // timestep for trajectory output. The full fluid and marker fields are
        // downloaded only at VTK output steps below.
        particle.downloadRigidStateToHost(hostParticle);
        const IBMSedimentationLoadsCUDA loads =
            particle.downloadSedimentationLoads();

        const double translationalKE =
            hostParticle.translationalKineticEnergy();
        const double rotationalKE =
            hostParticle.rotationalKineticEnergy();
        const double totalKE = translationalKE + rotationalKE;

        const double particleReynolds =
            2.0 * hostParticle.radius
            * std::hypot(hostParticle.vx, hostParticle.vy)
            / SedimentationBenchmark::latticeViscosity;

        const double verticalForceResidual =
            loads.hydrodynamicForceY + loads.externalForceY;

        csv << step << ','
            << step * SedimentationBenchmark::physicalTimeStep << ','
            << hostParticle.x << ',' << hostParticle.y << ','
            << hostParticle.x * SedimentationBenchmark::dx << ','
            << hostParticle.y * SedimentationBenchmark::dx << ','
            << hostParticle.vx << ',' << hostParticle.vy << ','
            << hostParticle.ax << ',' << hostParticle.ay << ','
            << hostParticle.theta << ',' << hostParticle.omega << ','
            << hostParticle.alpha << ','
            << loads.hydrodynamicForceX << ','
            << loads.hydrodynamicForceY << ','
            << loads.gravityForceY << ','
            << loads.buoyancyForceY << ','
            << loads.externalForceY << ','
            << loads.wallForceX << ',' << loads.wallForceY << ','
            << loads.maximumWallOverlap << ','
            << loads.wallContactActive << ','
            << loads.totalForceX << ',' << loads.totalForceY << ','
            << translationalKE << ',' << rotationalKE << ',' << totalKE << ','
            << particleReynolds << ',' << verticalForceResidual << '\n';

        if (step % outputInterval == 0)
        {
            fluid.downloadMacroscopic(density, velocityX, velocityY);
            particle.downloadToHost(hostParticle);

            io::writeFluidField(
                fluid.nx(),
                fluid.ny(),
                velocityX,
                velocityY,
                numberedFilename("fluid_gpu_", step)
            );
            io::writeParticle(
                hostParticle,
                numberedFilename("particle_gpu_", step)
            );

            std::cout
                << "step=" << step
                << " y=" << hostParticle.y
                << " vy=" << hostParticle.vy
                << " ay=" << hostParticle.ay
                << " Re_p=" << particleReynolds
                << " Fh_y=" << loads.hydrodynamicForceY
                << " Fg=" << loads.gravityForceY
                << " Fb=" << loads.buoyancyForceY
                << " Fwall_y=" << loads.wallForceY
                << " overlap=" << loads.maximumWallOverlap
                << " Fnet_y=" << loads.totalForceY
                << " KE=" << totalKE
                << '\n';
        }
    }

    particle.synchronize();

    std::cout
        << "\nCUDA sedimentation completed.\n"
        << "Trajectory: sedimentation_gpu.csv\n"
        << "Full fluid and marker data were copied to the host only at "
        << "VTK output steps.\n";

    return 0;
}

} // namespace cuda_backend
