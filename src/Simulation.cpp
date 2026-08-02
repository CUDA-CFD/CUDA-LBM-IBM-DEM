#include "Simulation.hpp"

#include "IBMCoupling.hpp"
#include "VTKWriter.hpp"
#include "WallContact.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

Simulation::Simulation()
    : fluid_(
          SedimentationBenchmark::nx,
          SedimentationBenchmark::ny,
          SedimentationBenchmark::latticeViscosity,
          0.0,
          true
      ),
      particle_(
          SedimentationBenchmark::initialXLattice,
          SedimentationBenchmark::initialYLattice,
          SedimentationBenchmark::particleRadiusLattice,
          SedimentationBenchmark::latticeParticleDensity
      ),
      csv_("sedimentation.csv") {
    fluid_.initialize();

    if (!csv_) {
        throw std::runtime_error("Unable to open sedimentation.csv");
    }

    std::cout << "Benchmark: " << SedimentationBenchmark::name << '\n'
              << "Reference: " << SedimentationBenchmark::reference << '\n'
              << "Grid: " << SedimentationBenchmark::nx << " x "
              << SedimentationBenchmark::ny << '\n'
              << "Particle diameter: "
              << 2.0 * SedimentationBenchmark::particleRadiusLattice
              << " lattice units\n"
              << "tau: " << SedimentationBenchmark::relaxationTime << '\n'
              << "physical dt: "
              << SedimentationBenchmark::physicalTimeStep << " s\n\n";

    csv_ << "step,time_s,x_lu,y_lu,x_m,y_m,vx_lu,vy_lu,ax_lu,ay_lu,theta,omega,alpha,"
            "FhydroX,FhydroY,FgravityY,FbuoyancyY,FexternalY,"
            "FwallX,FwallY,wallOverlap,wallContactActive,"
            "FtotalX,FtotalY,translationalKE,rotationalKE,totalKE,"
            "particleReynolds,verticalForceResidual\n";
}

void Simulation::run(int steps, int outputInterval) {
    const double particleArea = particle_.area();

    // Downward weight of the two-dimensional disk with unit thickness.
    const double gravityForceY =
        -particle_.density * particleArea * gravity_;

    // Upward Archimedes force from the displaced fluid.
    const double buoyancyForceY =
        fluidDensity_ * particleArea * gravity_;

    const double externalForceY =
        gravityForceY + buoyancyForceY;

    for (int t = 0; t < steps; ++t) {
        fluid_.clearForces();
        fluid_.computeMacroscopic();

        ibm::interpolate(fluid_, particle_);
        ibm::computeDirectForcing(particle_, dt_);
        ibm::spreadToFluid(fluid_, particle_);
        ibm::accumulateHydrodynamicLoads(particle_);

        const double hydrodynamicForceX = particle_.forceX;
        const double hydrodynamicForceY = particle_.forceY;

        particle_.addExternalForce(0.0, externalForceY);

        // Prevent the rigid particle from crossing the container boundary.
        const WallContactLoads wallContact = computeWallContact(
            particle_,
            static_cast<double>(SedimentationBenchmark::nx),
            static_cast<double>(SedimentationBenchmark::ny),
            wallNormalStiffness_,
            wallNormalDamping_
        );
        particle_.addExternalForce(wallContact.forceX, wallContact.forceY);

        const double totalForceX = particle_.forceX;
        const double totalForceY = particle_.forceY;

        fluid_.computeMacroscopic();
        fluid_.collideAndForce();
        fluid_.stream();

        particle_.integrate(dt_);

        const double translationalKE =
            particle_.translationalKineticEnergy();
        const double rotationalKE =
            particle_.rotationalKineticEnergy();
        const double totalKE = translationalKE + rotationalKE;

        const double particleReynolds =
            2.0 * particle_.radius * std::hypot(particle_.vx, particle_.vy) /
            fluid_.viscosity();

        // At terminal settling this residual should approach zero.
        const double verticalForceResidual =
            hydrodynamicForceY + externalForceY;

        csv_ << t << ','
             << t * SedimentationBenchmark::physicalTimeStep << ','
             << particle_.x << ',' << particle_.y << ','
             << particle_.x * SedimentationBenchmark::dx << ','
             << particle_.y * SedimentationBenchmark::dx << ','
             << particle_.vx << ',' << particle_.vy << ',' 
             << particle_.ax << ',' << particle_.ay << ','
             << particle_.theta << ',' << particle_.omega << ','
             << particle_.alpha << ','
             << hydrodynamicForceX << ',' << hydrodynamicForceY << ','
             << gravityForceY << ',' << buoyancyForceY << ','
             << externalForceY << ','
             << wallContact.forceX << ',' << wallContact.forceY << ','
             << wallContact.maximumOverlap << ','
             << static_cast<int>(wallContact.active) << ','
             << totalForceX << ',' << totalForceY << ','
             << translationalKE << ',' << rotationalKE << ',' << totalKE << ','
             << particleReynolds << ',' << verticalForceResidual << '\n';

        if (t % outputInterval == 0) {
            std::ostringstream fluidFilename;
            std::ostringstream particleFilename;

            fluidFilename << "fluid_"
                          << std::setw(6)
                          << std::setfill('0')
                          << t
                          << ".vtk";

            particleFilename << "particle_"
                             << std::setw(6)
                             << std::setfill('0')
                             << t
                             << ".vtk";

            io::writeFluid(fluid_, fluidFilename.str());
            io::writeParticle(particle_, particleFilename.str());

            std::cout << "step=" << t
                      << " y=" << particle_.y
                      << " vy=" << particle_.vy
                      << " ay=" << particle_.ay
                      << " Re_p=" << particleReynolds
                      << " Fh_y=" << hydrodynamicForceY
                      << " Fg=" << gravityForceY
                      << " Fb=" << buoyancyForceY
                      << " Fwall_y=" << wallContact.forceY
                      << " overlap=" << wallContact.maximumOverlap
                      << " Fnet_y=" << totalForceY
                      << " KE=" << totalKE
                      << '\n';
        }
    }
}
