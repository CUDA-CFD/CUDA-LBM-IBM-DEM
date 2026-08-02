#include "Simulation.hpp"

#include "IBMCoupling.hpp"
#include "VTKWriter.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

Simulation::Simulation()
    : fluid_(100, 220, 0.10, 0.0),
      particle_(50.0, 175.0, 8.0, 1.25),
      csv_("sedimentation.csv") {
    fluid_.initialize();

    if (!csv_) {
        throw std::runtime_error("Unable to open sedimentation.csv");
    }

    csv_ << "step,time,x,y,vx,vy,omega,"
            "FhydroX,FhydroY,FgravityY,FbuoyancyY,"
            "FexternalY,FtotalX,FtotalY\n";
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

        // Commit 006: gravity and buoyancy are the only external loads.
        // Particle-wall contact is intentionally deferred to Commit 007.
        particle_.addExternalForce(0.0, externalForceY);

        fluid_.computeMacroscopic();
        fluid_.collideAndForce();
        fluid_.stream();

        particle_.integrate(dt_);

        csv_ << t << ',' << t * dt_ << ','
             << particle_.x << ',' << particle_.y << ','
             << particle_.vx << ',' << particle_.vy << ','
             << particle_.omega << ','
             << hydrodynamicForceX << ',' << hydrodynamicForceY << ','
             << gravityForceY << ',' << buoyancyForceY << ','
             << externalForceY << ','
             << particle_.forceX << ',' << particle_.forceY << '\n';

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

            const double forceBalanceY =
                hydrodynamicForceY + externalForceY;

            const double particleReynolds =
                2.0 * particle_.radius * std::abs(particle_.vy) /
                fluid_.viscosity();

            std::cout << "step=" << t
                      << " y=" << particle_.y
                      << " vy=" << particle_.vy
                      << " Re_p=" << particleReynolds
                      << " Fh_y=" << hydrodynamicForceY
                      << " Fg=" << gravityForceY
                      << " Fb=" << buoyancyForceY
                      << " Fnet_y=" << forceBalanceY
                      << '\n';
        }
    }
}
