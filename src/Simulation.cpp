#include "Simulation.hpp"
#include "IBMCoupling.hpp"
#include "VTKWriter.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

Simulation::Simulation()
    : fluid_(100, 220, 0.10, 0.0),
      particle_(50.0, 175.0, 8.0, 1.25),
      csv_("sedimentation.csv") {
    fluid_.initialize();
    csv_ << "step,time,x,y,vx,vy,omega,FhydroX,FhydroY,FgravityBuoyancyY,FtotalX,FtotalY\n";
}

void Simulation::run(int steps, int outputInterval) {
    constexpr double pi = 3.14159265358979323846;
    const double displacedArea = pi * particle_.radius * particle_.radius;
    const double effectiveWeightY =
        -(particle_.density - fluidDensity_) * displacedArea * gravity_;

    for (int t = 0; t < steps; ++t) {
        fluid_.clearForces();
        fluid_.computeMacroscopic();

        ibm::interpolate(fluid_, particle_);
        ibm::computeDirectForcing(particle_, dt_);
        ibm::spreadToFluid(fluid_, particle_);
        ibm::accumulateHydrodynamicLoads(particle_);

        const double FhydroX = particle_.forceX;
        const double FhydroY = particle_.forceY;

        particle_.addExternalForce(0.0, effectiveWeightY);
        particle_.addWallContact(
            0.5, fluid_.nx() - 1.5,
            0.5, fluid_.ny() - 1.5,
            wallKn_, wallCn_);

        fluid_.computeMacroscopic();
        fluid_.collideAndForce();
        fluid_.stream();

        particle_.integrate(dt_);

        csv_ << t << ',' << t * dt_ << ','
             << particle_.x << ',' << particle_.y << ','
             << particle_.vx << ',' << particle_.vy << ',' << particle_.omega << ','
             << FhydroX << ',' << FhydroY << ',' << effectiveWeightY << ','
             << particle_.forceX << ',' << particle_.forceY << '\n';

        if (t % outputInterval == 0) {
            std::ostringstream a, b;
            a << "fluid_" << std::setw(6) << std::setfill('0') << t << ".vtk";
            b << "particle_" << std::setw(6) << std::setfill('0') << t << ".vtk";
            io::writeFluid(fluid_, a.str());
            io::writeParticle(particle_, b.str());

            const double forceBalanceY = FhydroY + effectiveWeightY;
            const double reynolds = 2.0 * particle_.radius * std::abs(particle_.vy) /
                                    fluid_.viscosity();
            std::cout << "step=" << t
                      << " y=" << particle_.y
                      << " vy=" << particle_.vy
                      << " Re_p=" << reynolds
                      << " Fh_y=" << FhydroY
                      << " W-B=" << effectiveWeightY
                      << " balance_y=" << forceBalanceY
                      << '\n';
        }
    }
}
