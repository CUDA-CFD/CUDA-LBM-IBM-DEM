#include "IBMCoupling.hpp"
#include "IBMParticle.hpp"

namespace ibm
{

void computeDirectForcing(
    IBMParticle& particle,
    double dt)
{
    for (int marker = 0;
         marker < particle.markerCount;
         ++marker)
    {
        const double rx =
            particle.markerX[marker] - particle.x;

        const double ry =
            particle.markerY[marker] - particle.y;

        const double boundaryVelocityX =
            particle.vx - particle.omega * ry;

        const double boundaryVelocityY =
            particle.vy + particle.omega * rx;

        particle.markerForceX[marker] =
            (boundaryVelocityX -
             particle.fluidUx[marker]) / dt;

        particle.markerForceY[marker] =
            (boundaryVelocityY -
             particle.fluidUy[marker]) / dt;
    }
}

} // namespace ibm