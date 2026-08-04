#include "IBMCoupling.hpp"
#include "IBMParticle.hpp"

namespace ibm
{

void accumulateHydrodynamicLoads(
    IBMParticle& particle)
{
    particle.clearLoads();

    for (int marker = 0;
         marker < particle.markerCount;
         ++marker)
    {
        const double forceX =
            -particle.markerForceX[marker] *
            particle.ds;

        const double forceY =
            -particle.markerForceY[marker] *
            particle.ds;

        const double rx =
            particle.markerX[marker] - particle.x;

        const double ry =
            particle.markerY[marker] - particle.y;

        particle.forceX += forceX;
        particle.forceY += forceY;

        particle.torque +=
            rx * forceY - ry * forceX;
    }
}

} // namespace ibm