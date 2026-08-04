#include "IBMCoupling.hpp"
#include "DeltaKernel.hpp"
#include "IBMParticle.hpp"
#include "LBMSolver.hpp"

#include <cmath>

namespace ibm
{

void spreadToFluid(
    LBMSolver& fluid,
    const IBMParticle& particle)
{
    for (int marker = 0;
         marker < particle.markerCount;
         ++marker)
    {
        const int baseX = static_cast<int>(
            std::floor(particle.markerX[marker]));

        const int baseY = static_cast<int>(
            std::floor(particle.markerY[marker]));

        for (int y = baseY - 1; y <= baseY + 2; ++y)
        {
            for (int x = baseX - 1; x <= baseX + 2; ++x)
            {
                if (x < 0 || x >= fluid.nx() ||
                    y < 0 || y >= fluid.ny())
                {
                    continue;
                }

                const double weight =
                    deltaKernel(x - particle.markerX[marker]) *
                    deltaKernel(y - particle.markerY[marker]);

                const int node = fluid.nodeIndex(x, y);

                fluid.forceX(node) +=
                    particle.markerForceX[marker] *
                    weight *
                    particle.ds;

                fluid.forceY(node) +=
                    particle.markerForceY[marker] *
                    weight *
                    particle.ds;
            }
        }
    }
}

} // namespace ibm