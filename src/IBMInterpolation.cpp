#include "IBMCoupling.hpp"
#include "IBMParticle.hpp"
#include "LBMSolver.hpp"
#include "DeltaKernel.hpp"
#include <cmath>

namespace ibm
{



void interpolate(
    const LBMSolver& fluid,
    IBMParticle& particle)
{
    for (int marker = 0;
         marker < particle.markerCount;
         ++marker)
    {
        particle.fluidUx[marker] = 0.0;
        particle.fluidUy[marker] = 0.0;

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

                particle.fluidUx[marker] +=
                    fluid.velocityX(node) * weight;

                particle.fluidUy[marker] +=
                    fluid.velocityY(node) * weight;
            }
        }
    }
}

} // namespace ibm