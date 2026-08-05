#pragma once

#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

namespace ibm
{

HD inline double deltaKernel(double r)
{
    r = std::abs(r);

    if (r < 1.0)
    {
        return 0.125 *
            (3.0 - 2.0 * r +
             std::sqrt(1.0 + 4.0 * r - 4.0 * r * r));
    }

    if (r < 2.0)
    {
        return 0.125 *
            (5.0 - 2.0 * r -
             std::sqrt(-7.0 + 12.0 * r - 4.0 * r * r));
    }

    return 0.0;
}

} // namespace ibm

#undef HD