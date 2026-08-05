#include "LBMSolver.hpp"
#include "D2Q9.hpp"

#include <algorithm>

void LBMSolver::initialize()
{
    for (int n = 0; n < n_; ++n) {
        rho_[n] = 1.0;
        ux_[n] = 0.0;
        uy_[n] = 0.0;
        forceX_[n] = 0.0;
        forceY_[n] = 0.0;

        for (int i = 0; i < lbm::D2Q9::Q; ++i) {
            f_[popIndex(n, i)] = lbm::D2Q9::w[i];
            fNew_[popIndex(n, i)] = 0.0;
        }
    }
}

void LBMSolver::clearForces()
{
    std::fill(forceX_.begin(), forceX_.end(), 0.0);
    std::fill(forceY_.begin(), forceY_.end(), 0.0);
}
