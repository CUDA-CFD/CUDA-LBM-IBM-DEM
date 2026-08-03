#include "LBMSolver.hpp"
#include "D2Q9.hpp"

void LBMSolver::computeMacroscopic()
{
    for (int n = 0; n < n_; ++n) {
        double r = 0.0;
        double jx = 0.0;
        double jy = 0.0;

        for (int i = 0; i < lbm::D2Q9::Q; ++i) {
            const double fi = f_[popIndex(n, i)];
            r += fi;
            jx += fi * lbm::D2Q9::cx[i];
            jy += fi * lbm::D2Q9::cy[i];
        }

        rho_[n] = r;
        ux_[n] = (jx + 0.5 * forceX_[n]) / r;
        uy_[n] = (jy + 0.5 * forceY_[n]) / r;
    }
}
