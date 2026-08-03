#include "LBMSolver.hpp"
#include "D2Q9.hpp"

void LBMSolver::collideAndForce()
{
    for (int n = 0; n < n_; ++n) {
        const double u2 = ux_[n] * ux_[n] + uy_[n] * uy_[n];

        for (int i = 0; i < lbm::D2Q9::Q; ++i) {
            const int p = popIndex(n, i);
            const double cu =
                lbm::D2Q9::cx[i] * ux_[n] +
                lbm::D2Q9::cy[i] * uy_[n];

            const double feq =
                lbm::D2Q9::w[i] * rho_[n] *
                (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);

            const double ciF =
                lbm::D2Q9::cx[i] * forceX_[n] +
                lbm::D2Q9::cy[i] * forceY_[n];

            const double cmiF =
                (lbm::D2Q9::cx[i] - ux_[n]) * forceX_[n] +
                (lbm::D2Q9::cy[i] - uy_[n]) * forceY_[n];

            const double Si =
                lbm::D2Q9::w[i] * (1.0 - 0.5 * omega_) *
                (3.0 * cmiF + 9.0 * cu * ciF);

            f_[p] = f_[p] - omega_ * (f_[p] - feq) + Si;
        }
    }
}
