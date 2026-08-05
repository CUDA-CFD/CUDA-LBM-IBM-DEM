#include "LBMSolver.hpp"
#include "D2Q9.hpp"

#include <stdexcept>

LBMSolver::LBMSolver(
    int nx,
    int ny,
    double viscosity,
    double lidVelocity,
    bool periodicY
)
    : nx_(nx),
      ny_(ny),
      n_(nx * ny),
      lidVelocity_(lidVelocity),
      nu_(viscosity),
      tau_(3.0 * nu_ + 0.5),
      omega_(1.0 / tau_),
      periodicY_(periodicY),
      f_(n_ * lbm::D2Q9::Q),
      fNew_(n_ * lbm::D2Q9::Q),
      rho_(n_),
      ux_(n_),
      uy_(n_),
      forceX_(n_),
      forceY_(n_)
{
    if (nx_ < 3 || ny_ < 3 || nu_ <= 0.0 || tau_ <= 0.5) {
        throw std::invalid_argument("Invalid LBM parameters");
    }
}

int LBMSolver::popIndex(int node, int dir) const noexcept
{
    return node * lbm::D2Q9::Q + dir;
}
