#include "LBMSolver.hpp"
#include "D2Q9.hpp"
#include <algorithm>
#include <stdexcept>

LBMSolver::LBMSolver(
    int nx,
    int ny,
    double viscosity,
    double lidVelocity,
    bool periodicY
)
    : nx_(nx), ny_(ny), n_(nx * ny), lidVelocity_(lidVelocity), nu_(viscosity),
      tau_(3.0 * nu_ + 0.5), omega_(1.0 / tau_), periodicY_(periodicY),
      f_(n_ * lbm::D2Q9::Q), fNew_(n_ * lbm::D2Q9::Q), rho_(n_), ux_(n_), uy_(n_),
      forceX_(n_), forceY_(n_) {
    if (nx_ < 3 || ny_ < 3 || nu_ <= 0.0 || tau_ <= 0.5) {
        throw std::invalid_argument("Invalid LBM parameters");
    }
}

int LBMSolver::popIndex(int node, int dir) const noexcept {
    return node * lbm::D2Q9::Q + dir;
}

void LBMSolver::initialize() {
    for (int n = 0; n < n_; ++n) {
        rho_[n] = 1.0;
        ux_[n] = uy_[n] = forceX_[n] = forceY_[n] = 0.0;
        for (int i = 0; i < lbm::D2Q9::Q; ++i) {
            f_[popIndex(n, i)] = lbm::D2Q9::w[i];
            fNew_[popIndex(n, i)] = 0.0;
        }
    }
}

void LBMSolver::clearForces() {
    std::fill(forceX_.begin(), forceX_.end(), 0.0);
    std::fill(forceY_.begin(), forceY_.end(), 0.0);
}

void LBMSolver::computeMacroscopic() {
    for (int n = 0; n < n_; ++n) {
        double r = 0.0, jx = 0.0, jy = 0.0;
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

void LBMSolver::collideAndForce() {
    for (int n = 0; n < n_; ++n) {
        const double u2 = ux_[n] * ux_[n] + uy_[n] * uy_[n];
        for (int i = 0; i < lbm::D2Q9::Q; ++i) {
            const int p = popIndex(n, i);
            const double cu = lbm::D2Q9::cx[i] * ux_[n] + lbm::D2Q9::cy[i] * uy_[n];
            const double feq = lbm::D2Q9::w[i] * rho_[n] *
                (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
            const double ciF = lbm::D2Q9::cx[i] * forceX_[n] + lbm::D2Q9::cy[i] * forceY_[n];
            const double cmiF = (lbm::D2Q9::cx[i] - ux_[n]) * forceX_[n] +
                                (lbm::D2Q9::cy[i] - uy_[n]) * forceY_[n];
            const double Si = lbm::D2Q9::w[i] * (1.0 - 0.5 * omega_) *
                              (3.0 * cmiF + 9.0 * cu * ciF);
            f_[p] = f_[p] - omega_ * (f_[p] - feq) + Si;
        }
    }
}

void LBMSolver::stream() {
    std::fill(fNew_.begin(), fNew_.end(), 0.0);

    for (int y = 0; y < ny_; ++y) {
        for (int x = 0; x < nx_; ++x) {
            const int node = nodeIndex(x, y);

            for (int i = 0; i < lbm::D2Q9::Q; ++i) {
                int destinationX = x + lbm::D2Q9::cx[i];
                int destinationY = y + lbm::D2Q9::cy[i];
                const int source = popIndex(node, i);

                // The sedimentation benchmark is periodic in the vertical
                // direction and no-slip at the two side walls.
                if (periodicY_) {
                    if (destinationX < 0 || destinationX >= nx_) {
                        const int reflected =
                            popIndex(node, lbm::D2Q9::opposite[i]);
                        fNew_[reflected] = f_[source];
                        continue;
                    }

                    if (destinationY < 0) {
                        destinationY += ny_;
                    } else if (destinationY >= ny_) {
                        destinationY -= ny_;
                    }

                    const int destinationNode =
                        nodeIndex(destinationX, destinationY);
                    fNew_[popIndex(destinationNode, i)] = f_[source];
                    continue;
                }

                const bool destinationInside =
                    destinationX >= 0 && destinationX < nx_ &&
                    destinationY >= 0 && destinationY < ny_;

                if (destinationInside) {
                    const int destinationNode =
                        nodeIndex(destinationX, destinationY);
                    fNew_[popIndex(destinationNode, i)] = f_[source];
                } else {
                    const int reflected =
                        popIndex(node, lbm::D2Q9::opposite[i]);

                    if (destinationY >= ny_ && lidVelocity_ != 0.0) {
                        fNew_[reflected] =
                            f_[source]
                            - 6.0 * lbm::D2Q9::w[i] * rho_[node]
                                  * lbm::D2Q9::cx[i] * lidVelocity_;
                    } else {
                        fNew_[reflected] = f_[source];
                    }
                }
            }
        }
    }

    f_.swap(fNew_);
}
