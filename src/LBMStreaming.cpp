#include "LBMSolver.hpp"
#include "D2Q9.hpp"

#include <algorithm>

void LBMSolver::stream()
{
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
