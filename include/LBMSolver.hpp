#pragma once
#include <vector>

class LBMSolver {
public:
    LBMSolver(
        int nx,
        int ny,
        double viscosity,
        double lidVelocity = 0.0,
        bool periodicY = false
    );

    void initialize();
    void computeMacroscopic();
    void collideAndForce();
    void stream();
    void clearForces();

    int nx() const noexcept { return nx_; }
    int ny() const noexcept { return ny_; }
    int nodes() const noexcept { return n_; }
    double lidVelocity() const noexcept { return lidVelocity_; }
    double viscosity() const noexcept { return nu_; }
    double relaxationFrequency() const noexcept { return omega_; }
    bool periodicY() const noexcept { return periodicY_; }

    int nodeIndex(int x, int y) const noexcept { return y * nx_ + x; }
    double& forceX(int node) { return forceX_[node]; }
    double& forceY(int node) { return forceY_[node]; }
    double velocityX(int node) const { return ux_[node]; }
    double velocityY(int node) const { return uy_[node]; }
    double density(int node) const { return rho_[node]; }

    const std::vector<double>& rho() const noexcept { return rho_; }
    const std::vector<double>& ux() const noexcept { return ux_; }
    const std::vector<double>& uy() const noexcept { return uy_; }
    const std::vector<double>& forceX() const noexcept { return forceX_; }
    const std::vector<double>& forceY() const noexcept { return forceY_; }

private:
    int nx_, ny_, n_;
    double lidVelocity_, nu_, tau_, omega_;
    bool periodicY_;
    std::vector<double> f_, fNew_, rho_, ux_, uy_, forceX_, forceY_;
    int popIndex(int node, int dir) const noexcept;
};
