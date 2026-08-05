#pragma once

#include "CUDAUtils.cuh"

#include <vector>

class LBMSolver;

namespace cuda_backend {

struct LBMHostState {
    std::vector<double> populations;
    std::vector<double> density;
    std::vector<double> velocityX;
    std::vector<double> velocityY;
};

class LBMSolverCUDA {
public:
    LBMSolverCUDA(
        int nx,
        int ny,
        double viscosity,
        double lidVelocity = 0.0,
        bool periodicY = false
    );

    LBMSolverCUDA(const LBMSolverCUDA&) = delete;
    LBMSolverCUDA& operator=(const LBMSolverCUDA&) = delete;
    LBMSolverCUDA(LBMSolverCUDA&&) = delete;
    LBMSolverCUDA& operator=(LBMSolverCUDA&&) = delete;

    void initializeFromHost(const LBMSolver& solver);
    void uploadForces(
        const std::vector<double>& forceX,
        const std::vector<double>& forceY
    );
    void clearForces();

    void computeMacroscopic();
    void collideAndForce();
    void stream();
    void step();
    void synchronize() const;

    LBMHostState downloadState() const;
    void downloadPopulations(std::vector<double>& populations) const;
    void downloadMacroscopic(
        std::vector<double>& density,
        std::vector<double>& velocityX,
        std::vector<double>& velocityY
    ) const;
    void downloadForces(
        std::vector<double>& forceX,
        std::vector<double>& forceY
    ) const;

    const double* deviceVelocityX() const noexcept
    {
        return velocityX_.data();
    }
    const double* deviceVelocityY() const noexcept
    {
        return velocityY_.data();
    }
    double* deviceForceX() noexcept { return forceX_.data(); }
    double* deviceForceY() noexcept { return forceY_.data(); }

    int nx() const noexcept { return nx_; }
    int ny() const noexcept { return ny_; }
    int nodes() const noexcept { return numberOfNodes_; }
    double relaxationFrequency() const noexcept {
        return relaxationFrequency_;
    }

private:
    int nx_;
    int ny_;
    int numberOfNodes_;
    double lidVelocity_;
    double viscosity_;
    double relaxationFrequency_;
    bool periodicY_;

    DeviceArray<double> populationsA_;
    DeviceArray<double> populationsB_;
    DeviceArray<double> density_;
    DeviceArray<double> velocityX_;
    DeviceArray<double> velocityY_;
    DeviceArray<double> forceX_;
    DeviceArray<double> forceY_;

    DeviceArray<double>* currentPopulations_;
    DeviceArray<double>* nextPopulations_;
};

} // namespace cuda_backend
