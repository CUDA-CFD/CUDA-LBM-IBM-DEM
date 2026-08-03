#include "LBMSolverCUDA.cuh"

#include "LBMCollision.cuh"
#include "LBMMacroscopic.cuh"
#include "LBMStreaming.cuh"
#include "LBMSolver.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace {

constexpr int Q = 9;

void requireMatchingSolver(
    const LBMSolver& solver,
    int nx,
    int ny,
    double viscosity,
    double lidVelocity,
    bool periodicY)
{
    if (solver.nx() != nx || solver.ny() != ny) {
        throw std::invalid_argument(
            "Host and CUDA LBM dimensions do not match."
        );
    }

    if (solver.viscosity() != viscosity
        || solver.lidVelocity() != lidVelocity
        || solver.periodicY() != periodicY) {
        throw std::invalid_argument(
            "Host and CUDA LBM parameters do not match."
        );
    }
}

} // namespace

namespace cuda_backend {

LBMSolverCUDA::LBMSolverCUDA(
    int nx,
    int ny,
    double viscosity,
    double lidVelocity,
    bool periodicY)
    : nx_(nx),
      ny_(ny),
      numberOfNodes_(nx * ny),
      lidVelocity_(lidVelocity),
      viscosity_(viscosity),
      relaxationFrequency_(1.0 / (3.0 * viscosity + 0.5)),
      periodicY_(periodicY),
      populationsA_(
          static_cast<std::size_t>(numberOfNodes_) * Q
      ),
      populationsB_(
          static_cast<std::size_t>(numberOfNodes_) * Q
      ),
      density_(static_cast<std::size_t>(numberOfNodes_)),
      velocityX_(static_cast<std::size_t>(numberOfNodes_)),
      velocityY_(static_cast<std::size_t>(numberOfNodes_)),
      forceX_(static_cast<std::size_t>(numberOfNodes_)),
      forceY_(static_cast<std::size_t>(numberOfNodes_)),
      currentPopulations_(&populationsA_),
      nextPopulations_(&populationsB_)
{
    if (nx_ < 3 || ny_ < 3 || viscosity_ <= 0.0) {
        throw std::invalid_argument("Invalid CUDA LBM parameters.");
    }
}

void LBMSolverCUDA::initializeFromHost(const LBMSolver& solver)
{
    requireMatchingSolver(
        solver,
        nx_,
        ny_,
        viscosity_,
        lidVelocity_,
        periodicY_
    );

    populationsA_.copyFromHost(solver.populations());
    forceX_.copyFromHost(solver.forceX());
    forceY_.copyFromHost(solver.forceY());

    CUDA_CHECK(cudaMemset(
        populationsB_.data(),
        0,
        populationsB_.size() * sizeof(double)
    ));

    currentPopulations_ = &populationsA_;
    nextPopulations_ = &populationsB_;

    computeMacroscopic();
}

void LBMSolverCUDA::uploadForces(
    const std::vector<double>& forceX,
    const std::vector<double>& forceY)
{
    forceX_.copyFromHost(forceX);
    forceY_.copyFromHost(forceY);
}

void LBMSolverCUDA::clearForces()
{
    CUDA_CHECK(cudaMemset(
        forceX_.data(),
        0,
        forceX_.size() * sizeof(double)
    ));
    CUDA_CHECK(cudaMemset(
        forceY_.data(),
        0,
        forceY_.size() * sizeof(double)
    ));
}

void LBMSolverCUDA::computeMacroscopic()
{
    launchComputeMacroscopic(
        currentPopulations_->data(),
        forceX_.data(),
        forceY_.data(),
        density_.data(),
        velocityX_.data(),
        velocityY_.data(),
        numberOfNodes_
    );
}

void LBMSolverCUDA::collideAndForce()
{
    launchCollideAndForce(
        currentPopulations_->data(),
        density_.data(),
        velocityX_.data(),
        velocityY_.data(),
        forceX_.data(),
        forceY_.data(),
        relaxationFrequency_,
        numberOfNodes_
    );
}

void LBMSolverCUDA::stream()
{
    launchPullStreaming(
        currentPopulations_->data(),
        nextPopulations_->data(),
        density_.data(),
        nx_,
        ny_,
        lidVelocity_,
        periodicY_
    );

    std::swap(currentPopulations_, nextPopulations_);
}

void LBMSolverCUDA::step()
{
    computeMacroscopic();
    collideAndForce();
    stream();
}

void LBMSolverCUDA::synchronize() const
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

LBMHostState LBMSolverCUDA::downloadState() const
{
    LBMHostState state;
    downloadPopulations(state.populations);
    downloadMacroscopic(
        state.density,
        state.velocityX,
        state.velocityY
    );
    return state;
}

void LBMSolverCUDA::downloadPopulations(
    std::vector<double>& populations) const
{
    currentPopulations_->copyToHost(populations);
}

void LBMSolverCUDA::downloadMacroscopic(
    std::vector<double>& density,
    std::vector<double>& velocityX,
    std::vector<double>& velocityY) const
{
    density_.copyToHost(density);
    velocityX_.copyToHost(velocityX);
    velocityY_.copyToHost(velocityY);
}

} // namespace cuda_backend
