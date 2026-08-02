#pragma once
class LBMSolver;
class IBMParticle;
namespace ibm {
double deltaKernel(double r);
void interpolate(const LBMSolver& fluid, IBMParticle& particle);
void computeDirectForcing(IBMParticle& particle,double dt);
void spreadToFluid(LBMSolver& fluid,const IBMParticle& particle);
void accumulateHydrodynamicLoads(IBMParticle& particle);
}
