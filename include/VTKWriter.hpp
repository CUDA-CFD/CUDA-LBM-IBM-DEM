#pragma once
#include <string>
class LBMSolver;
class IBMParticle;
namespace io {
void writeFluid(const LBMSolver& fluid,const std::string& filename);
void writeParticle(const IBMParticle& particle,const std::string& filename);
}
