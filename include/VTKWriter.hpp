#pragma once

#include <string>
#include <vector>

class LBMSolver;
class IBMParticle;

namespace io
{

void writeFluid(const LBMSolver& fluid, const std::string& filename);
void writeFluidField(
    int nx,
    int ny,
    const std::vector<double>& velocityX,
    const std::vector<double>& velocityY,
    const std::string& filename
);
void writeParticle(const IBMParticle& particle, const std::string& filename);

} // namespace io
