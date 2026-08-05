#include "VTKWriter.hpp"

#include "IBMParticle.hpp"
#include "LBMSolver.hpp"

#include <fstream>
#include <stdexcept>

namespace io
{

void writeFluidField(
    int nx,
    int ny,
    const std::vector<double>& velocityX,
    const std::vector<double>& velocityY,
    const std::string& filename)
{
    const std::size_t expectedSize =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);

    if (nx <= 0 || ny <= 0
        || velocityX.size() != expectedSize
        || velocityY.size() != expectedSize)
    {
        throw std::invalid_argument("Invalid fluid field for VTK output.");
    }

    std::ofstream out(filename);
    if (!out)
    {
        throw std::runtime_error("Unable to open VTK fluid output file.");
    }

    out << "# vtk DataFile Version 3.0\n"
        << "LBM fluid\n"
        << "ASCII\n"
        << "DATASET STRUCTURED_POINTS\n"
        << "DIMENSIONS " << nx << ' ' << ny << " 1\n"
        << "ORIGIN 0 0 0\n"
        << "SPACING 1 1 1\n"
        << "POINT_DATA " << expectedSize << "\n"
        << "VECTORS velocity double\n";

    for (std::size_t node = 0; node < expectedSize; ++node)
    {
        out << velocityX[node] << ' '
            << velocityY[node] << " 0\n";
    }
}

void writeFluid(const LBMSolver& fluid, const std::string& filename)
{
    writeFluidField(
        fluid.nx(),
        fluid.ny(),
        fluid.ux(),
        fluid.uy(),
        filename
    );
}

void writeParticle(
    const IBMParticle& particle,
    const std::string& filename)
{
    std::ofstream out(filename);
    if (!out)
    {
        throw std::runtime_error("Unable to open VTK particle output file.");
    }

    out << "# vtk DataFile Version 3.0\n"
        << "IBM particle\n"
        << "ASCII\n"
        << "DATASET POLYDATA\n"
        << "POINTS " << particle.markerCount << " double\n";

    for (int marker = 0; marker < particle.markerCount; ++marker)
    {
        out << particle.markerX[marker] << ' '
            << particle.markerY[marker] << " 0\n";
    }

    out << "LINES " << particle.markerCount << ' '
        << 3 * particle.markerCount << '\n';

    for (int marker = 0; marker < particle.markerCount; ++marker)
    {
        out << "2 " << marker << ' '
            << (marker + 1) % particle.markerCount << '\n';
    }
}

} // namespace io
