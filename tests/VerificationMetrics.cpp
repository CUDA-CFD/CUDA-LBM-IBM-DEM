#include "VerificationMetrics.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

ArrayErrorMetrics calculateArrayErrorMetrics(
    const std::vector<double>& reference,
    const std::vector<double>& candidate)
{
    if (reference.size() != candidate.size()) {
        throw std::invalid_argument(
            "Array size mismatch in CPU-GPU verification."
        );
    }

    ArrayErrorMetrics metrics;
    double squaredErrorSum = 0.0;
    double squaredReferenceSum = 0.0;

    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double difference = candidate[index] - reference[index];
        const double absoluteDifference = std::abs(difference);

        squaredErrorSum += difference * difference;
        squaredReferenceSum += reference[index] * reference[index];

        if (absoluteDifference > metrics.maxAbsolute) {
            metrics.maxAbsolute = absoluteDifference;
            metrics.maxIndex = index;
            metrics.referenceValue = reference[index];
            metrics.candidateValue = candidate[index];
        }
    }

    metrics.l2 = std::sqrt(squaredErrorSum);
    metrics.relativeL2 =
        squaredReferenceSum > 0.0
        ? metrics.l2 / std::sqrt(squaredReferenceSum)
        : metrics.l2;

    return metrics;
}

void writeVerificationCsvHeader(const std::string& filename)
{
    const std::filesystem::path path(filename);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream output(filename, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Unable to create GPU verification CSV: " + filename
        );
    }

    output
        << "step,"
        << "rho_max,rho_l2,rho_relative_l2,"
        << "ux_max,ux_l2,ux_relative_l2,"
        << "uy_max,uy_l2,uy_relative_l2,"
        << "f_max,f_l2,f_relative_l2\n";
}

void appendVerificationCsv(
    const std::string& filename,
    const VerificationSnapshot& snapshot)
{
    std::ofstream output(filename, std::ios::app);
    if (!output) {
        throw std::runtime_error(
            "Unable to append GPU verification CSV: " + filename
        );
    }

    output.precision(17);
    output
        << snapshot.step << ','
        << snapshot.density.maxAbsolute << ','
        << snapshot.density.l2 << ','
        << snapshot.density.relativeL2 << ','
        << snapshot.velocityX.maxAbsolute << ','
        << snapshot.velocityX.l2 << ','
        << snapshot.velocityX.relativeL2 << ','
        << snapshot.velocityY.maxAbsolute << ','
        << snapshot.velocityY.l2 << ','
        << snapshot.velocityY.relativeL2 << ','
        << snapshot.populations.maxAbsolute << ','
        << snapshot.populations.l2 << ','
        << snapshot.populations.relativeL2 << '\n';
}
