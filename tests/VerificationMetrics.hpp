#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct ArrayErrorMetrics {
    double maxAbsolute = 0.0;
    double l2 = 0.0;
    double relativeL2 = 0.0;
    std::size_t maxIndex = 0;
    double referenceValue = 0.0;
    double candidateValue = 0.0;
};

ArrayErrorMetrics calculateArrayErrorMetrics(
    const std::vector<double>& reference,
    const std::vector<double>& candidate
);

struct VerificationSnapshot {
    int step = 0;
    ArrayErrorMetrics density;
    ArrayErrorMetrics velocityX;
    ArrayErrorMetrics velocityY;
    ArrayErrorMetrics populations;
};

void writeVerificationCsvHeader(const std::string& filename);
void appendVerificationCsv(
    const std::string& filename,
    const VerificationSnapshot& snapshot
);
