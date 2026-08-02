#pragma once

#include <vector>

class IBMParticle {
public:
    IBMParticle(double x, double y, double radius, double density);

    void updateMarkers();
    void clearLoads();
    void addExternalForce(double fx, double fy);
    void integrate(double dt);

    [[nodiscard]] double area() const noexcept;
    [[nodiscard]] double translationalKineticEnergy() const noexcept;
    [[nodiscard]] double rotationalKineticEnergy() const noexcept;

    double x{};
    double y{};
    double vx{};
    double vy{};
    double ax{};
    double ay{};
    double theta{};
    double omega{};
    double alpha{};
    double radius{};
    double density{};
    double mass{};
    double inertia{};

    double forceX{};
    double forceY{};
    double torque{};

    int markerCount{};
    double ds{};

    std::vector<double> markerX;
    std::vector<double> markerY;
    std::vector<double> fluidUx;
    std::vector<double> fluidUy;
    std::vector<double> markerForceX;
    std::vector<double> markerForceY;
};
