#pragma once
#include <vector>

class IBMParticle {
public:
    IBMParticle(double x, double y, double radius, double density);
    void updateMarkers();
    void clearLoads();
    void addExternalForce(double fx, double fy);
    void addWallContact(double xmin, double xmax, double ymin, double ymax,
                        double kn, double cn);
    void integrate(double dt);

    double x{}, y{}, vx{}, vy{}, theta{}, omega{}, radius{}, density{}, mass{}, inertia{};
    double forceX{}, forceY{}, torque{};
    int markerCount{};
    double ds{};
    std::vector<double> markerX, markerY;
    std::vector<double> fluidUx, fluidUy;
    std::vector<double> markerForceX, markerForceY;
};
