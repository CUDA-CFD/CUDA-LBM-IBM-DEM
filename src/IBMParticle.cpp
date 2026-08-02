#include "IBMParticle.hpp"
#include <cmath>

IBMParticle::IBMParticle(double x0, double y0, double r, double rhoP)
    : x(x0), y(y0), radius(r), density(rhoP) {
    constexpr double pi = 3.14159265358979323846;
    mass = density * pi * radius * radius;
    inertia = 0.5 * mass * radius * radius;
    markerCount = static_cast<int>(std::round(2.0 * pi * radius));
    ds = 2.0 * pi * radius / markerCount;
    markerX.resize(markerCount);
    markerY.resize(markerCount);
    fluidUx.resize(markerCount);
    fluidUy.resize(markerCount);
    markerForceX.resize(markerCount);
    markerForceY.resize(markerCount);
    updateMarkers();
}

void IBMParticle::updateMarkers() {
    constexpr double pi = 3.14159265358979323846;
    for (int k = 0; k < markerCount; ++k) {
        const double a = 2.0 * pi * k / markerCount + theta;
        markerX[k] = x + radius * std::cos(a);
        markerY[k] = y + radius * std::sin(a);
    }
}

void IBMParticle::clearLoads() {
    forceX = forceY = torque = 0.0;
}

void IBMParticle::addExternalForce(double fx, double fy) {
    forceX += fx;
    forceY += fy;
}

void IBMParticle::addWallContact(double xmin, double xmax, double ymin, double ymax,
                                 double kn, double cn) {
    double overlap = radius - (x - xmin);
    if (overlap > 0.0) forceX += std::max(0.0, kn * overlap - cn * vx);

    overlap = radius - (xmax - x);
    if (overlap > 0.0) forceX -= std::max(0.0, kn * overlap + cn * vx);

    overlap = radius - (y - ymin);
    if (overlap > 0.0) forceY += std::max(0.0, kn * overlap - cn * vy);

    overlap = radius - (ymax - y);
    if (overlap > 0.0) forceY -= std::max(0.0, kn * overlap + cn * vy);
}

void IBMParticle::integrate(double dt) {
    vx += forceX / mass * dt;
    vy += forceY / mass * dt;
    omega += torque / inertia * dt;
    x += vx * dt;
    y += vy * dt;
    theta += omega * dt;
    updateMarkers();
}
