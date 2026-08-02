#include "IBMParticle.hpp"

#include <cmath>

namespace {
constexpr double pi = 3.14159265358979323846;
}

IBMParticle::IBMParticle(double x0, double y0, double r, double rhoP)
    : x(x0), y(y0), radius(r), density(rhoP) {
    mass = density * area();
    inertia = 0.5 * mass * radius * radius;

    markerCount = static_cast<int>(std::round(2.0 * pi * radius));
    ds = 2.0 * pi * radius / static_cast<double>(markerCount);

    markerX.resize(markerCount);
    markerY.resize(markerCount);
    fluidUx.resize(markerCount);
    fluidUy.resize(markerCount);
    markerForceX.resize(markerCount);
    markerForceY.resize(markerCount);

    updateMarkers();
}

void IBMParticle::updateMarkers() {
    for (int k = 0; k < markerCount; ++k) {
        const double angle =
            2.0 * pi * static_cast<double>(k) /
                static_cast<double>(markerCount) +
            theta;

        markerX[k] = x + radius * std::cos(angle);
        markerY[k] = y + radius * std::sin(angle);
    }
}

void IBMParticle::clearLoads() {
    forceX = 0.0;
    forceY = 0.0;
    torque = 0.0;
}

void IBMParticle::addExternalForce(double fx, double fy) {
    forceX += fx;
    forceY += fy;
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

double IBMParticle::area() const noexcept {
    return pi * radius * radius;
}
