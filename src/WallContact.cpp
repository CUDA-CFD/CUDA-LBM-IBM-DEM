#include "WallContact.hpp"

#include <algorithm>

namespace {

double compressiveForce(
    double overlap,
    double normalVelocityIntoWall,
    double stiffness,
    double damping
) {
    if (overlap <= 0.0) {
        return 0.0;
    }

    // Damping opposes motion into the wall. The clamp prevents an adhesive
    // force during separation.
    return std::max(
        0.0,
        stiffness * overlap + damping * normalVelocityIntoWall
    );
}

} // namespace

WallContactLoads computeWallContact(
    const IBMParticle& particle,
    double width,
    double height,
    double normalStiffness,
    double normalDamping
) {
    WallContactLoads loads;

    // Bottom wall: inward normal is +y, and -vy is the approach speed.
    const double bottomOverlap = particle.radius - particle.y;
    const double bottomForce = compressiveForce(
        bottomOverlap,
        -particle.vy,
        normalStiffness,
        normalDamping
    );
    loads.forceY += bottomForce;

    // Top wall: inward normal is -y, and +vy is the approach speed.
    const double topOverlap = particle.y + particle.radius - height;
    const double topForce = compressiveForce(
        topOverlap,
        particle.vy,
        normalStiffness,
        normalDamping
    );
    loads.forceY -= topForce;

    // Left wall: inward normal is +x, and -vx is the approach speed.
    const double leftOverlap = particle.radius - particle.x;
    const double leftForce = compressiveForce(
        leftOverlap,
        -particle.vx,
        normalStiffness,
        normalDamping
    );
    loads.forceX += leftForce;

    // Right wall: inward normal is -x, and +vx is the approach speed.
    const double rightOverlap = particle.x + particle.radius - width;
    const double rightForce = compressiveForce(
        rightOverlap,
        particle.vx,
        normalStiffness,
        normalDamping
    );
    loads.forceX -= rightForce;

    loads.maximumOverlap = std::max(
        {0.0, bottomOverlap, topOverlap, leftOverlap, rightOverlap}
    );
    loads.active = loads.maximumOverlap > 0.0;

    return loads;
}
