#include "Simulation.hpp"

#include <exception>
#include <iostream>

int main() {
    try {
        Simulation simulation;
        simulation.run(
            Simulation::defaultSteps(),
            Simulation::defaultOutputInterval()
        );
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}
