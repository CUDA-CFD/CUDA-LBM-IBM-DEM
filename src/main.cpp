#include "Simulation.hpp"
#include <exception>
#include <iostream>

int main() {
    try {
        Simulation sim;
        sim.run(12000, 500);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
