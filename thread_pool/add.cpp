#include "add.hpp"
#include <fstream>
#include <iostream>

int sumIntegersFromFile(const std::string& filename) {
    std::ifstream file(filename);
    int sum = 0, value;

    while (file >> value) {
        sum += value;
    }

    return sum;
}

