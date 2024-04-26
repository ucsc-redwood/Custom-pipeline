#include <iostream>
#include <fstream>
#include <algorithm>
#include <cfloat>

void readDataFromFile(const std::string& filename, float* data, int dataSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return;
    }

    float value;
    int count = 0;
    while (file >> value) {
        if (count < dataSize) {
            data[count++] = value;
        }
    }

    if (count != dataSize) {
        std::cerr << "Data size mismatch. Expected " << dataSize << " elements, but file contains " << count << "." << std::endl;
    }

    file.close();
}
