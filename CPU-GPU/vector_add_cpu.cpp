#include <iostream>
#include <omp.h>

int main() {
    const int SIZE = 11;
    float vectorA[SIZE], vectorB[SIZE], result[SIZE];

    for (int i = 0; i < SIZE; i++) {
        vectorA[i] = i; 
        vectorB[i] = i + 1;
    }

    for (int i = 0; i < SIZE; i++) {
        result[i] = vectorA[i] + vectorB[i];
    }

    for (int i = 0; i < SIZE; i++) {
        std::cout << "C[" << i << "] = " << result[i] << std::endl;
    }

    return 0;
}

