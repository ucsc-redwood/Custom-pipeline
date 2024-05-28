#include <iostream>
#include "conv2d.hpp"
#include "app_params.hpp"

int main() {
    // Initialize parameters for Conv2d
    AppParams params;
    params.input_height = 5;
    params.input_width = 5;
    params.weight_input_channels = 1;
    params.weight_output_channels = 1;
    params.kernel_size = 3;
    params.stride = 1;
    params.padding = 1;

    // Create input data (5x5) with a single channel
    float input_data[5][5] = {
        {1, 2, 3, 0, 1},
        {0, 1, 2, 3, 0},
        {1, 2, 1, 0, 1},
        {0, 1, 2, 3, 0},
        {1, 2, 3, 0, 1}
    };

    // Create weight data (3x3) with a single filter
    float weight_data[1][3][3] = {
        {
            {1, 0, -1},
            {1, 0, -1},
            {1, 0, -1}
        }
    };

    // Create bias data
    float bias_data[1] = {0};

    // Allocate memory for the output data
    int output_height = (params.input_height + 2 * params.padding - params.kernel_size) / params.stride + 1;
    int output_width = (params.input_width + 2 * params.padding - params.kernel_size) / params.stride + 1;
    float* output_data = new float[output_height * output_width];

    // Create Conv2d object and set parameters
    Conv2d conv2d;
    conv2d.compute_constant(params);

    // Run the convolution operation
    conv2d.run(1, 0, &input_data[0][0], &weight_data[0][0][0], bias_data, output_data, params.weight_output_channels, nullptr, nullptr, nullptr, nullptr, 1);

    // Print the output data
    std::cout << "Conv2d Output:" << std::endl;
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            std::cout << output_data[i * output_width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    delete[] output_data;

    return 0;
}

