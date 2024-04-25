#include "image_processing.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cfloat>
#include <chrono>

void readDataFromFile(const std::string& filename, float* data, int& dataSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return;
    }
    float value;
    int count = 0;
    while (file >> value) {
        ++count;
    }

    if (count != dataSize) {
        std::cerr << "Data size mismatch. Expected " << dataSize << " elements, but file contains " << count << "." << std::endl;
        return;
    }

    file.clear();
    file.seekg(0, std::ios::beg);

    int index = 0;
    while (file >> value) {
        data[index++] = value;
    }

    file.close();
}

void maxpool2d(float* input_data, int input_channels, int input_height, int input_width,
               int pool_size, int stride, float* output_data) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    int total_iterations = input_channels * output_height * output_width;

    for (int index = 0; index < total_iterations; index++) {
        int c = index / (output_height * output_width);
        int h = (index / output_width) % output_height;
        int w = index % output_width;

        float max_val = -FLT_MAX;
        for (int p = 0; p < pool_size * pool_size; p++) {
            int ph = p / pool_size;
            int pw = p % pool_size;

            int input_h = h * stride + ph;
            int input_w = w * stride + pw;
            if (input_h < input_height && input_w < input_width) {
                int input_index = c * (input_height * input_width) + input_h * input_width + input_w;
                max_val = std::max(max_val, input_data[input_index]);
            }
        }
        int output_index = c * (output_height * output_width) + h * output_width + w;
        output_data[output_index] = max_val;
    }
}

void conv2d(float* input_data, int image_input_channels, int input_height, int input_width,
            float* weight_data, int weight_output_channels, int weight_input_channels, int weight_height, int weight_width,
            float* bias_data, int bias_number_of_elements, int kernel_size, int stride, int padding, 
            bool relu, float* output_data) {
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    for (int index = 0; index < weight_output_channels * output_height * output_width; index++) {
        int out_channel = index / (output_height * output_width);
        int y = (index / output_width) % output_height;
        int x = index % output_width;

        float sum = 0.0f;
        for (int in_channel = 0; in_channel < weight_input_channels; ++in_channel) {
            for (int ky = 0; ky < weight_height; ++ky) {
                int image_y = y * stride + ky - padding;
                for (int kx = 0; kx < weight_width; ++kx) {
                    int image_x = x * stride + kx - padding;
                    if (image_y >= 0 && image_y < input_height && image_x >= 0 && image_x < input_width) {
                        int input_index = (in_channel * input_height + image_y) * input_width + image_x;
                        int weight_index = (((out_channel * weight_input_channels + in_channel) * weight_height + ky) * weight_width + kx);
                        sum += input_data[input_index] * weight_data[weight_index];
                    }
                }
            }
        }
        if (bias_data && out_channel < bias_number_of_elements) {
            sum += bias_data[out_channel];
        }
        if (relu && sum < 0) {
            sum = 0.0f;
        }
        output_data[(out_channel * output_height + y) * output_width + x] = sum;
    }
}

void linearLayer(float* input_data, float* weights, float* bias, float* output_data, int input_size, int output_size) {
    for (int i = 0; i < output_size; ++i) {
        float sum = 0;
        for (int j = 0; j < input_size; ++j) {
            sum += input_data[j] * weights[i * input_size + j];
        }
        sum += bias[i];
        output_data[i] = sum;
    }
}

