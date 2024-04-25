#include "cifar_cuda.h"
#include <fstream>
#include <iostream>
#include <cfloat>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cfloat>

void readDataFromFile(const std::string& filename, float* data, int dataSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return;
    }

    int index = 0;
    float value;
    while (index < dataSize && file >> value) {
        data[index++] = value;
    }

    if (index != dataSize) {
        std::cerr << "Error: Data size mismatch. Expected " << dataSize << " elements, but file contains " << index << "." << std::endl;
        return;
    }

    file.close();
}

__global__ void maxpool2d_kernel(float* input_data, int input_channels, int input_height, int input_width,
                                    int pool_size, int stride, float* output_data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = gridDim.x * blockDim.x;

    if (index < total_output) {
        int output_height = (input_height - pool_size) / stride + 1;
        int output_width = (input_width - pool_size) / stride + 1;

        int n = index / (output_height * output_width); // Channel index
        int h = (index % (output_height * output_width)) / output_width; // Height index
        int w = index % output_width; // Width index

        float max_value = -FLT_MAX;
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int ih = h * stride + ph;
                int iw = w * stride + pw;
                int idx = (n * input_height + ih) * input_width + iw;
                max_value = fmaxf(max_value, input_data[idx]);
            }
        }
        output_data[index] = max_value;
    }
}

void maxpool2d(float* input_data, int input_channels, int input_height, int input_width,
                  int pool_size, int stride, float* output_data, int threadsPerBlock) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    int total_output = input_channels * output_height * output_width;
    int numBlocks = (total_output + threadsPerBlock - 1) / threadsPerBlock;

    maxpool2d_kernel<<<numBlocks, threadsPerBlock>>>(input_data, input_channels, input_height, input_width,
                                                        pool_size, stride, output_data);
    cudaDeviceSynchronize();
}
__global__ void conv2d_kernel(float* input_data, int input_channels, int input_height, int input_width,
                                 float* weight_data, int output_channels, int weight_input_channels, int weight_height, int weight_width,
                                 float* bias_data, int bias_number_of_elements, int stride, int padding,
                                 bool relu, float* output_data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int output_height = (input_height + 2 * padding - weight_height) / stride + 1;
    int output_width = (input_width + 2 * padding - weight_width) / stride + 1;
    int total_output = output_channels * output_height * output_width;

    if (index < total_output) {
        int k = index / (output_height * output_width); // Output channel
        int i = (index % (output_height * output_width)) / output_width; // Output height
        int j = index % output_width; // Output width

        float sum = 0.0;
        for (int c = 0; c < input_channels; ++c) {
            for (int y = 0; y < weight_height; ++y) {
                int in_y = i * stride - padding + y;
                for (int x = 0; x < weight_width; ++x) {
                    int in_x = j * stride - padding + x;
                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                        sum += input_data[(c * input_height + in_y) * input_width + in_x] *
                               weight_data[((k * input_channels + c) * weight_height + y) * weight_width + x];
                    }
                }
            }
        }

        if (bias_data && k < bias_number_of_elements) {
            sum += bias_data[k];
        }
        if (relu && sum < 0) {
            sum = 0.0;
        }
        output_data[index] = sum;
    }
}

void conv2d(float* input_data, int input_channels, int input_height, int input_width,
               float* weight_data, int output_channels, int weight_input_channels, int weight_height, int weight_width,
               float* bias_data, int bias_number_of_elements, int kernel_size, int stride, int padding,
               bool relu, float* output_data, int threadsPerBlock) {
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    int total_output = output_channels * output_height * output_width;
    int numBlocks = (total_output + threadsPerBlock - 1) / threadsPerBlock;

    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(input_data, input_channels, input_height, input_width,
                                                     weight_data, output_channels, weight_input_channels, weight_height, weight_width,
                                                     bias_data, bias_number_of_elements, stride, padding, relu, output_data);
    cudaDeviceSynchronize();
}
__global__ void linearLayer_kernel(float* input_data, float* weights, float* bias, float* output_data,
                                   int input_size, int output_size) {
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index < output_size) {
        float temp = 0;
        for (int j = 0; j < input_size; ++j) {
            temp += input_data[j] * weights[output_index * input_size + j];
        }
        output_data[output_index] = temp + bias[output_index];
    }
}

void linearLayer(float* input_data, float* weights, float* bias, float* output_data,
                 int input_size, int output_size, int threadsPerBlock) {
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    linearLayer_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_data, weights, bias, output_data,
                                                           input_size, output_size);
}
