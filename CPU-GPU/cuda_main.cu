#include "cifar_cuda.h"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <vector>

void processImageCuda(const std::string& filePath, int threadsPerBlock) {

    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    using std::chrono::microseconds;
    using std::chrono::steady_clock;

    int imageDataSize = 3072;
    float* h_image_data = new float[imageDataSize];
    readDataFromFile(filePath, h_image_data, imageDataSize);

    float* d_image_data;
    cudaMalloc(&d_image_data, imageDataSize * sizeof(float));
    cudaMemcpy(d_image_data, h_image_data, imageDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize parameters
    int image_input_channels = 3;
    int input_height = 32;
    int input_width = 32;
    int weight_output_channels = 64;
    int weight_input_channels = 3;
    int weight_height = 3;
    int weight_width = 3;
    int bias_number_of_elements = 64;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int pool_size = 2;
    int pool_stride = 2;
    bool relu = true;

    // Load the weights and data for the first convolutional layer
    int weightDataSize = 1728;
    float* h_weight_data = new float[weightDataSize];
    readDataFromFile("data/features_0_weight.txt", h_weight_data, weightDataSize);

    int biasDataSize = 64;
    float* h_bias_data = new float[biasDataSize];
    readDataFromFile("data/features_0_bias.txt", h_bias_data, biasDataSize);

    float* d_weight_data;
    cudaMalloc(&d_weight_data, weightDataSize * sizeof(float));
    cudaMemcpy(d_weight_data, h_weight_data, weightDataSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_bias_data;
    cudaMalloc(&d_bias_data, biasDataSize * sizeof(float));
    cudaMemcpy(d_bias_data, h_bias_data, biasDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Load the weights and data for the second convolutional layer
    int secondWeightDataSize = 110592;
    float* h_second_weight_data = new float[secondWeightDataSize];
    readDataFromFile("data/features_3_weight.txt", h_second_weight_data, secondWeightDataSize);

    int secondBiasDataSize = 192;
    float* h_second_bias_data = new float[secondBiasDataSize];
    readDataFromFile("data/features_3_bias.txt", h_second_bias_data, secondBiasDataSize);

    float* d_second_weight_data;
    cudaMalloc(&d_second_weight_data, secondWeightDataSize * sizeof(float));
    cudaMemcpy(d_second_weight_data, h_second_weight_data, secondWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_second_bias_data;
    cudaMalloc(&d_second_bias_data, secondBiasDataSize * sizeof(float));
    cudaMemcpy(d_second_bias_data, h_second_bias_data, secondBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Load the weights and data for the third convolutional layer
    int thirdWeightDataSize = 663552;
    float* third_h_weight_data = new float[thirdWeightDataSize];
    readDataFromFile("data/features_6_weight.txt", third_h_weight_data, thirdWeightDataSize);

    int thirdBiasDataSize = 384;
    float* third_h_bias_data = new float[thirdBiasDataSize];
    readDataFromFile("data/features_6_bias.txt", third_h_bias_data, thirdBiasDataSize);

    float* d_third_weight_data;
    cudaMalloc(&d_third_weight_data, thirdWeightDataSize * sizeof(float));
    cudaMemcpy(d_third_weight_data, third_h_weight_data, thirdWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_third_bias_data;
    cudaMalloc(&d_third_bias_data, thirdBiasDataSize * sizeof(float));
    cudaMemcpy(d_third_bias_data, third_h_bias_data, thirdBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Load the weights and data for the fourth convolutional layer
    int fourthWeightDataSize = 884736;
    float* fourth_h_weight_data = new float[fourthWeightDataSize];
    readDataFromFile("data/features_8_weight.txt", fourth_h_weight_data, fourthWeightDataSize);

    int fourthBiasDataSize = 256;
    float* fourth_h_bias_data = new float[fourthBiasDataSize];
    readDataFromFile("data/features_8_bias.txt", fourth_h_bias_data, fourthBiasDataSize);

    float* d_fourth_weight_data;
    cudaMalloc(&d_fourth_weight_data, fourthWeightDataSize * sizeof(float));
    cudaMemcpy(d_fourth_weight_data, fourth_h_weight_data, fourthWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_fourth_bias_data;
    cudaMalloc(&d_fourth_bias_data, fourthBiasDataSize * sizeof(float));
    cudaMemcpy(d_fourth_bias_data, fourth_h_bias_data, fourthBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Load the weights and data for the fifth convolutional layer
    int fifthWeightDataSize = 589824;
    float* fifth_h_weight_data = new float[fifthWeightDataSize];
    readDataFromFile("data/features_10_weight.txt", fifth_h_weight_data, fifthWeightDataSize);

    int fifthBiasDataSize = 256;
    float* fifth_h_bias_data = new float[fifthBiasDataSize];
    readDataFromFile("data/features_10_bias.txt", fifth_h_bias_data, fifthBiasDataSize);

    float* d_fifth_weight_data;
    cudaMalloc(&d_fifth_weight_data, fifthWeightDataSize * sizeof(float));
    cudaMemcpy(d_fifth_weight_data, fifth_h_weight_data, fifthWeightDataSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_fifth_bias_data;
    cudaMalloc(&d_fifth_bias_data, fifthBiasDataSize * sizeof(float));
    cudaMemcpy(d_fifth_bias_data, fifth_h_bias_data, fifthBiasDataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Load the weights and bias for the linear layer
    int linearWeightSize = 40960;
    float* linear_h_weight_data = new float[linearWeightSize];
    readDataFromFile("data/classifier_weight.txt", linear_h_weight_data, linearWeightSize);

    int linearBiasSize = 10;
    float* linear_h_bias_data = new float[linearBiasSize];
    readDataFromFile("data/classifier_bias.txt", linear_h_bias_data, linearBiasSize);

    float* d_linear_weight_data;
    cudaMalloc(&d_linear_weight_data, linearWeightSize * sizeof(float));
    cudaMemcpy(d_linear_weight_data, linear_h_weight_data, linearWeightSize * sizeof(float), cudaMemcpyHostToDevice);

    float* d_linear_bias_data;
    cudaMalloc(&d_linear_bias_data, linearBiasSize * sizeof(float));
    cudaMemcpy(d_linear_bias_data, linear_h_bias_data, linearBiasSize * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for convolution output on the device
    int conv_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int conv_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    float* d_conv_output_data;
    cudaMalloc(&d_conv_output_data, weight_output_channels * conv_output_height * conv_output_width * sizeof(float));

    auto start_conv1 = std::chrono::steady_clock::now();
    conv2d(d_image_data, image_input_channels, input_height, input_width,
                                       d_weight_data, weight_output_channels, weight_input_channels, weight_height, weight_width,
                                       d_bias_data, bias_number_of_elements, kernel_size, stride, padding, relu, d_conv_output_data, threadsPerBlock );

    cudaDeviceSynchronize();
    auto end_conv1 = std::chrono::steady_clock::now();

    // Allocate memory for max pooling output on the device
    int pooled_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    int pooled_output_width = (conv_output_width - pool_size) / pool_stride + 1;
    float* d_maxpool_output_data;
    cudaMalloc(&d_maxpool_output_data, weight_output_channels * pooled_output_height * pooled_output_width * sizeof(float));
    
    // Call the max pooling function
    auto start_maxpool1 = std::chrono::steady_clock::now();
    maxpool2d(d_conv_output_data, weight_output_channels, conv_output_height, conv_output_width, pool_size, pool_stride, d_maxpool_output_data, threadsPerBlock);
    auto end_maxpool1 = std::chrono::steady_clock::now();

    // Allocate memory for the output of the second convolutional layer on the device
    int second_conv_output_height = pooled_output_height;
    int second_conv_output_width = pooled_output_width;
    float* d_second_conv_output_data;
    cudaMalloc(&d_second_conv_output_data, 192 * second_conv_output_height * second_conv_output_width * sizeof(float));
    
    // Call the convolution function for the second layer
    auto start_conv2 = std::chrono::steady_clock::now(); 
    conv2d(d_maxpool_output_data, 64, pooled_output_height, pooled_output_width,
                                               d_second_weight_data, 192, 64, 3, 3,
                                               d_second_bias_data, 192, 3, 1, 1,
                                               true, d_second_conv_output_data, threadsPerBlock);
    cudaDeviceSynchronize(); // Ensure all kernel executions are complete before proceeding
    auto end_conv2 = std::chrono::steady_clock::now();

    // Copy the result back to the host
    float* second_conv_output_data = new float[192 * second_conv_output_height * second_conv_output_width];
    cudaMemcpy(second_conv_output_data, d_second_conv_output_data, 192 * second_conv_output_height * second_conv_output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Allocate memory for the output of the second max pooling layer on the device
    int second_pooled_output_height = (second_conv_output_height - pool_size) / pool_stride + 1;
    int second_pooled_output_width = (second_conv_output_width - pool_size) / pool_stride + 1;
    float* d_second_maxpool_output_data;
    cudaMalloc(&d_second_maxpool_output_data, 192 * second_pooled_output_height * second_pooled_output_width * sizeof(float));
    
    // Apply the second max pooling on the second convolution output
    auto start_maxpool2 = std::chrono::steady_clock::now(); 
    maxpool2d(d_second_conv_output_data, 192, second_conv_output_height, second_conv_output_width, pool_size, pool_stride, d_second_maxpool_output_data, threadsPerBlock);
    cudaDeviceSynchronize();
    auto end_maxpool2 = std::chrono::steady_clock::now();

    // Copy the result back to the host
    float* second_maxpool_output_data = new float[192 * second_pooled_output_height * second_pooled_output_width];
    cudaMemcpy(second_maxpool_output_data, d_second_maxpool_output_data, 192 * second_pooled_output_height * second_pooled_output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Allocate memory for the output of the third convolutional layer on the device
    int third_conv_output_height = second_pooled_output_height;
    int third_conv_output_width = second_pooled_output_width;
    float* d_third_conv_output_data;
    cudaMalloc(&d_third_conv_output_data, 384 * third_conv_output_height * third_conv_output_width * sizeof(float));
    
    // Apply the third convolution
    auto start_conv3 = std::chrono::steady_clock::now();
    conv2d(d_second_maxpool_output_data, 192, second_pooled_output_height, second_pooled_output_width,
                                               d_third_weight_data, 384, 192, 3, 3,
                                               d_third_bias_data, 384, 3, 1, 1,
                                               true, d_third_conv_output_data, threadsPerBlock);
    cudaDeviceSynchronize();
    auto end_conv3 = std::chrono::steady_clock::now();
    
    // Copy the result back to the host
    float* third_conv_output_data = new float[384 * third_conv_output_height * third_conv_output_width];
    cudaMemcpy(third_conv_output_data, d_third_conv_output_data, 384 * third_conv_output_height * third_conv_output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Allocate memory for the output of the fourth convolutional layer on the device
    int fourth_conv_output_channels = 256;
    int fourth_conv_output_height = third_conv_output_height;
    int fourth_conv_output_width = third_conv_output_width;
    float* d_fourth_conv_output_data;
    cudaMalloc(&d_fourth_conv_output_data, fourth_conv_output_channels * fourth_conv_output_height * fourth_conv_output_width * sizeof(float));
    
    // Call the convolution function for the fourth layer
    auto start_conv4 = std::chrono::steady_clock::now();
    conv2d(d_third_conv_output_data, 384, third_conv_output_height, third_conv_output_width,
                                               d_fourth_weight_data, fourth_conv_output_channels, 384, 3, 3,
                                               d_fourth_bias_data, fourth_conv_output_channels, 3, 1, 1,
                                               true, d_fourth_conv_output_data, threadsPerBlock);
    cudaDeviceSynchronize();
    auto end_conv4 = std::chrono::steady_clock::now();
    
    // Copy the result back to the host
    float* fourth_conv_output_data = new float[fourth_conv_output_channels * fourth_conv_output_height * fourth_conv_output_width];
    cudaMemcpy(fourth_conv_output_data, d_fourth_conv_output_data, fourth_conv_output_channels * fourth_conv_output_height * fourth_conv_output_width * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Allocate memory for the output of the fifth convolutional layer on the device
    int fifth_conv_output_channels = 256;
    int fifth_conv_output_height = fourth_conv_output_height;
    int fifth_conv_output_width = fourth_conv_output_width;
    float* d_fifth_conv_output_data;
    cudaMalloc(&d_fifth_conv_output_data, fifth_conv_output_channels * fifth_conv_output_height * fifth_conv_output_width * sizeof(float));
    
    // Call the convolution function for the fifth layer
    auto start_conv5 = std::chrono::steady_clock::now(); 
    conv2d(d_fourth_conv_output_data, 256, fourth_conv_output_height, fourth_conv_output_width,
                                               d_fifth_weight_data, fifth_conv_output_channels, 256, 3, 3,
                                               d_fifth_bias_data, fifth_conv_output_channels, 3, 1, 1,
                                               true, d_fifth_conv_output_data, threadsPerBlock);
    cudaDeviceSynchronize(); // Ensure all kernel executions are complete before proceeding
    auto end_conv5 = std::chrono::steady_clock::now();
    
    // Copy the result back to the host
    float* fifth_conv_output_data = new float[fifth_conv_output_channels * fifth_conv_output_height * fifth_conv_output_width];
    cudaMemcpy(fifth_conv_output_data, d_fifth_conv_output_data, fifth_conv_output_channels * fifth_conv_output_height * fifth_conv_output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // Define parameters for the max pooling layer after the fifth convolution on the device
    int pool_size_after_fifth = 2;
    int pool_stride_after_fifth = 2;
    
    // Calculate the output dimensions for the max pooling layer
    int fifth_pooled_output_height = (fifth_conv_output_height - pool_size_after_fifth) / pool_stride_after_fifth + 1;
    int fifth_pooled_output_width = (fifth_conv_output_width - pool_size_after_fifth) / pool_stride_after_fifth + 1;
    
    // Allocate memory for the output of the max pooling layer on the device
    float* d_fifth_maxpool_output_data;
    cudaMalloc(&d_fifth_maxpool_output_data, fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width * sizeof(float));
   
    // Call the max pooling function for the fifth layer
    auto start_maxpool3 = std::chrono::steady_clock::now();
    maxpool2d(d_fifth_conv_output_data, fifth_conv_output_channels, fifth_conv_output_height, fifth_conv_output_width,
                                                  pool_size_after_fifth, pool_stride_after_fifth, d_fifth_maxpool_output_data, threadsPerBlock);
    cudaDeviceSynchronize(); // Ensure all kernel executions are complete before proceeding
    auto end_maxpool3 = std::chrono::steady_clock::now();
    
    // Copy the result back to the host
    float* fifth_maxpool_output_data = new float[fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width];
    cudaMemcpy(fifth_maxpool_output_data, d_fifth_maxpool_output_data, fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // After the third max pooling layer, flatten the output
    int totalElements = fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width;
    float* d_flattened_output;
    cudaMalloc(&d_flattened_output, totalElements * sizeof(float));
    float* flattened_output = new float[totalElements];
    for (int c = 0; c < fifth_conv_output_channels; c++) {
        for (int h = 0; h < fifth_pooled_output_height; h++) {
            for (int w = 0; w < fifth_pooled_output_width; w++) {
                int flat_index = c * (fifth_pooled_output_height * fifth_pooled_output_width) + h * fifth_pooled_output_width + w;
                int original_index = (c * fifth_pooled_output_height + h) * fifth_pooled_output_width + w;
                flattened_output[flat_index] = fifth_maxpool_output_data[original_index];
            }
        }
    }
    cudaMemcpy(d_flattened_output, flattened_output, totalElements * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate memory for the output of the linear layer on the device
    int linear_output_size = 10;
    float* d_linear_output_data;
    cudaMalloc(&d_linear_output_data, linear_output_size * sizeof(float));
  
    // Call the linear layer function kernel
    auto start_linear = std::chrono::steady_clock::now();
    linearLayer(d_flattened_output, d_linear_weight_data, d_linear_bias_data, d_linear_output_data, totalElements, linear_output_size, threadsPerBlock);
    cudaDeviceSynchronize(); // Ensure all kernel executions are complete before proceeding
    auto end_linear = std::chrono::steady_clock::now();
    
    // Copy the result back to the host
    float* linear_output_data = new float[linear_output_size];
    cudaMemcpy(linear_output_data, d_linear_output_data, linear_output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Find the index of the maximum element in the linear layer output on the host
    int max_index = 0;
    float max_value = linear_output_data[0];
    for (int i = 1; i < linear_output_size; ++i) {
        if (linear_output_data[i] > max_value) {
            max_value = linear_output_data[i];
            max_index = i;
        }
    }
    
    std::cout << "Predicted Image: ";
    switch (max_index) {
        case 0: std::cout << "airplanes"; break;
        case 1: std::cout << "cars"; break;
        case 2: std::cout << "birds"; break;
        case 3: std::cout << "cats"; break;
        case 4: std::cout << "deer"; break;
        case 5: std::cout << "dogs"; break;
        case 6: std::cout << "frogs"; break;
        case 7: std::cout << "horses"; break;
        case 8: std::cout << "ships"; break;
        case 9: std::cout << "trucks"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;

    double total_time_ms = 0.0;
    total_time_ms += duration_cast<milliseconds>(end_conv1 - start_conv1).count();
    total_time_ms += duration_cast<microseconds>(end_maxpool1 - start_maxpool1).count() / 1000.0;
    total_time_ms += duration_cast<milliseconds>(end_conv2 - start_conv2).count();
    total_time_ms += duration_cast<microseconds>(end_maxpool2 - start_maxpool2).count() / 1000.0;
    total_time_ms += duration_cast<milliseconds>(end_conv3 - start_conv3).count();
    total_time_ms += duration_cast<milliseconds>(end_conv4 - start_conv4).count();
    total_time_ms += duration_cast<milliseconds>(end_conv5 - start_conv5).count();
    total_time_ms += duration_cast<microseconds>(end_maxpool3 - start_maxpool3).count() / 1000.0;
    total_time_ms += duration_cast<microseconds>(end_linear - start_linear).count() / 1000.0;
    
    std::cout << "Total time: " << total_time_ms << " ms" << std::endl;

    // Free device memory
    cudaFree(d_image_data);
    cudaFree(d_weight_data);
    cudaFree(d_bias_data);
    cudaFree(d_second_weight_data);
    cudaFree(d_second_bias_data);
    cudaFree(d_third_weight_data);
    cudaFree(d_third_bias_data);
    cudaFree(d_fourth_weight_data);
    cudaFree(d_fourth_bias_data);
    cudaFree(d_fifth_weight_data);
    cudaFree(d_fifth_bias_data);
    cudaFree(d_linear_weight_data);
    cudaFree(d_linear_bias_data);
    cudaFree(d_conv_output_data);
    cudaFree(d_maxpool_output_data);
    cudaFree(d_second_conv_output_data);
    cudaFree(d_second_maxpool_output_data);
    cudaFree(d_third_conv_output_data);
    cudaFree(d_fourth_conv_output_data);
    cudaFree(d_fifth_conv_output_data);
    cudaFree(d_fifth_maxpool_output_data);
    cudaFree(d_flattened_output);
    cudaFree(d_linear_output_data);

    // Free host memory
    delete[] h_image_data;
    delete[] h_weight_data;
    delete[] h_bias_data;
    delete[] h_second_weight_data;
    delete[] h_second_bias_data;
    delete[] third_h_weight_data;
    delete[] third_h_bias_data;
    delete[] fourth_h_weight_data;
    delete[] fourth_h_bias_data;
    delete[] fifth_h_weight_data;
    delete[] fifth_h_bias_data;
    delete[] linear_h_weight_data;
    delete[] linear_h_bias_data;
    delete[] second_conv_output_data;
    delete[] second_maxpool_output_data;
    delete[] third_conv_output_data;
    delete[] fourth_conv_output_data;
    delete[] fifth_conv_output_data;
    delete[] fifth_maxpool_output_data;
    delete[] flattened_output;
    delete[] linear_output_data;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_file_path> <threads_per_block>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];
    int threadsPerBlock = std::stoi(argv[2]);

    processImageCuda(filePath, threadsPerBlock);
    return 0;
}