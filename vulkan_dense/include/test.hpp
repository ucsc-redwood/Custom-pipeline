#pragma once
#include <string>
#include "application.hpp"
#include "conv2d.hpp"
#include "maxpool2d.hpp"
#include "linearLayer.hpp"
#include "app_params.hpp"
#include "utils.hpp"
#include <iostream>
using namespace std;

class Pipe : public ApplicationBase {
 public:
  Pipe(AppParams param) : ApplicationBase() {
    params_ = param;
  }
  void run();

  void result();

  void allocate(std::string file_path);
  ~Pipe();

  protected:
  AppParams params_;

  // Essential data memory
  float* image_data;
  float* weight_data;
  float* bias_data;
  
  float* second_weight_data;
  float* second_bias_data;

  float* third_weight_data;
  float* third_bias_data;

  float* fourth_weight_data;
  float* fourth_bias_data;

  float* fifth_weight_data;
  float* fifth_bias_data;

  float* linear_weight_data;
  float* linear_bias_data;

  float* conv_output_data;
  float* maxpool_output_data;

  float* second_conv_output_data;
  float* second_maxpool_output_data;

  float* third_conv_output_data;
    
  float* fourth_conv_output_data;

  float* fifth_conv_output_data;
  float* fifth_maxpool_output_data;

  float* flattened_output;
  float* linear_output_data;



  VkBuffer image_data_buffer;
    VkBuffer weight_data_buffer;
    VkBuffer bias_data_buffer;

    VkBuffer second_weight_data_buffer;
    VkBuffer second_bias_data_buffer;

    VkBuffer third_weight_data_buffer;
    VkBuffer third_bias_data_buffer;

    VkBuffer fourth_weight_data_buffer;
    VkBuffer fourth_bias_data_buffer;

    VkBuffer fifth_weight_data_buffer;
    VkBuffer fifth_bias_data_buffer;

    VkBuffer linear_weight_data_buffer;
    VkBuffer linear_bias_data_buffer;

    VkBuffer conv_output_data_buffer;
    VkBuffer maxpool_output_data_buffer;

    VkBuffer second_conv_output_data_buffer;
    VkBuffer second_maxpool_output_data_buffer;

    VkBuffer third_conv_output_data_buffer;

    VkBuffer fourth_conv_output_data_buffer;

    VkBuffer fifth_conv_output_data_buffer;
    VkBuffer fifth_maxpool_output_data_buffer;

    VkBuffer flattened_output_buffer;
    VkBuffer linear_output_data_buffer;

  VkDeviceMemory image_data_memory;
    VkDeviceMemory weight_data_memory;
    VkDeviceMemory bias_data_memory;

    VkDeviceMemory second_weight_data_memory;
    VkDeviceMemory second_bias_data_memory;

    VkDeviceMemory third_weight_data_memory;
    VkDeviceMemory third_bias_data_memory;

    VkDeviceMemory fourth_weight_data_memory;
    VkDeviceMemory fourth_bias_data_memory;

    VkDeviceMemory fifth_weight_data_memory;
    VkDeviceMemory fifth_bias_data_memory;

    VkDeviceMemory linear_weight_data_memory;
    VkDeviceMemory linear_bias_data_memory;

    VkDeviceMemory conv_output_data_memory;
    VkDeviceMemory maxpool_output_data_memory;

    VkDeviceMemory second_conv_output_data_memory;
    VkDeviceMemory second_maxpool_output_data_memory;

    VkDeviceMemory third_conv_output_data_memory;

    VkDeviceMemory fourth_conv_output_data_memory;

    VkDeviceMemory fifth_conv_output_data_memory;
    VkDeviceMemory fifth_maxpool_output_data_memory;

    VkDeviceMemory flattened_output_memory;
    VkDeviceMemory linear_output_data_memory;
};


void Pipe::allocate(std::string file_path) {
    int image_data_size = 3072;
    int weight_data_size = 1728;
    int bias_data_size = 64;
    int second_weight_data_size = 110592;
    int second_bias_data_size = 192;
    int third_weight_data_size = 663552;
    int third_bias_data_size = 384;
    int fourth_weight_data_size = 663552;
    int fourth_bias_data_size = 256;
    int fifth_weight_data_size = 589824;
    int fifth_bias_data_size = 256;
    int linear_weight_size = 40960;
    int linear_bias_size = 10;
    int conv_output_height = (params_.input_height + 2 * params_.padding - params_.kernel_size) / params_.stride + 1;
    int conv_output_width = (params_.input_width + 2 * params_.padding - params_.kernel_size) / params_.stride + 1;
    int pooled_output_height = (conv_output_height - params_.pool_size) / params_.pool_stride + 1;
    int pooled_output_width = (conv_output_width - params_.pool_size) / params_.pool_stride + 1;
    int second_conv_output_height = pooled_output_height;
    int second_conv_output_width = pooled_output_width;
    int second_pooled_output_height = (second_conv_output_height - params_.pool_size) / params_.pool_stride + 1;
    int second_pooled_output_width = (second_conv_output_width - params_.pool_size) / params_.pool_stride + 1;
    int third_conv_output_height = second_pooled_output_height;
    int third_conv_output_width = second_pooled_output_width;
    int fourth_conv_output_channels = 256;
    int fourth_conv_output_height = third_conv_output_height;
    int fourth_conv_output_width = third_conv_output_width;
    int fifth_conv_output_channels = 256;
    int fifth_conv_output_height = fourth_conv_output_height;
    int fifth_conv_output_width = fourth_conv_output_width;
    int pool_size_after_fifth = 2;
    int pool_stride_after_fifth = 2;
    int fifth_pooled_output_height = (fifth_conv_output_height - pool_size_after_fifth) / pool_stride_after_fifth + 1;
    int fifth_pooled_output_width = (fifth_conv_output_width - pool_size_after_fifth) / pool_stride_after_fifth + 1;
    int total_elements = fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width;
    int linear_output_size = 10;
    int max_index = 0;
    void *mapped;
    // --- Essentials ---
    //u_points.resize(n);
    //u_morton_keys.resize(n);
    //u_unique_morton_keys.resize(n);
    // map and initialize to zero
    create_shared_empty_storage_buffer(image_data_size*sizeof(float), &image_data_buffer, &image_data_memory, &mapped);
    image_data = static_cast< float*>(mapped);
    readDataFromFile(file_path, image_data, image_data_size);

    create_shared_empty_storage_buffer(weight_data_size*sizeof(float), &weight_data_buffer, &weight_data_memory, &mapped);
    weight_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/features_0_weight.txt", weight_data, weight_data_size);

    create_shared_empty_storage_buffer(bias_data_size*sizeof(float), &bias_data_buffer, &bias_data_memory, &mapped);
    bias_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/features_0_bias.txt", bias_data, bias_data_size);

    create_shared_empty_storage_buffer(second_weight_data_size*sizeof(float), &second_weight_data_buffer, &second_weight_data_memory, &mapped);
    second_weight_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/features_3_weight.txt", second_weight_data, second_weight_data_size);

    create_shared_empty_storage_buffer(second_bias_data_size*sizeof(float), &second_bias_data_buffer, &second_bias_data_memory, &mapped);
    second_bias_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/features_3_bias.txt", second_bias_data, second_bias_data_size);

    create_shared_empty_storage_buffer(third_weight_data_size*sizeof(float), &third_weight_data_buffer, &third_weight_data_memory, &mapped);
    third_weight_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/features_6_weight.txt", third_weight_data, third_weight_data_size);

    create_shared_empty_storage_buffer(third_bias_data_size*sizeof(float), &third_bias_data_buffer, &third_bias_data_memory, &mapped);
    third_bias_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/features_6_bias.txt", third_bias_data, third_bias_data_size);

    create_shared_empty_storage_buffer(fourth_weight_data_size*sizeof(float), &fourth_weight_data_buffer, &fourth_weight_data_memory, &mapped);
    fourth_weight_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/features_8_weight.txt", fourth_weight_data, fourth_weight_data_size);

    create_shared_empty_storage_buffer(fourth_bias_data_size*sizeof(float), &fourth_bias_data_buffer, &fourth_bias_data_memory, &mapped);
    fourth_bias_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/features_8_bias.txt", fourth_bias_data, fourth_bias_data_size);

    create_shared_empty_storage_buffer(fifth_weight_data_size*sizeof(float), &fifth_weight_data_buffer, &fifth_weight_data_memory, &mapped);
    fifth_weight_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/features_10_weight.txt", fifth_weight_data, fifth_weight_data_size);

    create_shared_empty_storage_buffer(fifth_bias_data_size*sizeof(float), &fifth_bias_data_buffer, &fifth_bias_data_memory, &mapped);
    fifth_bias_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/features_10_bias.txt", fifth_bias_data, fifth_bias_data_size);

    create_shared_empty_storage_buffer(linear_weight_size*sizeof(float), &linear_weight_data_buffer, &linear_weight_data_memory, &mapped);
    linear_weight_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/classifier_weight.txt", linear_weight_data, linear_weight_size);

    create_shared_empty_storage_buffer(linear_bias_size*sizeof(float), &linear_bias_data_buffer, &linear_bias_data_memory, &mapped);
    linear_bias_data = static_cast< float*>(mapped);
    readDataFromFile("../../../../data/classifier_bias.txt", linear_bias_data, linear_bias_size);

    create_shared_empty_storage_buffer(params_.weight_output_channels * conv_output_height * conv_output_width * sizeof(float), &conv_output_data_buffer, &conv_output_data_memory, &mapped);
    conv_output_data = static_cast< float*>(mapped);
    std::fill_n(conv_output_data, params_.weight_output_channels * conv_output_height * conv_output_width, 0.0f);

    create_shared_empty_storage_buffer(params_.weight_output_channels * pooled_output_height * pooled_output_width * sizeof(float), &maxpool_output_data_buffer, &maxpool_output_data_memory, &mapped);
    maxpool_output_data = static_cast< float*>(mapped);
    std::fill_n(maxpool_output_data, params_.weight_output_channels * pooled_output_height * pooled_output_width, 0.0f);

    create_shared_empty_storage_buffer(192 * second_conv_output_height * second_conv_output_width * sizeof(float), &second_conv_output_data_buffer, &second_conv_output_data_memory, &mapped);
    second_conv_output_data = static_cast< float*>(mapped);
    std::fill_n(second_conv_output_data, 192 * second_conv_output_height * second_conv_output_width, 0.0f);

    create_shared_empty_storage_buffer(192 * second_pooled_output_height * second_pooled_output_width * sizeof(float), &second_maxpool_output_data_buffer, &second_maxpool_output_data_memory, &mapped);
    second_maxpool_output_data = static_cast< float*>(mapped);
    std::fill_n(second_maxpool_output_data, 192 * second_pooled_output_height * second_pooled_output_width, 0.0f);

    create_shared_empty_storage_buffer(384 * third_conv_output_height * third_conv_output_width * sizeof(float), &third_conv_output_data_buffer, &third_conv_output_data_memory, &mapped);
    third_conv_output_data = static_cast< float*>(mapped);
    std::fill_n(third_conv_output_data, 384 * third_conv_output_height * third_conv_output_width, 0.0f);

    create_shared_empty_storage_buffer(fourth_conv_output_channels * fourth_conv_output_height * fourth_conv_output_width * sizeof(float), &fourth_conv_output_data_buffer, &fourth_conv_output_data_memory, &mapped);
    fourth_conv_output_data = static_cast< float*>(mapped);
    std::fill_n(fourth_conv_output_data, fourth_conv_output_channels * fourth_conv_output_height * fourth_conv_output_width, 0.0f);

    create_shared_empty_storage_buffer(fifth_conv_output_channels * fifth_conv_output_height * fifth_conv_output_width * sizeof(float), &fifth_conv_output_data_buffer, &fifth_conv_output_data_memory, &mapped);
    fifth_conv_output_data = static_cast< float*>(mapped);
    std::fill_n(fifth_conv_output_data, fifth_conv_output_channels * fifth_conv_output_height * fifth_conv_output_width, 0.0f);

    create_shared_empty_storage_buffer(fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width * sizeof(float), &fifth_maxpool_output_data_buffer, &fifth_maxpool_output_data_memory, &mapped);
    fifth_maxpool_output_data = static_cast< float*>(mapped);
    std::fill_n(fifth_maxpool_output_data, fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width, 0.0f);

    create_shared_empty_storage_buffer(total_elements * sizeof(float), &flattened_output_buffer, &flattened_output_memory, &mapped);
    flattened_output = static_cast< float*>(mapped);
    for (int c = 0; c < fifth_conv_output_channels; c++) {
        for (int h = 0; h < fifth_pooled_output_height; h++) {
            for (int w = 0; w < fifth_pooled_output_width; w++) {
                int flat_index = c * (fifth_pooled_output_height * fifth_pooled_output_width) + h * fifth_pooled_output_width + w;
                int original_index = (c * fifth_pooled_output_height + h) * fifth_pooled_output_width + w;
                flattened_output[flat_index] = fifth_maxpool_output_data[original_index];
            }
        }
    }

    create_shared_empty_storage_buffer(linear_output_size * sizeof(float), &linear_output_data_buffer, &linear_output_data_memory, &mapped);
    linear_output_data = static_cast< float*>(mapped);
    std::fill_n(linear_output_data, linear_output_size, 0.0f);
};


void Pipe::result() {
    int linear_output_size = 10;
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
}

void Pipe::run(){
    int conv_output_height = (params_.input_height + 2 * params_.padding - params_.kernel_size) / params_.stride + 1;
    int conv_output_width = (params_.input_width + 2 * params_.padding - params_.kernel_size) / params_.stride + 1;

    // --- Convolution 1 ---
    Conv2d conv1 = Conv2d();
    conv1.compute_constant(params_);
    conv1.run(1, 0, image_data, weight_data, bias_data, conv_output_data, params_.weight_output_channels, image_data_buffer, weight_data_buffer, bias_data_buffer, conv_output_data_buffer, 1);
    
    // --- output 1 ---
    std::cout << "Conv1 Output:" << std::endl;
    for (int i = 0; i < params_.weight_output_channels * conv_output_height * conv_output_width; ++i) {
        std::cout << conv_output_data[i] << " ";
    }
    std::cout << std::endl;
    
    // --- Maxpool 1 ---
    Maxpool2d maxpool1 = Maxpool2d();
    maxpool1.compute_constant(params_.weight_output_channels, conv_output_height, conv_output_width, params_.pool_size, params_.pool_stride);
    maxpool1.run(1, 0, conv_output_data, maxpool_output_data, conv_output_data_buffer, maxpool_output_data_buffer, 1);
    
    // --- output 2 ---
    std::cout << "Maxpool1 Output:" << std::endl;
    for (int i = 0; i < params_.weight_output_channels * conv_output_height * conv_output_width; ++i) {
        std::cout << maxpool_output_data[i] << " ";
    }
    std::cout << std::endl;
    
    // --- Convolution 2 ---
    Conv2d conv2 = Conv2d();
    AppParams second_conv_params = params_;
    second_conv_params.weight_input_channels = params_.weight_output_channels;
    second_conv_params.weight_output_channels = 192;
    second_conv_params.kernel_size = 3;
    second_conv_params.stride = 1;
    second_conv_params.padding = 1;
    conv2.compute_constant(second_conv_params);
    conv2.run(1, 0, maxpool_output_data, second_weight_data, second_bias_data, second_conv_output_data, 192, maxpool_output_data_buffer, second_weight_data_buffer, second_bias_data_buffer, second_conv_output_data_buffer, 1);
    
    // --- output 3 ---
    std::cout << "Conv2 Output:" << std::endl;
    for (int i = 0; i < 192 * second_conv_output_height * second_conv_output_width; ++i) {
        std::cout << second_conv_output_data[i] << " ";
    }
    std::cout << std::endl;
    
    // --- Second Maxpool ---
    Maxpool2d maxpool2 = Maxpool2d();
    maxpool2.compute_constant(192, second_conv_output_height, second_conv_output_width, params_.pool_size, params_.pool_stride);
    maxpool2.run(1, 0, second_conv_output_data, second_maxpool_output_data, second_conv_output_data_buffer, second_maxpool_output_data_buffer, 1);
    
    // --- output 4 ---
    std::cout << "Maxpool2 Output:" << std::endl;
    for (int i = 0; i < 192 * second_pooled_output_height * second_pooled_output_width; ++i) {
        std::cout << second_maxpool_output_data[i] << " ";
    }
    std::cout << std::endl;
        
    // --- Third Convolution ---
    Conv2d conv3 = Conv2d();
    AppParams third_conv_params = params_;
    third_conv_params.weight_input_channels = 192;
    third_conv_params.weight_output_channels = 384;
    third_conv_params.kernel_size = 3;
    third_conv_params.stride = 1;
    third_conv_params.padding = 1;
    conv3.compute_constant(third_conv_params);
    conv3.run(1, 0, second_maxpool_output_data, third_weight_data, third_bias_data, third_conv_output_data, 384, second_maxpool_output_data_buffer, third_weight_data_buffer, third_bias_data_buffer, third_conv_output_data_buffer, 1);

    std::cout << "Conv3 Output:" << std::endl;
    for (int i = 0; i < 384 * third_conv_output_height * third_conv_output_width; ++i) {
        std::cout << third_conv_output_data[i] << " ";
    }
    std::cout << std::endl;

    // --- Fourth Convolution ---
    Conv2d conv4 = Conv2d();
    AppParams fourth_conv_params = params_;
    fourth_conv_params.weight_input_channels = 384;
    fourth_conv_params.weight_output_channels = 256;
    fourth_conv_params.kernel_size = 3;
    fourth_conv_params.stride = 1;
    fourth_conv_params.padding = 1;
    conv4.compute_constant(fourth_conv_params);
    conv4.run(1, 0, third_conv_output_data, fourth_weight_data, fourth_bias_data, fourth_conv_output_data, 256, third_conv_output_data_buffer, fourth_weight_data_buffer, fourth_bias_data_buffer, fourth_conv_output_data_buffer, 1);

    std::cout << "Conv4 Output:" << std::endl;
    for (int i = 0; i < 256 * fourth_conv_output_height * fourth_conv_output_width; ++i) {
        std::cout << fourth_conv_output_data[i] << " ";
    }
    std::cout << std::endl;

    int fifth_conv_output_height = (params_.input_height + 2 * params_.padding - params_.kernel_size) / params_.stride + 1;
    int fifth_conv_output_width = (params_.input_width + 2 * params_.padding - params_.kernel_size) / params_.stride + 1;    
    // --- Fifth Convolution ---
    Conv2d conv5 = Conv2d();
    AppParams fifth_conv_params = params_;
    fifth_conv_params.weight_input_channels = 256;
    fifth_conv_params.weight_output_channels = 256;
    fifth_conv_params.kernel_size = 3;
    fifth_conv_params.stride = 1;
    fifth_conv_params.padding = 1;
    conv5.compute_constant(fifth_conv_params);
    conv5.run(1, 0, fourth_conv_output_data, fifth_weight_data, fifth_bias_data, fifth_conv_output_data, 256, fourth_conv_output_data_buffer, fifth_weight_data_buffer, fifth_bias_data_buffer, fifth_conv_output_data_buffer, 1);

    std::cout << "Conv5 Output:" << std::endl;
    for (int i = 0; i < 256 * fifth_conv_output_height * fifth_conv_output_width; ++i) {
        std::cout << fifth_conv_output_data[i] << " ";
    }
    std::cout << std::endl;

    int fifth_conv_output_channels = 256;
    int fifth_pooled_output_height = (fifth_conv_output_height - 2) / 2 + 1;
    int fifth_pooled_output_width = (fifth_conv_output_width - 2) / 2 + 1;
    // --- Maxpool after Fifth Convolution ---
    Maxpool2d maxpool3 = Maxpool2d();
    maxpool3.compute_constant(256, fifth_conv_output_height, fifth_conv_output_width, 2, 2);
    maxpool3.run(1, 0, fifth_conv_output_data, fifth_maxpool_output_data, fifth_conv_output_data_buffer, fifth_maxpool_output_data_buffer, 1);    

    std::cout << "Maxpool3 Output:" << std::endl;
    for (int i = 0; i < 256 * fifth_pooled_output_height * fifth_pooled_output_width; ++i) {
        std::cout << fifth_maxpool_output_data[i] << " ";
    }
    std::cout << std::endl;

    // Flattening the output of the third max pooling layer
    int totalElements = fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width;
    float* flattened_output = new float[totalElements];
    int index = 0;
    for (int c = 0; c < fifth_conv_output_channels; ++c) {
        for (int h = 0; h < fifth_pooled_output_height; ++h) {
            for (int w = 0; w < fifth_pooled_output_width; ++w) {
                flattened_output[index++] = fifth_maxpool_output_data[(c * fifth_pooled_output_height + h) * fifth_pooled_output_width + w];
            }
        }
    }

    std::cout << "Flattened Output:" << std::endl;
    for (int i = 0; i < totalElements; ++i) {
        std::cout << flattened_output[i] << " ";
    }
    std::cout << std::endl;

    LinearLayer linearLayer = LinearLayer();
    linearLayer.compute_constant(totalElements, 10);
    linearLayer.run(1, 0, flattened_output, linear_weight_data, linear_bias_data, linear_output_data, flattened_output_buffer, linear_weight_data_buffer, linear_bias_data_buffer, linear_output_data_buffer, 1);

    std::cout << "Linear Layer Output:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << linear_output_data[i] << " ";
    }
    std::cout << std::endl;

    result();
}

Pipe::~Pipe() {
  // --- Essentials ---
    vkUnmapMemory(singleton.device, image_data_memory);
    vkDestroyBuffer(singleton.device, image_data_buffer, nullptr);
    vkFreeMemory(singleton.device, image_data_memory, nullptr);

    vkUnmapMemory(singleton.device, weight_data_memory);
    vkDestroyBuffer(singleton.device, weight_data_buffer, nullptr);
    vkFreeMemory(singleton.device, weight_data_memory, nullptr);

    vkUnmapMemory(singleton.device, bias_data_memory);
    vkDestroyBuffer(singleton.device, bias_data_buffer, nullptr);
    vkFreeMemory(singleton.device, bias_data_memory, nullptr);

    vkUnmapMemory(singleton.device, second_weight_data_memory);
    vkDestroyBuffer(singleton.device, second_weight_data_buffer, nullptr);
    vkFreeMemory(singleton.device, second_weight_data_memory, nullptr);

    vkUnmapMemory(singleton.device, second_bias_data_memory);
    vkDestroyBuffer(singleton.device, second_bias_data_buffer, nullptr);
    vkFreeMemory(singleton.device, second_bias_data_memory, nullptr);

    vkUnmapMemory(singleton.device, third_weight_data_memory);
    vkDestroyBuffer(singleton.device, third_weight_data_buffer, nullptr);
    vkFreeMemory(singleton.device, third_weight_data_memory, nullptr);

    vkUnmapMemory(singleton.device, third_bias_data_memory);
    vkDestroyBuffer(singleton.device, third_bias_data_buffer, nullptr);
    vkFreeMemory(singleton.device, third_bias_data_memory, nullptr);

    vkUnmapMemory(singleton.device, fourth_weight_data_memory);
    vkDestroyBuffer(singleton.device, fourth_weight_data_buffer, nullptr);
    vkFreeMemory(singleton.device, fourth_weight_data_memory, nullptr);

    vkUnmapMemory(singleton.device, fourth_bias_data_memory);
    vkDestroyBuffer(singleton.device, fourth_bias_data_buffer, nullptr);
    vkFreeMemory(singleton.device, fourth_bias_data_memory, nullptr);

    vkUnmapMemory(singleton.device, fifth_weight_data_memory);
    vkDestroyBuffer(singleton.device, fifth_weight_data_buffer, nullptr);
    vkFreeMemory(singleton.device, fifth_weight_data_memory, nullptr);

    vkUnmapMemory(singleton.device, fifth_bias_data_memory);
    vkDestroyBuffer(singleton.device, fifth_bias_data_buffer, nullptr);
    vkFreeMemory(singleton.device, fifth_bias_data_memory, nullptr);

    vkUnmapMemory(singleton.device, linear_weight_data_memory);
    vkDestroyBuffer(singleton.device, linear_weight_data_buffer, nullptr);
    vkFreeMemory(singleton.device, linear_weight_data_memory, nullptr);

    vkUnmapMemory(singleton.device, linear_bias_data_memory);
    vkDestroyBuffer(singleton.device, linear_bias_data_buffer, nullptr);
    vkFreeMemory(singleton.device, linear_bias_data_memory, nullptr);

    vkUnmapMemory(singleton.device, conv_output_data_memory);
    vkDestroyBuffer(singleton.device, conv_output_data_buffer, nullptr);
    vkFreeMemory(singleton.device, conv_output_data_memory, nullptr);

    vkUnmapMemory(singleton.device, maxpool_output_data_memory);
    vkDestroyBuffer(singleton.device, maxpool_output_data_buffer, nullptr);
    vkFreeMemory(singleton.device, maxpool_output_data_memory, nullptr);

    vkUnmapMemory(singleton.device, second_conv_output_data_memory);
    vkDestroyBuffer(singleton.device, second_conv_output_data_buffer, nullptr);
    vkFreeMemory(singleton.device, second_conv_output_data_memory, nullptr);

    vkUnmapMemory(singleton.device, second_maxpool_output_data_memory);
    vkDestroyBuffer(singleton.device, second_maxpool_output_data_buffer, nullptr);
    vkFreeMemory(singleton.device, second_maxpool_output_data_memory, nullptr);

    vkUnmapMemory(singleton.device, third_conv_output_data_memory);
    vkDestroyBuffer(singleton.device, third_conv_output_data_buffer, nullptr);
    vkFreeMemory(singleton.device, third_conv_output_data_memory, nullptr);

    vkUnmapMemory(singleton.device, fourth_conv_output_data_memory);
    vkDestroyBuffer(singleton.device, fourth_conv_output_data_buffer, nullptr);
    vkFreeMemory(singleton.device, fourth_conv_output_data_memory, nullptr);

    vkUnmapMemory(singleton.device, fifth_conv_output_data_memory);
    vkDestroyBuffer(singleton.device, fifth_conv_output_data_buffer, nullptr);
    vkFreeMemory(singleton.device, fifth_conv_output_data_memory, nullptr);

    vkUnmapMemory(singleton.device, fifth_maxpool_output_data_memory);
    vkDestroyBuffer(singleton.device, fifth_maxpool_output_data_buffer, nullptr);
    vkFreeMemory(singleton.device, fifth_maxpool_output_data_memory, nullptr);

    vkUnmapMemory(singleton.device, flattened_output_memory);
    vkDestroyBuffer(singleton.device, flattened_output_buffer, nullptr);
    vkFreeMemory(singleton.device, flattened_output_memory, nullptr);

    vkUnmapMemory(singleton.device, linear_output_data_memory);
    vkDestroyBuffer(singleton.device, linear_output_data_buffer, nullptr);
    vkFreeMemory(singleton.device, linear_output_data_memory, nullptr);

    // --- Clean up ---
}
