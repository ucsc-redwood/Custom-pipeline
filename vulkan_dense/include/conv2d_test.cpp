#include <vulkan/vulkan.h>
#include "naive_pipeline.hpp"
#include "conv2d.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>

// Function to initialize Vulkan
void initVulkan(VkInstance& instance, VkDevice& device) {
    // Create Vulkan instance
    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance!");
    }

    // Create Vulkan device
    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    result = vkCreateDevice(instance, &deviceCreateInfo, nullptr, &device);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan device!");
    }
}

int main() {
    VkInstance instance;
    VkDevice device;
    initVulkan(instance, device);

    // Define simple test parameters
    AppParams test_params;
    test_params.weight_input_channels = 1;
    test_params.weight_output_channels = 1;
    test_params.input_height = 9;
    test_params.input_width = 9;
    test_params.kernel_size = 3;
    test_params.stride = 1;
    test_params.padding = 0;
    test_params.relu = false;
    test_params.pool_size = 0; // Not used in this test
    test_params.pool_stride = 0; // Not used in this test

    // Create a Pipe instance with the test parameters
    Pipe test_pipeline(test_params, device);

    // Define test data
    std::vector<float> input_data = {
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        9, 8, 7, 6, 5, 4, 3, 2, 1,
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        9, 8, 7, 6, 5, 4, 3, 2, 1,
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        9, 8, 7, 6, 5, 4, 3, 2, 1,
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        9, 8, 7, 6, 5, 4, 3, 2, 1,
        1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    std::vector<float> weight_data = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };
    std::vector<float> bias_data = {0};
    std::vector<float> output_data(49, 0); // Output size: 7x7

    // Allocate and map buffers
    test_pipeline.allocate();
    test_pipeline.map_memory();

    // Copy input data to the buffers
    test_pipeline.copy_to_buffer(input_data.data(), test_pipeline.image_data_buffer, input_data.size() * sizeof(float));
    test_pipeline.copy_to_buffer(weight_data.data(), test_pipeline.weight_data_buffer, weight_data.size() * sizeof(float));
    test_pipeline.copy_to_buffer(bias_data.data(), test_pipeline.bias_data_buffer, bias_data.size() * sizeof(float));

    // Run the pipeline
    Conv2d conv2d(device);
    conv2d.compute_constant(test_params);
    conv2d.run(1, 0, input_data.data(), weight_data.data(), bias_data.data(), output_data.data(), 1,
                test_pipeline.image_data_buffer, test_pipeline.weight_data_buffer,
                test_pipeline.bias_data_buffer, test_pipeline.conv_output_data_buffer, 1);

    // Copy output data from the buffer
    test_pipeline.copy_from_buffer(output_data.data(), test_pipeline.conv_output_data_buffer, output_data.size() * sizeof(float));

    // Print the output data
    std::cout << "Convolution Output:" << std::endl;
    for (int i = 0; i < 49; ++i) {
        std::cout << output_data[i] << " ";
        if ((i + 1) % 7 == 0) {
            std::cout << std::endl;
        }
    }

    // Cleanup
    test_pipeline.unmap_memory();
    test_pipeline.cleanup();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}
