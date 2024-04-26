#include "image_processing.hpp"
#include "ThreadPool.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <benchmark/benchmark.h>

std::mutex cout_mutex;

void processImageFile(const std::string &filePath)
{
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    using std::chrono::milliseconds;
    using std::chrono::steady_clock;

    std::cout << "Processing file: " << filePath << std::endl;

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

    // Load the image data
    int imageDataSize = 3072;
    float *image_data = new float[imageDataSize];
    readDataFromFile(filePath, image_data, imageDataSize);

    // Load the weights and data for the first convolutional layer
    int weightDataSize = 1728;
    float *weight_data = new float[weightDataSize];
    readDataFromFile("data/features_0_weight.txt", weight_data, weightDataSize);
    int biasDataSize = 64;
    float *bias_data = new float[biasDataSize];
    readDataFromFile("data/features_0_bias.txt", bias_data, biasDataSize);

    // Load the weights and data for the second convolutional layer
    int secondWeightDataSize = 110592;
    float *second_weight_data = new float[secondWeightDataSize];
    readDataFromFile("data/features_3_weight.txt", second_weight_data, secondWeightDataSize);
    int secondBiasDataSize = 192;
    float *second_bias_data = new float[secondBiasDataSize];
    readDataFromFile("data/features_3_bias.txt", second_bias_data, secondBiasDataSize);

    // Load weights and biases for the third convolutional layer
    int thirdWeightDataSize = 663552;
    float *third_weight_data = new float[thirdWeightDataSize];
    readDataFromFile("data/features_6_weight.txt", third_weight_data, thirdWeightDataSize);
    int thirdBiasDataSize = 384;
    float *third_bias_data = new float[thirdBiasDataSize];
    readDataFromFile("data/features_6_bias.txt", third_bias_data, thirdBiasDataSize);

    // Load weights and biases for the fourth convolutional layer
    int fourthWeightDataSize = 884736;
    float *fourth_weight_data = new float[fourthWeightDataSize];
    readDataFromFile("data/features_8_weight.txt", fourth_weight_data, fourthWeightDataSize);
    int fourthBiasDataSize = 256;
    float *fourth_bias_data = new float[fourthBiasDataSize];
    readDataFromFile("data/features_8_bias.txt", fourth_bias_data, fourthBiasDataSize);

    // Load weights and biases for the fifth convolutional layer
    int fifthWeightDataSize = 589824;
    float *fifth_weight_data = new float[fifthWeightDataSize];
    readDataFromFile("data/features_10_weight.txt", fifth_weight_data, fifthWeightDataSize);
    int fifthBiasDataSize = 256;
    float *fifth_bias_data = new float[fifthBiasDataSize];
    readDataFromFile("data/features_10_bias.txt", fifth_bias_data, fifthBiasDataSize);

    // Load the weights and bias for the linear layer
    int linearWeightSize = 40960;
    float *linear_weight_data = new float[linearWeightSize];
    readDataFromFile("data/classifier_weight.txt", linear_weight_data, linearWeightSize);
    int linearBiasSize = 10;
    float *linear_bias_data = new float[linearBiasSize];
    readDataFromFile("data/classifier_bias.txt", linear_bias_data, linearBiasSize);

    // Allocate memory for convolution output
    int conv_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int conv_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    float *conv_output_data = new float[weight_output_channels * conv_output_height * conv_output_width];

    // Call the convolution function
    auto start_conv1 = steady_clock::now();
    conv2d(image_data, image_input_channels, input_height, input_width,
           weight_data, weight_output_channels, weight_input_channels, weight_height, weight_width,
           bias_data, bias_number_of_elements, kernel_size, stride, padding, relu, conv_output_data);
    auto end_conv1 = steady_clock::now();

    // Allocate memory for max pooling output
    int pooled_output_height = (conv_output_height - pool_size) / pool_stride + 1;
    int pooled_output_width = (conv_output_width - pool_size) / pool_stride + 1;
    float *maxpool_output_data = new float[weight_output_channels * pooled_output_height * pooled_output_width];

    // Apply max pooling on the convolution output
    auto start_maxpool1 = steady_clock::now();
    maxpool2d(conv_output_data, weight_output_channels, conv_output_height, conv_output_width, pool_size, pool_stride, maxpool_output_data);
    auto end_maxpool1 = steady_clock::now();

    // Allocate memory for the output of the second convolutional layer
    int second_conv_output_height = pooled_output_height;
    int second_conv_output_width = pooled_output_width;
    float *second_conv_output_data = new float[192 * second_conv_output_height * second_conv_output_width];

    // Call the convolution function for the second layer
    auto start_conv2 = steady_clock::now();
    conv2d(maxpool_output_data, 64, pooled_output_height, pooled_output_width,
           second_weight_data, 192, 64, 3, 3,
           second_bias_data, 192, 3, 1, 1,
           true, second_conv_output_data);
    auto end_conv2 = steady_clock::now();

    // Allocate memory for the output of the second max pooling layer
    int second_pooled_output_height = (second_conv_output_height - pool_size) / pool_stride + 1;
    int second_pooled_output_width = (second_conv_output_width - pool_size) / pool_stride + 1;
    float *second_maxpool_output_data = new float[192 * second_pooled_output_height * second_pooled_output_width];

    // Apply the second max pooling on the second convolution output
    auto start_maxpool2 = steady_clock::now();
    maxpool2d(second_conv_output_data, 192, second_conv_output_height, second_conv_output_width, pool_size, pool_stride, second_maxpool_output_data);
    auto end_maxpool2 = steady_clock::now();

    // Allocate memory for the output of the third convolutional layer
    int third_conv_output_height = second_pooled_output_height;
    int third_conv_output_width = second_pooled_output_width;
    float *third_conv_output_data = new float[384 * third_conv_output_height * third_conv_output_width];

    // Apply the third convolution
    auto start_conv3 = steady_clock::now();
    conv2d(second_maxpool_output_data, 192, second_pooled_output_height, second_pooled_output_width,
           third_weight_data, 384, 192, 3, 3,
           third_bias_data, 384, 3, 1, 1,
           true, third_conv_output_data);
    auto end_conv3 = steady_clock::now();

    // Allocate memory for the output of the fourth convolutional layer
    int fourth_conv_output_channels = 256;
    int fourth_conv_output_height = third_conv_output_height;
    int fourth_conv_output_width = third_conv_output_width;
    float *fourth_conv_output_data = new float[fourth_conv_output_channels * fourth_conv_output_height * fourth_conv_output_width];

    // Call the convolution function for the fourth layer
    auto start_conv4 = steady_clock::now();
    conv2d(third_conv_output_data, 384, third_conv_output_height, third_conv_output_width,
           fourth_weight_data, fourth_conv_output_channels, 384, 3, 3,
           fourth_bias_data, fourth_conv_output_channels, 3, 1, 1,
           true, fourth_conv_output_data);
    auto end_conv4 = steady_clock::now();

    // Allocate memory for the output of the fifth convolutional layer
    int fifth_conv_output_channels = 256;
    int fifth_conv_output_height = fourth_conv_output_height;
    int fifth_conv_output_width = fourth_conv_output_width;
    float *fifth_conv_output_data = new float[fifth_conv_output_channels * fifth_conv_output_height * fifth_conv_output_width];

    // Apply the fifth convolution
    auto start_conv5 = steady_clock::now();
    conv2d(fourth_conv_output_data, 256, fourth_conv_output_height, fourth_conv_output_width,
           fifth_weight_data, fifth_conv_output_channels, 256, 3, 3,
           fifth_bias_data, fifth_conv_output_channels, 3, 1, 1,
           true, fifth_conv_output_data);
    auto end_conv5 = steady_clock::now();

    // Define parameters for the max pooling layer after the fifth convolution
    int pool_size_after_fifth = 2;
    int pool_stride_after_fifth = 2;

    // Calculate the output dimensions for the max pooling layer
    int fifth_pooled_output_height = (fifth_conv_output_height - pool_size_after_fifth) / pool_stride_after_fifth + 1;
    int fifth_pooled_output_width = (fifth_conv_output_width - pool_size_after_fifth) / pool_stride_after_fifth + 1;

    // Allocate memory for the output of the max pooling layer
    float *fifth_maxpool_output_data = new float[fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width];

    // Apply max pooling on the fifth convolution output
    auto start_maxpool3 = steady_clock::now();
    maxpool2d(fifth_conv_output_data, fifth_conv_output_channels, fifth_conv_output_height, fifth_conv_output_width,
              pool_size_after_fifth, pool_stride_after_fifth, fifth_maxpool_output_data);
    auto end_maxpool3 = steady_clock::now();

    // After the third max pooling layer, flatten the output
    int totalElements = fifth_conv_output_channels * fifth_pooled_output_height * fifth_pooled_output_width;
    float *flattened_output = new float[totalElements];

    int index = 0;
    for (int c = 0; c < fifth_conv_output_channels; ++c)
    {
        for (int h = 0; h < fifth_pooled_output_height; ++h)
        {
            for (int w = 0; w < fifth_pooled_output_width; ++w)
            {
                flattened_output[index++] = fifth_maxpool_output_data[(c * fifth_pooled_output_height + h) * fifth_pooled_output_width + w];
            }
        }
    }

    // Define the output size for the linear layer
    int linear_output_size = 10; // Since the bias size is 10
    float *linear_output_data = new float[linear_output_size];

    // Call the linear layer function
    auto start_linear = steady_clock::now();
    linearLayer(flattened_output, linear_weight_data, linear_bias_data, linear_output_data, totalElements, linear_output_size);
    auto end_linear = steady_clock::now();

    // Find the index of the maximum element in the linear layer output
    int max_index = 0;
    float max_value = linear_output_data[0];
    for (int i = 1; i < linear_output_size; ++i)
    {
        if (linear_output_data[i] > max_value)
        {
            max_value = linear_output_data[i];
            max_index = i;
        }
    }

    // Map the index to the corresponding class and print the prediction
    std::cout << "Predicted Image: ";
    switch (max_index)
    {
    case 0:
        std::cout << "airplanes";
        break;
    case 1:
        std::cout << "cars";
        break;
    case 2:
        std::cout << "birds";
        break;
    case 3:
        std::cout << "cats";
        break;
    case 4:
        std::cout << "deer";
        break;
    case 5:
        std::cout << "dogs";
        break;
    case 6:
        std::cout << "frogs";
        break;
    case 7:
        std::cout << "horses";
        break;
    case 8:
        std::cout << "ships";
        break;
    case 9:
        std::cout << "trucks";
        break;
    default:
        std::cout << "Unknown";
        break;
    }
    std::cout << std::endl;

    double total_time_ms = 0;

    // Add each layer's time to the total time
    //  For milliseconds, add directly. For microseconds, divide by 1000 to convert to milliseconds.
    total_time_ms += duration_cast<milliseconds>(end_conv1 - start_conv1).count();
    total_time_ms += duration_cast<microseconds>(end_maxpool1 - start_maxpool1).count() / 1000.0;
    total_time_ms += duration_cast<milliseconds>(end_conv2 - start_conv2).count();
    total_time_ms += duration_cast<microseconds>(end_maxpool2 - start_maxpool2).count() / 1000.0;
    total_time_ms += duration_cast<milliseconds>(end_conv3 - start_conv3).count();
    total_time_ms += duration_cast<milliseconds>(end_conv4 - start_conv4).count();
    total_time_ms += duration_cast<milliseconds>(end_conv5 - start_conv5).count();
    total_time_ms += duration_cast<microseconds>(end_maxpool3 - start_maxpool3).count() / 1000.0;
    total_time_ms += duration_cast<microseconds>(end_linear - start_linear).count() / 1000.0;

    /*
    // Print the times for all convolutional and max pooling layers
    std::cout << "conv1 " << duration_cast<milliseconds>(end_conv1 - start_conv1).count() << "ms "
              << "maxpool1 " << duration_cast<microseconds>(end_maxpool1 - start_maxpool1).count() << "us "
              << "conv2 " << duration_cast<milliseconds>(end_conv2 - start_conv2).count() << "ms "
              << "maxpool2 " << duration_cast<microseconds>(end_maxpool2 - start_maxpool2).count() << "us "
              << "conv3 " << duration_cast<milliseconds>(end_conv3 - start_conv3).count() << "ms "
              << "conv4 " << duration_cast<milliseconds>(end_conv4 - start_conv4).count() << "ms "
              << "conv5 " << duration_cast<milliseconds>(end_conv5 - start_conv5).count() << "ms "
              << "maxpool3 " << duration_cast<microseconds>(end_maxpool3 - start_maxpool3).count() << "us "
              << "linear " << duration_cast<microseconds>(end_linear - start_linear).count() << "us " << std::endl;

         */

    std::cout << "Total time: " << total_time_ms << "ms" << std::endl;

    // Free allocated memory
    delete[] image_data;
    delete[] weight_data;
    delete[] bias_data;
    delete[] conv_output_data;
    delete[] maxpool_output_data;
    delete[] second_weight_data;
    delete[] second_bias_data;
    delete[] second_conv_output_data;
    delete[] second_maxpool_output_data;
    delete[] third_weight_data;
    delete[] third_bias_data;
    delete[] fourth_weight_data;
    delete[] fourth_bias_data;
    delete[] fourth_conv_output_data;
    delete[] fifth_weight_data;
    delete[] fifth_bias_data;
    delete[] fifth_conv_output_data;
    delete[] flattened_output;
}

// Define an array of strings of image paths
std::vector<std::string> fileNames = {
    "images/flattened_automobile_automobile_1.txt",
    "images/flattened_bird_bird_1.txt",
    "images/flattened_deer_deer_1.txt",
    "images/flattened_deer_deer_2.txt",
    "images/flattened_dog_dog_1.txt",
    "images/flattened_horse_horse_1.txt",
    "images/flattened_ship_ship_1.txt",
    "images/flattened_ship_ship_2.txt",
    "images/flattened_truck_truck_1.txt"};

#define MAX_THREADS 8

static void BM_processImageFile(benchmark::State &state)
{
    int numThreads = state.range(0);
    ThreadPool pool(numThreads);
    for (auto _ : state)
    {
        // auto start = std::chrono::high_resolution_clock::now();
        for (const std::string &filename : fileNames)
   git      {
            pool.enqueue([filename](const std::string &fname)
                         {
	    processImageFile(fname);
            std::lock_guard<std::mutex> cout_lock(cout_mutex); },
                         filename);
        }
        pool.wait_complete();
        // auto end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> duration = end - start;
        // state.SetIterationTime(duration.count());
    }
}

BENCHMARK(BM_processImageFile)
    // ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, MAX_THREADS)
    ->Iterations(1)
    ->ArgName("Threads");

BENCHMARK_MAIN();

// int main(int argc, char **argv)
// {
//     if (argc != 3)
//     {
//         std::cerr << "Usage: " << argv[0] << " <image_file_path> <max_number_of_threads>" << std::endl;
//         return 1;
//     }

//     std::string filePath = argv[1];
//     int numberOfThreads = std::stoi(argv[2]);
//     ThreadPool pool(numberOfThreads);

//     pool.enqueue([filePath]()
//                  { processImageFile(filePath); });

//     pool.wait_complete();

//     return 0;
// }
