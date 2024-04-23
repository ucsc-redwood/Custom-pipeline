#ifndef IMAGE_PROCESSING_HPP
#define IMAGE_PROCESSING_HPP

#include <string>

void readDataFromFile(const std::string& filename, float* data, int& dataSize);
void maxpool2d(float* input_data, int input_channels, int input_height, int input_width, int pool_size, int stride, float* output_data);
void conv2d(float* input_data, int image_input_channels, int input_height, int input_width, float* weight_data, int weight_output_channels, int weight_input_channels, int weight_height, int weight_width, float* bias_data, int bias_number_of_elements, int kernel_size, int stride, int padding, bool relu, float* output_data);
void linearLayer(float* input_data, float* weights, float* bias, float* output_data, int input_size, int output_size);

#endif // IMAGE_PROCESSING_HPP

