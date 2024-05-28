import torch
import torch.nn as nn

# Parameters similar to C++ test
in_channels = 3
out_channels = 16
kernel_size = 3
stride = 1
padding = 1
input_width = 16
input_height = 16

# Create a Conv2d layer
conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

# Example input tensor [batch_size, in_channels, height, width]
input_tensor = torch.randn(1, in_channels, input_height, input_width)

# Apply the convolution operation
output_tensor = conv(input_tensor)

# Print the output shape to confirm correct dimensionality
print(output_tensor.shape)

