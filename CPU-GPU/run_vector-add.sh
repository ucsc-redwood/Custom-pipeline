#!/bin/bash

# Convert the shader
python3 convert_to_spv.py vector_add.comp

# Build the C++ application with C++17 standard
g++ -std=c++17 vector_add.cpp easyvk.cpp -o my_vector_add_app -lvulkan -I.

# Run the application
./my_vector_add_app

# Delete the generated SPIR-V file and the application binary
rm vector_add.spv
rm my_vector_add_app

