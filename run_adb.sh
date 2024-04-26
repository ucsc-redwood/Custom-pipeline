#!/bin/bash

# Executable name
executable="cpu"
# Path to local executable
local_executable_path="./build/android/armeabi-v7a/release/$executable"

# Push the executable to the Android device
adb push $local_executable_path /data/local/tmp

# Run the executable on the Android device

# Get a random file from the folder
#random_file=$(ls ./images/ | shuf -n 1)

# Run the executable on the Android device with the random file
#adb shell "cd /data/local/tmp && ./$executable ./images/$random_file 6"

# Check if the -v flag is provided
if [[ $1 == "-v" ]]; then
    # Run the executable on the Android device and print the output
    adb shell "cd /data/local/tmp && ./$executable --benchmark_out=bench_output.txt --benchmark_out_format=console"
else
    # Run the executable on the Android device and redirect the output to /dev/null
    adb shell "cd /data/local/tmp && ./$executable --benchmark_out=bench_output.txt --benchmark_out_format=console > /dev/null 2>&1"
fi

# Print the output of the executable
adb shell "cat /data/local/tmp/bench_output.txt"

