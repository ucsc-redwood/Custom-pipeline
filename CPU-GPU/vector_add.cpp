#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <thread> // Include the thread library

const int size = 1024 * 16;

// Define a function to perform GPU computation
void gpu_computation() {
    auto instance = easyvk::Instance(true);
    auto physicalDevices = instance.physicalDevices();
    auto device = easyvk::Device(instance, physicalDevices.at(0));
    std::cout << "Using device: " << device.properties.deviceName << " for GPU computation\n";

    auto a = easyvk::Buffer(device, size, sizeof(float));
    auto b = easyvk::Buffer(device, size, sizeof(float));
    auto c = easyvk::Buffer(device, size, sizeof(float));

    for (int i = 0; i < size; i++) {
        a.store<float>(i, static_cast<float>(i));
        b.store<float>(i, static_cast<float>(i + 1));
    }
    c.clear();
    std::vector<easyvk::Buffer> bufs = {a, b, c};

    const char* testFile = "vector_add.spv";
    auto program = easyvk::Program(device, testFile, bufs);

    program.setWorkgroups(size);
    program.setWorkgroupSize(256);

    program.initialize("main");

    program.run();

    for (int i = 0; i < size; i++) {
        assert(c.load<float>(i) == a.load<float>(i) + b.load<float>(i));
    }

    for (int i = 0; i < size; i++) {
        std::cout << "C[" << i << "] = " << c.load<float>(i) << std::endl;
        if (i >= 10) {
            break;
        }
    }

    program.teardown();
    a.teardown();
    b.teardown();
    c.teardown();

    device.teardown();
    instance.teardown();
}

// Define a function to perform CPU computation
void cpu_computation() {
    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> c(size);

    for (int i = 0; i < size; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i + 1);
    }

    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }

    for (int i = 0; i < size; i++) {
        std::cout << "C[" << i << "] = " << c[i] << std::endl;
        if (i >= 10) {
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    // easyvk misses join ~ c api - 
    // Launch GPU computation in a separate thread
    std::thread gpu_thread(gpu_computation);

    // Launch CPU computation in the main thread
    cpu_computation();

    // Wait for GPU thread to finish
    gpu_thread.join();

    return 0;
}
