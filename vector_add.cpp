#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <vector>

const int size = 1024 * 16;

int main(int argc, char* argv[]) {
    auto instance = easyvk::Instance(true);
    auto physicalDevices = instance.physicalDevices();
    auto device = easyvk::Device(instance, physicalDevices.at(0));
    std::cout << "Using device: " << device.properties.deviceName << "\n";

    auto numIters = 1;
    for (int n = 0; n < numIters; n++) {
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
    }

    device.teardown();
    instance.teardown();
    return 0;
}

