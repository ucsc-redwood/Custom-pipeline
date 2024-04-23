#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include "ThreadPool.hpp"
#include "add.hpp"

std::mutex cout_mutex;

int main() {
    std::vector<std::string> fileNames = {
        "input1.txt", "input2.txt", "input3.txt", "input4.txt", "input5.txt",
        "input6.txt", "input7.txt", "input8.txt", "input9.txt", "input10.txt"
    };

    ThreadPool pool(1);

    auto start = std::chrono::high_resolution_clock::now();

    for (const std::string& filename : fileNames) {
        pool.enqueue([filename](const std::string& fname) {
            int sum = sumIntegersFromFile(fname);
            std::lock_guard<std::mutex> cout_lock(cout_mutex);
            std::cout << "Sum of integers in " << fname << ": " << sum << std::endl;
        }, filename);
    }

    pool.wait_complete();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Total time to compute " << fileNames.size() << " tasks: " << duration.count() << " ms\n";

    return 0;
}

