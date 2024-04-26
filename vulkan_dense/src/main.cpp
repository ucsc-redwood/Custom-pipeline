#include "naive_pipeline.hpp"
#include "app_params.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file_path>" << std::endl;
        return 1;
    }
    std::string filePath = argv[1];
    AppParams params;
    Pipe pipe(params);
    pipe.allocate(filePath);
    pipe.run();
    
}