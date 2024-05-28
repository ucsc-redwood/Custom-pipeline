{
    depfiles_gcc = "main.o: src/main.cpp include/naive_pipeline.hpp include/application.hpp  include/singleton.hpp include/core/VulkanTools.h include/conv2d.hpp  include/app_params.hpp include/maxpool2d.hpp include/linearLayer.hpp  include/utils.hpp\
",
    files = {
        "src/main.cpp"
    },
    values = {
        "/usr/bin/gcc",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-Wall",
            "-Wextra",
            "-O3",
            "-std=c++20",
            "-Iinclude",
            "-isystem",
            "/home/riksharm/.xmake/packages/s/spirv-cross/1.3.268+0/d1511b27c10642198250cec612800bdf/include",
            "-isystem",
            "/home/riksharm/.xmake/packages/g/glm/1.0.1/098b42d4eeff4f23b1f5e66b16572043/include",
            "-isystem",
            "/home/riksharm/.xmake/packages/s/spdlog/v1.13.0/cca5b628022e49229205fe5617e4245e/include",
            "-isystem",
            "/home/riksharm/.xmake/packages/c/cli11/v2.4.1/9bb9e342af314b5a8eb896f12be78a27/include",
            "-DNDEBUG"
        }
    }
}