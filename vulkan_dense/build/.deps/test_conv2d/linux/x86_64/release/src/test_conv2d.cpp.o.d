{
    depfiles_gcc = "test_conv2d.o: src/test_conv2d.cpp include/conv2d.hpp  include/application.hpp include/singleton.hpp include/core/VulkanTools.h  include/app_params.hpp\
",
    files = {
        "src/test_conv2d.cpp"
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
            "-DNDEBUG"
        }
    }
}