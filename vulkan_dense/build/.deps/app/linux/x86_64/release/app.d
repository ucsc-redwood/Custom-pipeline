{
    files = {
        "build/.objs/app/linux/x86_64/release/src/main.cpp.o"
    },
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-L/home/riksharm/.xmake/packages/s/spirv-cross/1.3.268+0/d1511b27c10642198250cec612800bdf/lib",
            "-s",
            "-lspirv-cross-c",
            "-lspirv-cross-cpp",
            "-lspirv-cross-reflect",
            "-lspirv-cross-msl",
            "-lspirv-cross-util",
            "-lspirv-cross-hlsl",
            "-lspirv-cross-glsl",
            "-lspirv-cross-core",
            "-lvulkan",
            "-lpthread"
        }
    }
}