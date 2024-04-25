{
    values = {
        "/usr/bin/gcc",
        {
            "-m64",
            "-fvisibility=hidden",
            "-Os",
            "-Iinclude",
            "-isystem",
            "/home/riksharm/.xmake/packages/g/glm/1.0.1/098b42d4eeff4f23b1f5e66b16572043/include",
            "-DNDEBUG"
        }
    },
    files = {
        "include/volk.c"
    },
    depfiles_gcc = "volk.o: include/volk.c include/volk.h\
"
}