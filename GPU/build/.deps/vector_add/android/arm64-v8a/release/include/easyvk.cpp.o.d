{
    values = {
        "/usr/bin/gcc",
        {
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-Os",
            "-Iinclude",
            "-isystem",
            "/home/riksharm/.xmake/packages/g/glm/1.0.1/098b42d4eeff4f23b1f5e66b16572043/include",
            "-DNDEBUG"
        }
    },
    files = {
        "include/easyvk.cpp"
    },
    depfiles_gcc = "easyvk.o: include/easyvk.cpp include/easyvk.h include/volk.h\
"
}