set_project("Custom-Pipeline")
-- Global settings for debug and release modes
add_rules("mode.debug", "mode.release")

add_requires( "glm")
add_requires("vulkan-headers")

target("vector_add")
set_default(true)
set_plat("android")
set_arch("arm64-v8a")
set_kind("binary")
add_includedirs("include")
add_headerfiles("include/*.hpp", "include/*.h")
add_files("./*.cpp", "include/*.c", "include/*.cpp")
add_packages("vulkan-headers", "glm")
