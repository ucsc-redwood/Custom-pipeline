VULKAN_SDK = /path/to/vulkan/sdk

CFLAGS = -std=c++11 -I$(VULKAN_SDK)/include
LDFLAGS = -L$(VULKAN_SDK)/lib -lvulkan

all: main

main: main.cpp naive_pipeline.cpp conv2d.cpp utils.cpp
    g++ $(CFLAGS) -o main main.cpp naive_pipeline.cpp conv2d.cpp utils.cpp $(LDFLAGS)

clean:
    rm -f main
