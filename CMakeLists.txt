cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
project(convolution_hip LANGUAGES CXX HIP)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HIP REQUIRED)

file(GLOB_RECURSE LIB_SRC_FILES src/*.cpp)
add_library(convolution_hip STATIC ${LIB_SRC_FILES})

target_include_directories(convolution_hip PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)

set(GPU_PLATFORM "rocm" CACHE STRING "GPU platform: cuda or rocm")
set(GPU_ARCH "gfx1100" CACHE STRING "GPU architecture")

if (GPU_PLATFORM STREQUAL "cuda")
    add_definitions(-DCMAKE_HIP_COMPILER=nvcc)
    add_definitions(-DCMAKE_HIP_ARCHITECTURES=${GPU_ARCH})
    add_definitions(-DCMAKE_GPU_RUNTIME=CUDA)
    set(CUDA_TARGET_ARCH ${GPU_ARCH})
elseif (GPU_PLATFORM STREQUAL "rocm")
    set(CMAKE_CXX_STANDARD 17)
    add_definitions(-DCMAKE_HIP_COMPILER=hipcc)
    add_definitions(-DCMAKE_HIP_ARCHITECTURES=${GPU_ARCH})
    add_definitions(-DCMAKE_GPU_RUNTIME=ROCM)
    set(HIP_TARGET_ARCH ${GPU_ARCH})
endif()

# target_link_libraries(convolution_hip PRIVATE hip::device /opt/rocm/lib/libMIOpen.so)
target_link_libraries(convolution_hip PRIVATE hip::device)
target_include_directories(convolution_hip PRIVATE ${HIP_INCLUDE_DIRS})
target_compile_definitions(convolution_hip PRIVATE USE_HIP)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE convolution_hip)
