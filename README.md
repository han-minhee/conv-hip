# Optimized Convolution Kernels in HIP

## Overview
WIP. Works automatically for AMD/NVIDIA GPUs depending on the system.

## Results

### RX 7900 GRE
```
$ ./build.sh
Detected GPU platform: rocm
Detected GPU architecture: gfx1100
-- Configuring done (0.1s)
-- Generating done (0.0s)
[ 25%] Building CXX object CMakeFiles/conv_hip.dir/src/main.cpp.o
[ 50%] Linking CXX static library libconvolution_hip.a
[ 50%] Built target convolution_hip
[ 75%] Building CXX object CMakeFiles/main.dir/src/main.cpp.o
[100%] Linking CXX executable main
[100%] Built target main
Running CPU convolution...
CPU convolution complete.
 Now running GPU Convolutions...
GPU Convolution 0 (Naive) Time: 376.251 ms
GPU Convolution 1 (Tiling) Time: 319.504 ms
GPU Convolution 2 (Shared Memory) Time: 219.147 ms
GPU Convolution 4 (Winograd) Time: 113.624 ms
GPU Convolution 5 (Winograd with Shared Memory) Time: 72.9817 ms
GPU Convolution 6 (Winograd with Many Optimizations) Time: 30.6645 ms
```

### RTX 4060 Laptop
```
$ ./build.sh
./build.sh: line 4: rocm-smi: command not found
Detected GPU platform: cuda
Detected GPU architecture: sm89
[ 50%] Built target convolution_hip
[100%] Built target main
Running CPU convolution...
CPU convolution complete.
 Now running GPU Convolutions...
GPU Convolution 0 (Naive) Time: 588.517 ms
GPU Convolution 1 (Tiling) Time: 606.302 ms
GPU Convolution 2 (Shared Memory) Time: 551.555 ms
GPU Convolution 4 (Winograd) Time: 324.719 ms
GPU Convolution 5 (Winograd with Shared Memory) Time: 150.01 ms
GPU Convolution 6 (Winograd with Many Optimizations) Time: 108.688 ms
```