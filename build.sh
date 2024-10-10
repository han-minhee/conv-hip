#!/bin/bash

nvidia_info=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null)
rocm_info=$(rocm-smi --showproductname | grep -m1 -o 'gfx[0-9]\{3,5\}' 2>/dev/null)

if [ ! -z "$nvidia_info" ]; then
    compute_cap=$(echo "$nvidia_info" | awk -F',' '{print $1}')
    arch="sm$(echo $compute_cap | sed 's/\.//')"
    platform="cuda"
elif [ ! -z "$rocm_info" ]; then
    arch="gfx$(echo "$rocm_info" | sed 's/[^0-9]*//g' | sed 's/.$//')"
    platform="rocm"
else
    echo "No compatible GPU found."
    exit 1
fi

echo "Detected GPU platform: $platform"
echo "Detected GPU architecture: $arch"

mkdir -p build
cd build

# Pass the GPU platform and architecture to CMake
cmake .. -DGPU_PLATFORM=$platform -DGPU_ARCH=$arch

cmake --build . --config Debug

# Run the executable if build succeeded
if [ -f ./main ]; then
    ./main
else
    echo "Build failed or 'main' executable not found."
    exit 1
fi
