#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

#define CONV_3_TILE_H 8
#define CONV_3_TILE_W 8

__global__ void conv_3(
    // data
    float *input, float *weight, float *bias, float *output,

    // attributes
    int64_t group, int64_t padH, int64_t padW, int64_t strideH, int64_t strideW, int64_t dilationH, int64_t dilationW,

    // input shape
    int64_t N, int64_t inC, int64_t inH, int64_t inW,

    // weight shape (if bias exists, it has size of M)
    int64_t M, int64_t C_per_group, int64_t kH, int64_t kW,

    // output shapes
    int64_t outH, int64_t outW,

    // precomputed shared memory dimensions
    int64_t shared_input_h, int64_t shared_input_w, int threads_per_block, int total_elements, int num_blocks)
{
}
