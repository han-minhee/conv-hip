#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

#define CONV_1_TILE_H 8
#define CONV_1_TILE_W 8

__global__ void conv_1(
    // data
    float *input, float *weight, float *bias, float *output,

    // attributes
    int64_t group, int64_t padH, int64_t padW, int64_t strideH, int64_t strideW, int64_t dilationH, int64_t dilationW,

    // input shape
    int64_t N, int64_t inC, int64_t inH, int64_t inW,

    // weight shape (if bias exists, it has size of M)
    int64_t M, int64_t C_per_group, int64_t kH, int64_t kW,

    // output shapes are already computed in the host code
    int64_t outH, int64_t outW)
{
    // Step 1: Determine tile position and thread position within tile
    int h_tile = blockIdx.y * CONV_1_TILE_H;
    int w_tile = blockIdx.x * CONV_1_TILE_W;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Step 2: Calculate batch and output channel indices based on block index
    int n = blockIdx.z / M;
    int m_global = blockIdx.z % M;

    // Step 3: Determine the group and local channel index within the group
    int g = m_global / (M / group);
    int m = m_global % (M / group);

    // Step 4: Initialize output accumulator
    float value = 0.0f;

    // Step 5: Perform convolution by iterating over input channels and kernel dimensions
    for (int c = 0; c < C_per_group; ++c)
    {
        for (int kh = 0; kh < kH; ++kh)
        {
            for (int kw = 0; kw < kW; ++kw)
            {
                // Compute the specific output location for this thread
                int h_out = h_tile + ty;
                int w_out = w_tile + tx;

                // Check if the output location is within the valid output range
                if (h_out < outH && w_out < outW)
                {
                    // Calculate the corresponding input indices
                    int h_in = h_out * strideH + kh * dilationH - padH;
                    int w_in = w_out * strideW + kw * dilationW - padW;

                    // Check bounds for input and accumulate if valid
                    if (h_in >= 0 && h_in < inH && w_in >= 0 && w_in < inW)
                    {
                        int input_idx = n * (inC * inH * inW) + (g * C_per_group + c) * (inH * inW) + h_in * inW + w_in;
                        int weight_idx = (g * (M / group) + m) * (C_per_group * kH * kW) + c * (kH * kW) + kh * kW + kw;
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Step 6: Add bias if it exists
    if (bias)
    {
        value += bias[m_global];
    }

    // Step 7: Write the result back to the output tensor, ensuring it’s within valid bounds
    if (h_tile + ty < outH && w_tile + tx < outW)
    {
        int output_idx = n * M * outH * outW + m_global * outH * outW + (h_tile + ty) * outW + (w_tile + tx);
        if (output_idx < N * M * outH * outW)
            output[output_idx] = value;
    }
}
