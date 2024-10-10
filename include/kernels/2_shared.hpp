#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

#define CONV_2_TILE_H 8
#define CONV_2_TILE_W 8

__global__ void conv_2(
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
    // Step 1: Calculate tile and thread positions in the output grid
    int h_tile = blockIdx.y * CONV_2_TILE_H;
    int w_tile = blockIdx.x * CONV_2_TILE_W;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int h_out = h_tile + ty;
    int w_out = w_tile + tx;

    // Step 2: Perform boundary check to ensure the output indices are within range
    if (h_out >= outH && w_out >= outW)
        return;

    // Step 3: Determine batch and output channel indices for grouped convolutions
    int n = blockIdx.z / M;
    int m_global = blockIdx.z % M;
    int g = m_global / (M / group);
    int m = m_global % (M / group);

    // Step 4: Allocate shared memory for the input tile and calculate the starting position of the input tile
    extern __shared__ float shared_input[];
    int h_in_start = h_tile * strideH - padH;
    int w_in_start = w_tile * strideW - padW;

    // Step 5: Initialize the output accumulator
    float value = 0.0f;

    // Step 6: Load relevant input data into shared memory for each input channel in the current group
    for (int c = 0; c < C_per_group; ++c)
    {
        for (int i = 0; i < num_blocks; ++i)
        {
            int index = threadIdx.y * blockDim.x + threadIdx.x + i * threads_per_block;

            if (index < total_elements)
            {
                int s_y = index / shared_input_w;
                int s_x = index % shared_input_w;
                int h_in = h_in_start + s_y;
                int w_in = w_in_start + s_x;

                // Load input into shared memory or set to 0 if out of bounds
                if (h_in >= 0 && h_in < inH && w_in >= 0 && w_in < inW)
                {
                    int input_idx = n * (inC * inH * inW) + (g * C_per_group + c) * (inH * inW) + h_in * inW + w_in;
                    shared_input[s_y * shared_input_w + s_x] = input[input_idx];
                }
                else
                {
                    shared_input[s_y * shared_input_w + s_x] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Step 7: Perform the convolution by iterating over the kernel dimensions
        if (h_out < outH && w_out < outW)
        {
            for (int kh = 0; kh < kH; ++kh)
            {
                for (int kw = 0; kw < kW; ++kw)
                {
                    int h_in_s = ty * strideH + kh * dilationH;
                    int w_in_s = tx * strideW + kw * dilationW;

                    float input_value = shared_input[h_in_s * shared_input_w + w_in_s];
                    int weight_idx = (g * (M / group) + m) * (C_per_group * kH * kW) + c * (kH * kW) + kh * kW + kw;
                    float weight_value = weight[weight_idx];

                    value += input_value * weight_value;
                }
            }
        }
        __syncthreads();
    }

    // Step 8: Add bias if it exists, then store the final result in the output tensor
    if (bias)
    {
        value += bias[m_global];
    }

    int output_idx = n * M * outH * outW + m_global * outH * outW + h_out * outW + w_out;
    output[output_idx] = value;
}
