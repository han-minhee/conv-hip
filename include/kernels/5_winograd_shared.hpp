#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

#define CONV_5_TILE_W 16
#define CONV_5_TILE_H 16

__global__ void conv_5(
    // data
    float *input, float *transformed_kernel_data, float *bias, float *output,

    // attributes
    int64_t group, int64_t padH, int64_t padW, int64_t strideH, int64_t strideW, int64_t dilationH, int64_t dilationW,

    // input shape
    int64_t N, int64_t inC, int64_t inH, int64_t inW,

    // weight shape
    int64_t M, int64_t C_per_group, int64_t kH, int64_t kW,

    // output shapes
    int64_t outH, int64_t outW)
{
    // Step 1: Determine output tile position in the output grid
    int w_out_tile = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out_tile = blockIdx.y * blockDim.y + threadIdx.y;

    // Step 2: Check if the output tile is within bounds
    if (w_out_tile >= (outW + 1) / 2 || h_out_tile >= (outH + 1) / 2)
        return;

    // Step 3: Calculate batch and group indices
    int n = blockIdx.z / group;
    int g = blockIdx.z % group;

    // Step 4: Perform a boundary check to ensure valid batch and group indices
    if (n >= N || g >= group)
        return;

    // Step 5: Initialize shared memory for transformed kernel
    int M_per_group = M / group;
    __shared__ float shared_transformed_kernel[4][4];

    // Step 6: Iterate over the output channels for the group
    for (int64_t m = 0; m < M_per_group; ++m)
    {
        // Step 7: Initialize the output tile accumulator
        float output_tile[2][2] = {{0, 0}, {0, 0}};

        // Step 8: Loop over each input channel in the current group
        for (int64_t c = 0; c < C_per_group; ++c)
        {
            // Step 9: Load input data for the current tile
            float input_tile[4][4];

            for (int64_t ih = 0; ih < 4; ++ih)
            {
                for (int64_t iw = 0; iw < 4; ++iw)
                {
                    int h_in = (h_out_tile * 2) * strideH + ih - padH;
                    int w_in = (w_out_tile * 2) * strideW + iw - padW;

                    if (h_in >= 0 && h_in < inH && w_in >= 0 && w_in < inW)
                    {
                        int64_t input_idx = n * (inC * inH * inW) + (g * (C_per_group) + c) * (inH * inW) + h_in * inW + w_in;
                        input_tile[ih][iw] = input[input_idx];
                    }
                    else
                    {
                        input_tile[ih][iw] = 0;
                    }
                }
            }

            // Step 10: Transform the input tile using the Winograd transformation
            float transformed_input[4][4];
            winogradTransformInput(input_tile, transformed_input);

            // Step 11: Load the transformed kernel into shared memory
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                int64_t kernel_idx = ((g * (M_per_group) + m) * (C_per_group) + c) * 16;
                for (int64_t i = 0; i < 4; ++i)
                {
                    for (int64_t j = 0; j < 4; ++j)
                    {
                        shared_transformed_kernel[i][j] = transformed_kernel_data[kernel_idx + i * 4 + j];
                    }
                }
            }
            __syncthreads();

            // Step 12: Multiply transformed input and kernel element-wise to get the transformed output
            float transformed_output[4][4];
            for (int64_t i = 0; i < 4; ++i)
            {
                for (int64_t j = 0; j < 4; ++j)
                {
                    transformed_output[i][j] = transformed_input[i][j] * shared_transformed_kernel[i][j];
                }
            }

            // Step 13: Transform the partial output back to the spatial domain
            float output_tile_partial[2][2];
            winogradTransformOutput(transformed_output, output_tile_partial);

            // Step 14: Accumulate the partial output into the output tile
            for (int64_t i = 0; i < 2; ++i)
            {
                for (int64_t j = 0; j < 2; ++j)
                {
                    output_tile[i][j] += output_tile_partial[i][j];
                }
            }

            __syncthreads();
        }

        // Step 15: Add bias if it exists
        if (bias)
        {
            float bias_val = bias[g * (M_per_group) + m];
            for (int64_t i = 0; i < 2; ++i)
            {
                for (int64_t j = 0; j < 2; ++j)
                {
                    output_tile[i][j] += bias_val;
                }
            }
        }

        // Step 16: Write the accumulated output tile to the output tensor
        for (int64_t i = 0; i < 2; ++i)
        {
            for (int64_t j = 0; j < 2; ++j)
            {
                int64_t h = h_out_tile * 2 + i;
                int64_t w = w_out_tile * 2 + j;

                if (h < outH && w < outW)
                {
                    int64_t output_idx = n * (M * outH * outW) +
                                         (g * (M_per_group) + m) * (outH * outW) +
                                         h * outW + w;
                    output[output_idx] = output_tile[i][j];
                }
            }
        }
    }
}
