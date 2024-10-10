#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

#define CONV_6_TILE_H 16
#define CONV_6_TILE_W 16

__device__ inline void winogradTransformInput_inline(const float input[4][4], float transformed_input[4][4])
{
    float temp[4][4];

#pragma unroll
    for (int64_t i = 0; i < 4; ++i)
    {
        temp[i][0] = input[i][0] - input[i][2];
        temp[i][1] = input[i][1] + input[i][2];
        temp[i][2] = -input[i][1] + input[i][2];
        temp[i][3] = input[i][1] - input[i][3];
    }

#pragma unroll
    for (int64_t i = 0; i < 4; ++i)
    {
        transformed_input[0][i] = temp[0][i] - temp[2][i];
        transformed_input[1][i] = temp[1][i] + temp[2][i];
        transformed_input[2][i] = -temp[1][i] + temp[2][i];
        transformed_input[3][i] = temp[1][i] - temp[3][i];
    }
}

__device__ inline void winogradTransformOutput_inline(const float winograd_output[4][4], float output_tile[2][2])
{
    float temp[2][4];

#pragma unroll
    for (int64_t i = 0; i < 4; i += 2)
    {
        temp[0][i] = winograd_output[0][i] + winograd_output[1][i] + winograd_output[2][i];
        temp[1][i] = winograd_output[1][i] - winograd_output[2][i] - winograd_output[3][i];
    }

#pragma unroll
    for (int64_t i = 0; i < 2; ++i)
    {
        output_tile[i][0] = temp[i][0] + temp[i][1] + temp[i][2];
        output_tile[i][1] = temp[i][1] - temp[i][2] - temp[i][3];
    }
}

__global__ void conv_6(
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
    int w_out_tile = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out_tile = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (w_out_tile >= (outW + 1) / 2 || h_out_tile >= (outH + 1) / 2)
        return;

    int n = blockIdx.z / group;
    int g = blockIdx.z % group;

    // Boundary check
    if (n >= N || g >= group)
        return;

    int M_per_group = M / group;

    __shared__ float shared_transformed_kernel[4][4];
    // --- Step 2: Initialize Output Tile ---
    float output_tile[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    // --- Step 3: Iterate Over Output Channels ---
    for (int64_t m = 0; m < M_per_group; ++m)
    {

        // --- Step 4: Iterate Over Input Channels ---
        for (int64_t c = 0; c < C_per_group; ++c)
        {
            // --- Step 5: Load Input Tile from Global Memory ---
            float4 input_tile_vec[2][2];
            for (int64_t ih = 0; ih < 4; ih += 2)
            {
                for (int64_t iw = 0; iw < 4; iw += 2)
                {
                    float4 loaded = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    for (int64_t i = 0; i < 2; ++i)
                    {
                        for (int64_t j = 0; j < 2; ++j)
                        {
                            int h_in = (h_out_tile * 2) * strideH + (ih + i) - padH;
                            int w_in = (w_out_tile * 2) * strideW + (iw + j) - padW;

                            float value = 0.0f;
                            if (h_in >= 0 && h_in < inH && w_in >= 0 && w_in < inW)
                            {
                                int64_t input_idx = n * (inC * inH * inW) + (g * C_per_group + c) * (inH * inW) + h_in * inW + w_in;
                                value = input[input_idx];
                            }

                            if (i == 0 && j == 0)
                                loaded.x = value;
                            else if (i == 0 && j == 1)
                                loaded.y = value;
                            else if (i == 1 && j == 0)
                                loaded.z = value;
                            else if (i == 1 && j == 1)
                                loaded.w = value;
                        }
                    }
                    input_tile_vec[ih / 2][iw / 2] = loaded;
                }
            }

            float input_tile[4][4];

            for (int i = 0; i < 2; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    input_tile[i * 2][j * 2] = input_tile_vec[i][j].x;
                    input_tile[i * 2][j * 2 + 1] = input_tile_vec[i][j].y;
                    input_tile[i * 2 + 1][j * 2] = input_tile_vec[i][j].z;
                    input_tile[i * 2 + 1][j * 2 + 1] = input_tile_vec[i][j].w;
                }
            }

            // --- Step 6: Transform the Input Tile ---
            float transformed_input[4][4];
            winogradTransformInput_inline(input_tile, transformed_input);

            // --- Step 7: Load Transformed Kernel into Shared Memory ---
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                int64_t kernel_idx = ((g * M_per_group + m) * C_per_group + c) * 16;
                for (int64_t i = 0; i < 4; ++i)
                {
                    for (int64_t j = 0; j < 4; ++j)
                    {
                        shared_transformed_kernel[i][j] = transformed_kernel_data[kernel_idx + i * 4 + j];
                    }
                }
            }
            __syncthreads();

            // --- Step 8: Multiply Transformed Input with Transformed Kernel ---
            float transformed_output[4][4];
            for (int64_t ih = 0; ih < 4; ++ih)
            {
                for (int64_t iw = 0; iw < 4; ++iw)
                {
                    transformed_output[ih][iw] = transformed_input[ih][iw] * shared_transformed_kernel[ih][iw];
                }
            }

            // --- Step 9: Transform the Output Tile Back ---
            float output_tile_partial[2][2];
            winogradTransformOutput_inline(transformed_output, output_tile_partial);

// --- Step 10: Accumulate the Partial Output ---
#pragma unroll
            for (int64_t ih = 0; ih < 2; ++ih)
            {
#pragma unroll
                for (int64_t iw = 0; iw < 2; ++iw)
                {
                    output_tile[ih][iw] += output_tile_partial[ih][iw];
                }
            }

            __syncthreads(); // Ensure all threads have finished using the current kernel
        }

        // --- Step 11: Add Bias (if available) ---
        if (bias)
        {
            float bias_val = bias[g * M_per_group + m];
            for (int64_t ih = 0; ih < 2; ++ih)
            {
                for (int64_t iw = 0; iw < 2; ++iw)
                {
                    output_tile[ih][iw] += bias_val;
                }
            }
        }

// --- Step 12: Write the Output Tile to Global Memory ---
#pragma unroll
        for (int64_t ih = 0; ih < 2; ++ih)
        {
#pragma unroll
            for (int64_t iw = 0; iw < 2; ++iw)
            {
                int64_t h = h_out_tile * 2 + ih;
                int64_t w = w_out_tile * 2 + iw;

                if (h < outH && w < outW)
                {
                    int64_t output_idx = n * (M * outH * outW) +
                                         (g * M_per_group + m) * (outH * outW) +
                                         h * outW + w;
                    output[output_idx] = output_tile[ih][iw];
                }
            }
        }
    }
}
