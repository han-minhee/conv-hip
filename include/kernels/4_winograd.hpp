#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

#define CONV_4_TILE_H 4
#define CONV_4_TILE_W 4

// Step 1: Define transformation matrices for the Winograd kernel transformation
const float G[4][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.5f, 0.5f, 0.5f},
    {0.5f, -0.5f, 0.5f},
    {0.0f, 0.0f, 1.0f}};
const float GT[3][4] = {
    {1.0f, 0.5f, 0.5f, 0.0f},
    {0.0f, 0.5f, -0.5f, 0.0f},
    {0.0f, 0.5f, 0.5f, 1.0f}};

void winogradTransformKernel(const float *kernel, float *transformed_kernel)
{
    float temp[4][3];
    // Step 2: Perform kernel transformation using G * kernel
    for (int64_t i = 0; i < 4; ++i)
    {
        for (int64_t j = 0; j < 3; ++j)
        {
            temp[i][j] = 0;
            for (int64_t k = 0; k < 3; ++k)
            {
                temp[i][j] += G[i][k] * kernel[k * 3 + j];
            }
        }
    }

    // Step 3: Complete the transformation with temp * GT, store in transformed_kernel
    for (int64_t i = 0; i < 4; ++i)
    {
        for (int64_t j = 0; j < 4; ++j)
        {
            transformed_kernel[i * 4 + j] = 0;
            for (int64_t k = 0; k < 3; ++k)
            {
                transformed_kernel[i * 4 + j] += temp[i][k] * GT[k][j];
            }
        }
    }
}

__device__ void winogradTransformInput(const float d[4][4], float transformed_input[4][4])
{
    // Perform Winograd input transformation in two stages for a 4x4 tile
    float temp[4][4];

    for (int64_t i = 0; i < 4; ++i)
    {
        temp[i][0] = d[i][0] - d[i][2];
        temp[i][1] = d[i][1] + d[i][2];
        temp[i][2] = -d[i][1] + d[i][2];
        temp[i][3] = d[i][1] - d[i][3];
    }

    for (int64_t i = 0; i < 4; ++i)
    {
        transformed_input[0][i] = temp[0][i] - temp[2][i];
        transformed_input[1][i] = temp[1][i] + temp[2][i];
        transformed_input[2][i] = -temp[1][i] + temp[2][i];
        transformed_input[3][i] = temp[1][i] - temp[3][i];
    }
}

__device__ void winogradTransformOutput(const float m[4][4], float output_tile[2][2])
{
    // Transform the intermediate matrix m to obtain the final 2x2 output tile
    float temp[2][4];

    for (int64_t i = 0; i < 4; ++i)
    {
        temp[0][i] = m[0][i] + m[1][i] + m[2][i];
        temp[1][i] = m[1][i] - m[2][i] - m[3][i];
    }

    for (int64_t i = 0; i < 2; ++i)
    {
        output_tile[i][0] = temp[i][0] + temp[i][1] + temp[i][2];
        output_tile[i][1] = temp[i][1] - temp[i][2] - temp[i][3];
    }
}

__global__ void conv_4(
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
    // Step 1: Calculate output tile coordinates, then perform boundary checks
    int w_out_tile = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out_tile = blockIdx.y * blockDim.y + threadIdx.y;
    if (w_out_tile >= (outW + 1) / 2 || h_out_tile >= (outH + 1) / 2)
        return;

    int n = blockIdx.z / group;
    int g = blockIdx.z % group;
    if (n >= N || g >= group)
        return;

    // Step 2: Loop over output channels in the current group
    for (int64_t m = 0; m < M / group; ++m)
    {
        float output_tile[2][2] = {{0, 0}, {0, 0}};

        // Step 3: Loop over input channels in the current group, loading a 4x4 input tile
        for (int64_t c = 0; c < C_per_group; ++c)
        {
            float input_tile[4][4];
            for (int64_t ih = 0; ih < 4; ++ih)
            {
                for (int64_t iw = 0; iw < 4; ++iw)
                {
                    int h_in = (h_out_tile * 2) * strideH + ih - padH;
                    int w_in = (w_out_tile * 2) * strideW + iw - padW;

                    if (h_in >= 0 && h_in < inH && w_in >= 0 && w_in < inW)
                    {
                        int64_t input_idx = n * (inC * inH * inW) + (g * C_per_group + c) * (inH * inW) + h_in * inW + w_in;
                        input_tile[ih][iw] = input[input_idx];
                    }
                    else
                    {
                        input_tile[ih][iw] = 0;
                    }
                }
            }

            // Step 4: Perform input transformation for this tile
            float transformed_input[4][4];
            winogradTransformInput(input_tile, transformed_input);

            // Step 5: Load pre-transformed kernel for this input channel
            float transformed_kernel[4][4];
            int64_t kernel_idx = ((g * (M / group) + m) * C_per_group + c) * 16;
            for (int64_t i = 0; i < 4; ++i)
            {
                for (int64_t j = 0; j < 4; ++j)
                {
                    transformed_kernel[i][j] = transformed_kernel_data[kernel_idx + i * 4 + j];
                }
            }

            // Step 6: Element-wise multiply transformed input and kernel, storing the result in transformed_output
            float transformed_output[4][4];
            for (int64_t i = 0; i < 4; ++i)
            {
                for (int64_t j = 0; j < 4; ++j)
                {
                    transformed_output[i][j] = transformed_input[i][j] * transformed_kernel[i][j];
                }
            }

            // Step 7: Transform the intermediate output back to the spatial domain
            float output_tile_partial[2][2];
            winogradTransformOutput(transformed_output, output_tile_partial);

            // Accumulate the transformed output tile
            for (int64_t i = 0; i < 2; ++i)
            {
                for (int64_t j = 0; j < 2; ++j)
                {
                    output_tile[i][j] += output_tile_partial[i][j];
                }
            }
        }

        // Step 8: Add bias if it exists
        if (bias)
        {
            float bias_val = bias[g * (M / group) + m];
            for (int64_t i = 0; i < 2; ++i)
            {
                for (int64_t j = 0; j < 2; ++j)
                {
                    output_tile[i][j] += bias_val;
                }
            }
        }

        // Step 9: Store the final result in the output tensor, ensuring it’s within bounds
        for (int64_t i = 0; i < 2; ++i)
        {
            for (int64_t j = 0; j < 2; ++j)
            {
                int64_t h = h_out_tile * 2 + i;
                int64_t w = w_out_tile * 2 + j;

                if (h < outH && w < outW)
                {
                    int64_t output_idx = n * (M * outH * outW) +
                                         (g * (M / group) + m) * (outH * outW) +
                                         h * outW + w;
                    output[output_idx] = output_tile[i][j];
                }
            }
        }
    }
}
