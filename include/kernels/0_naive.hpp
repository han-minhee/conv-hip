#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

#define CONV_0_TILE 256

__global__ void conv_0(
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
    // Step 1: Calculate the flattened index and ensure it is within valid bounds
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * M * outH * outW) return;

    // Step 2: Decompose the flattened index into batch (n), output channel (m_global), output height (h_out), and width (w_out)
    int n = idx / (M * outH * outW);
    int m_global = (idx % (M * outH * outW)) / (outH * outW);
    int h_out = (idx % (outH * outW)) / outW;
    int w_out = idx % outW;

    // Step 3: Determine the group and the specific channel within the group
    int g = m_global / (M / group);
    int m = m_global % (M / group);

    // Step 4: Initialize output value and iterate over input channels and kernel dimensions
    float value = 0.0f;
    for (int c = 0; c < C_per_group; ++c) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                // Calculate input indices, including stride, dilation, and padding
                int h_in = h_out * strideH + kh * dilationH - padH;
                int w_in = w_out * strideW + kw * dilationW - padW;

                // Accumulate only if within bounds
                if (h_in >= 0 && h_in < inH && w_in >= 0 && w_in < inW) {
                    int input_idx = n * (inC * inH * inW) + (g * C_per_group + c) * (inH * inW) + h_in * inW + w_in;
                    int weight_idx = (g * (M / group) + m) * (C_per_group * kH * kW) + c * (kH * kW) + kh * kW + kw;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Step 5: Add bias if it exists, and store the result in the output array
    if (bias) {
        value += bias[m_global];
    }
    output[idx] = value;
}
