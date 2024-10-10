#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>
#include <utility>

#include <hip/hip_runtime.h>
// #include <miopen/miopen.h>

#include "kernels.hpp"
#include "utils.hpp"

// Reference CPU convolution for comparison
void cpu_convolution(
    const float *input, const float *weight, const float *bias, float *output,
    int64_t group, int64_t padH, int64_t padW, int64_t strideH, int64_t strideW,
    int64_t dilationH, int64_t dilationW,
    int64_t N, int64_t inC, int64_t inH, int64_t inW,
    int64_t M, int64_t C_per_group, int64_t kH, int64_t kW,
    int64_t outH, int64_t outW)
{
    // Initialize output to zero
    std::fill(output, output + N * M * outH * outW, 0.0f);

    // Perform convolution
    for (int64_t n = 0; n < N; ++n)
    {
        for (int64_t g = 0; g < group; ++g)
        {
            for (int64_t m = 0; m < M / group; ++m)
            {
                for (int64_t h_out = 0; h_out < outH; ++h_out)
                {
                    for (int64_t w_out = 0; w_out < outW; ++w_out)
                    {
                        int64_t output_idx = n * (M * outH * outW) +
                                             (g * (M / group) + m) * (outH * outW) +
                                             h_out * outW + w_out;

                        for (int64_t c = 0; c < C_per_group; ++c)
                        {
                            for (int64_t kh = 0; kh < kH; ++kh)
                            {
                                for (int64_t kw = 0; kw < kW; ++kw)
                                {
                                    int64_t h_in = h_out * strideH + kh * dilationH - padH;
                                    int64_t w_in = w_out * strideW + kw * dilationW - padW;

                                    if (h_in >= 0 && h_in < inH && w_in >= 0 && w_in < inW)
                                    {
                                        int64_t input_idx = n * (inC * inH * inW) +
                                                            (g * C_per_group + c) * (inH * inW) +
                                                            h_in * inW + w_in;

                                        int64_t weight_idx = (g * (M / group) + m) * (C_per_group * kH * kW) +
                                                             c * (kH * kW) + kh * kW + kw;

                                        output[output_idx] += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }

                        if (bias)
                        {
                            output[output_idx] += bias[g * (M / group) + m];
                        }
                    }
                }
            }
        }
    }
}

// Generate random data in the range [-0.5, 0.5] using mt19937
std::vector<float> generate_random_data(int64_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    std::vector<float> data(size);
    for (auto &val : data)
    {
        val = dis(gen);
    }
    return data;
}

// Calculate the number of FLOPs for convolution
int64_t calculate_convolution_flops(
    int64_t N, int64_t inC, int64_t inH, int64_t inW,
    int64_t M, int64_t C_group, int64_t kH, int64_t kW,
    int64_t outH, int64_t outW)
{
    return N * M * outH * outW * C_group * kH * kW * 2;
}

// Calculate output height and width
std::pair<int64_t, int64_t> get_outH_and_outW(
    int64_t inH, int64_t inW, int64_t kH, int64_t kW,
    int64_t padH, int64_t padW, int64_t strideH, int64_t strideW,
    int64_t dilationH, int64_t dilationW)
{
    int64_t outH = (inH + 2 * padH - (kH - 1) * dilationH - 1) / strideH + 1;
    int64_t outW = (inW + 2 * padW - (kW - 1) * dilationW - 1) / strideW + 1;
    return {outH, outW};
}

// Calculate the total output size
int64_t calculate_convolution_output_size(
    int64_t group, int64_t padH, int64_t padW, int64_t strideH, int64_t strideW,
    int64_t dilationH, int64_t dilationW,
    int64_t N, int64_t inC, int64_t inH, int64_t inW,
    int64_t M, int64_t C_per_group, int64_t kH, int64_t kW,
    int64_t outH, int64_t outW)
{
    return N * M * outH * outW;
}

// Utility functions to compute statistics
float get_min(const float *data, int64_t size)
{
    float min_val = data[0];
    for (int64_t i = 1; i < size; ++i)
    {
        if (data[i] < min_val)
            min_val = data[i];
    }
    return min_val;
}

float get_max(const float *data, int64_t size)
{
    float max_val = data[0];
    for (int64_t i = 1; i < size; ++i)
    {
        if (data[i] > max_val)
            max_val = data[i];
    }
    return max_val;
}

float get_mean(const float *data, int64_t size)
{
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i)
    {
        sum += data[i];
    }
    return sum / size;
}

float get_std(const float *data, int64_t size)
{
    float mean = get_mean(data, size);
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i)
    {
        sum += (data[i] - mean) * (data[i] - mean);
    }
    return std::sqrt(sum / size);
}

// Helper function to check correctness
void check_correctness(float *d_output, const float *ref_output, int64_t size)
{
    std::vector<float> output_gpu(size);
    hipCheck(hipMemcpy(output_gpu.data(), d_output, size * sizeof(float), hipMemcpyDeviceToHost));

    for (int64_t i = 0; i < size; ++i)
    {
        if (fabs(output_gpu[i] - ref_output[i]) > 1e-3)
        {
            std::cerr << "Mismatch at " << i << ": " << output_gpu[i] << " vs " << ref_output[i] << std::endl;
            exit(1);
        }
    }
}

// Helper function to run a kernel and measure its execution time
template <typename KernelLaunch>
double run_kernel_and_time(KernelLaunch launch, float *d_output, const float *ref_output, int64_t size)
{
    // Warmup
    launch();
    hipCheck(hipDeviceSynchronize());
    hipCheck(hipGetLastError());

    // Check correctness
    check_correctness(d_output, ref_output, size);

    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i)
    {
        launch();
        hipCheck(hipDeviceSynchronize());
        hipCheck(hipGetLastError());
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Return elapsed time in milliseconds
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Run convolution kernels (conv0 to conv6)
double run_conv0(int64_t group, int64_t padH, int64_t padW, int64_t strideH, int64_t strideW,
                 int64_t dilationH, int64_t dilationW, int64_t N, int64_t inC, int64_t inH, int64_t inW,
                 int64_t M, int64_t C_per_group, int64_t kH, int64_t kW, int64_t outH, int64_t outW,
                 float *d_input, float *d_weight, float *d_bias, float *d_output, const float *ref_output)
{
    dim3 blockDim(256);
    dim3 gridDim(CeilDiv(N * M * outH * outW, blockDim.x));

    auto launch = [&]()
    {
        conv_0<<<gridDim, blockDim>>>(d_input, d_weight, d_bias, d_output,
                                      group, padH, padW, strideH, strideW, dilationH, dilationW,
                                      N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW);
    };

    return run_kernel_and_time(launch, d_output, ref_output, N * M * outH * outW);
}

double run_conv1(int64_t group, int64_t padH, int64_t padW, int64_t strideH, int64_t strideW,
                 int64_t dilationH, int64_t dilationW, int64_t N, int64_t inC, int64_t inH, int64_t inW,
                 int64_t M, int64_t C_per_group, int64_t kH, int64_t kW, int64_t outH, int64_t outW,
                 float *d_input, float *d_weight, float *d_bias, float *d_output, const float *ref_output)
{
    dim3 blockDim(CONV_1_TILE_W, CONV_1_TILE_H);
    dim3 gridDim(CeilDiv(outW, blockDim.x), CeilDiv(outH, blockDim.y), N * M);

    auto launch = [&]()
    {
        conv_1<<<gridDim, blockDim>>>(d_input, d_weight, d_bias, d_output,
                                      group, padH, padW, strideH, strideW, dilationH, dilationW,
                                      N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW);
    };

    return run_kernel_and_time(launch, d_output, ref_output, N * M * outH * outW);
}

double run_conv2(int64_t group, int64_t padH, int64_t padW, int64_t strideH, int64_t strideW,
                 int64_t dilationH, int64_t dilationW, int64_t N, int64_t inC, int64_t inH, int64_t inW,
                 int64_t M, int64_t C_per_group, int64_t kH, int64_t kW, int64_t outH, int64_t outW,
                 float *d_input, float *d_weight, float *d_bias, float *d_output, const float *ref_output)
{
    dim3 blockDim(CONV_2_TILE_W, CONV_2_TILE_H);
    dim3 gridDim(CeilDiv(outW, blockDim.x), CeilDiv(outH, blockDim.y), N * M);

    // Compute shared memory size
    int shared_input_h = (CONV_2_TILE_H - 1) * strideH + (kH - 1) * dilationH + 1;
    int shared_input_w = (CONV_2_TILE_W - 1) * strideW + (kW - 1) * dilationW + 1;
    size_t shared_memory_size = shared_input_h * shared_input_w * sizeof(float);

    auto launch = [&]()
    {
        conv_2<<<gridDim, blockDim, shared_memory_size>>>(d_input, d_weight, d_bias, d_output,
                                                          group, padH, padW, strideH, strideW, dilationH, dilationW,
                                                          N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW,
                                                          shared_input_h, shared_input_w,
                                                          blockDim.x * blockDim.y, shared_input_h * shared_input_w,
                                                          CeilDiv(shared_input_h * shared_input_w, static_cast<int>(blockDim.x * blockDim.y)));
    };

    return run_kernel_and_time(launch, d_output, ref_output, N * M * outH * outW);
}

double run_conv3(int64_t group, int64_t padH, int64_t padW, int64_t strideH, int64_t strideW,
                 int64_t dilationH, int64_t dilationW, int64_t N, int64_t inC, int64_t inH, int64_t inW,
                 int64_t M, int64_t C_per_group, int64_t kH, int64_t kW, int64_t outH, int64_t outW,
                 float *d_input, float *d_weight, float *d_bias, float *d_output, const float *ref_output)
{
    dim3 blockDim(CONV_3_TILE_W, CONV_3_TILE_H);
    dim3 gridDim(CeilDiv(outW, blockDim.x), CeilDiv(outH, blockDim.y), N * M);

    // Compute shared memory size
    int shared_input_h = (CONV_3_TILE_H - 1) * strideH + (kH - 1) * dilationH + 1;
    int shared_input_w = (CONV_3_TILE_W - 1) * strideW + (kW - 1) * dilationW + 1;
    size_t shared_memory_size = shared_input_h * shared_input_w * sizeof(float);

    auto launch = [&]()
    {
        conv_3<<<gridDim, blockDim, shared_memory_size>>>(d_input, d_weight, d_bias, d_output,
                                                          group, padH, padW, strideH, strideW, dilationH, dilationW,
                                                          N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW,
                                                          shared_input_h, shared_input_w,
                                                          blockDim.x * blockDim.y, shared_input_h * shared_input_w,
                                                          CeilDiv(shared_input_h * shared_input_w, static_cast<int>(blockDim.x * blockDim.y)));
    };

    return run_kernel_and_time(launch, d_output, ref_output, N * M * outH * outW);
}

double run_winograd_conv(int conv_type, int64_t group, int64_t padH, int64_t padW,
                         int64_t strideH, int64_t strideW, int64_t dilationH, int64_t dilationW,
                         int64_t N, int64_t inC, int64_t inH, int64_t inW,
                         int64_t M, int64_t C_per_group, int64_t kH, int64_t kW,
                         int64_t outH, int64_t outW,
                         float *d_input, const float *weight, float *d_bias,
                         float *d_output, const float *ref_output)
{
    if (kH != 3 || kW != 3)
    {
        std::cerr << "Only 3x3 kernels are supported for Winograd convolution." << std::endl;
        exit(1);
    }

    // Transform kernels
    std::vector<float> transformed_kernel(M * C_per_group * 16); // 4x4 transformed kernels
    for (int64_t i = 0; i < M * C_per_group; ++i)
    {
        winogradTransformKernel(weight + i * 9, transformed_kernel.data() + i * 16);
    }

    // Allocate and copy transformed kernels to device
    float *d_transformed_kernel = nullptr;
    hipCheck(hipMalloc(&d_transformed_kernel, M * C_per_group * 16 * sizeof(float)));
    hipCheck(hipMemcpy(d_transformed_kernel, transformed_kernel.data(), M * C_per_group * 16 * sizeof(float), hipMemcpyHostToDevice));

    // Define block and grid dimensions based on convolution type
    dim3 blockDim, gridDim;

    switch (conv_type)
    {
    case 4:
        blockDim = dim3(CONV_4_TILE_W, CONV_4_TILE_H);
        gridDim = dim3(CeilDiv(outW, CONV_4_TILE_W * 2), CeilDiv(outH, CONV_4_TILE_H * 2), N * M);
        break;
    case 5:
        blockDim = dim3(CONV_5_TILE_W, CONV_5_TILE_H);
        gridDim = dim3(CeilDiv(outW, CONV_5_TILE_W * 2), CeilDiv(outH, CONV_5_TILE_H * 2), N * M);
        break;
    case 6:
        blockDim = dim3(CONV_6_TILE_W, CONV_6_TILE_H);
        gridDim = dim3(CeilDiv(outW, CONV_6_TILE_W * 2), CeilDiv(outH, CONV_6_TILE_H * 2), N * M);
        break;
    default:
        std::cerr << "Invalid convolution type for Winograd convolution." << std::endl;
        exit(1);
    }

    // Warmup and correctness check
    switch (conv_type)
    {
    case 4:
        conv_4<<<gridDim, blockDim>>>(d_input, d_transformed_kernel, d_bias, d_output,
                                      group, padH, padW, strideH, strideW, dilationH, dilationW,
                                      N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW);
        break;
    case 5:
        conv_5<<<gridDim, blockDim>>>(d_input, d_transformed_kernel, d_bias, d_output,
                                      group, padH, padW, strideH, strideW, dilationH, dilationW,
                                      N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW);
        break;
    case 6:
        conv_6<<<gridDim, blockDim>>>(d_input, d_transformed_kernel, d_bias, d_output,
                                      group, padH, padW, strideH, strideW, dilationH, dilationW,
                                      N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW);
        break;
    }

    hipCheck(hipDeviceSynchronize());
    hipCheck(hipGetLastError());
    check_correctness(d_output, ref_output, N * M * outH * outW);

    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i)
    {
        // Re-transform kernels
        for (int64_t j = 0; j < M * C_per_group; ++j)
        {
            winogradTransformKernel(weight + j * 9, transformed_kernel.data() + j * 16);
        }
        hipCheck(hipMemcpy(d_transformed_kernel, transformed_kernel.data(), M * C_per_group * 16 * sizeof(float), hipMemcpyHostToDevice));

        // Launch the appropriate kernel
        switch (conv_type)
        {
        case 4:
            conv_4<<<gridDim, blockDim>>>(d_input, d_transformed_kernel, d_bias, d_output,
                                          group, padH, padW, strideH, strideW, dilationH, dilationW,
                                          N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW);
            break;
        case 5:
            conv_5<<<gridDim, blockDim>>>(d_input, d_transformed_kernel, d_bias, d_output,
                                          group, padH, padW, strideH, strideW, dilationH, dilationW,
                                          N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW);
            break;
        case 6:
            conv_6<<<gridDim, blockDim>>>(d_input, d_transformed_kernel, d_bias, d_output,
                                          group, padH, padW, strideH, strideW, dilationH, dilationW,
                                          N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW);
            break;
        }

        hipCheck(hipDeviceSynchronize());
        hipCheck(hipGetLastError());
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Free transformed kernel memory
    hipCheck(hipFree(d_transformed_kernel));

    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main()
{
    // Convolution attributes
    int64_t group = 1;
    int64_t padH = 1;
    int64_t padW = 1;
    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t dilationH = 1;
    int64_t dilationW = 1;

    // Input shape
    int64_t N = 32;
    int64_t inC = 3;
    int64_t inH = 512;
    int64_t inW = 512;

    // Weight shape
    int64_t M = 8;
    int64_t C_per_group = inC / group;
    int64_t kH = 3;
    int64_t kW = 3;

    // Output shapes
    auto [outH, outW] = get_outH_and_outW(inH, inW, kH, kW, padH, padW, strideH, strideW, dilationH, dilationW);

    // Calculate FLOPs and output size
    int64_t flops = calculate_convolution_flops(N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW);
    int64_t output_size = calculate_convolution_output_size(group, padH, padW, strideH, strideW,
                                                            dilationH, dilationW, N, inC, inH, inW,
                                                            M, C_per_group, kH, kW, outH, outW);

    // Generate random data
    auto input = generate_random_data(N * inC * inH * inW);
    auto weight = generate_random_data(M * C_per_group * kH * kW);
    auto bias = generate_random_data(M);
    std::vector<float> ref_output(output_size, 0.0f);

    // Run CPU convolution
    std::cout << "Running CPU convolution..." << std::endl;
    cpu_convolution(input.data(), weight.data(), bias.data(), ref_output.data(),
                    group, padH, padW, strideH, strideW, dilationH, dilationW,
                    N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW);
    std::cout << "CPU convolution complete.\nNow running GPU Convolutions..." << std::endl;

    // Compute statistics
    // float min_val = get_min(ref_output.data(), output_size);
    // float max_val = get_max(ref_output.data(), output_size);
    // float mean = get_mean(ref_output.data(), output_size);
    // float std_dev = get_std(ref_output.data(), output_size);

    // // Print statistics
    // std::cout << "CPU convolution statistics:" << std::endl;
    // std::cout << "Min: " << min_val << std::endl;
    // std::cout << "Max: " << max_val << std::endl;
    // std::cout << "Mean: " << mean << std::endl;
    // std::cout << "Std Dev: " << std_dev << std::endl;

    // Prepare GPU data
    float *d_input = nullptr, *d_weight = nullptr, *d_bias = nullptr, *d_output = nullptr;
    hipCheck(hipMalloc(&d_input, N * inC * inH * inW * sizeof(float)));
    hipCheck(hipMalloc(&d_weight, M * C_per_group * kH * kW * sizeof(float)));
    hipCheck(hipMalloc(&d_bias, M * sizeof(float)));
    hipCheck(hipMalloc(&d_output, output_size * sizeof(float)));

    hipCheck(hipMemcpy(d_input, input.data(), N * inC * inH * inW * sizeof(float), hipMemcpyHostToDevice));
    hipCheck(hipMemcpy(d_weight, weight.data(), M * C_per_group * kH * kW * sizeof(float), hipMemcpyHostToDevice));
    hipCheck(hipMemcpy(d_bias, bias.data(), M * sizeof(float), hipMemcpyHostToDevice));

    // Run conv0: Naive
    hipCheck(hipMemset(d_output, 0, output_size * sizeof(float)));
    double conv0_time = run_conv0(group, padH, padW, strideH, strideW, dilationH, dilationW,
                                  N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW,
                                  d_input, d_weight, d_bias, d_output, ref_output.data());
    std::cout << "GPU Convolution 0 (Naive) Time: " << conv0_time << " ms" << std::endl;

    // Run conv1: Tiling
    hipCheck(hipMemset(d_output, 0, output_size * sizeof(float)));
    double conv1_time = run_conv1(group, padH, padW, strideH, strideW, dilationH, dilationW,
                                  N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW,
                                  d_input, d_weight, d_bias, d_output, ref_output.data());
    std::cout << "GPU Convolution 1 (Tiling) Time: " << conv1_time << " ms" << std::endl;

    // Run conv2: Shared Memory
    hipCheck(hipMemset(d_output, 0, output_size * sizeof(float)));
    double conv2_time = run_conv2(group, padH, padW, strideH, strideW, dilationH, dilationW,
                                  N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW,
                                  d_input, d_weight, d_bias, d_output, ref_output.data());
    std::cout << "GPU Convolution 2 (Shared Memory) Time: " << conv2_time << " ms" << std::endl;

    // // Run conv3: Vectorization
    // hipCheck(hipMemset(d_output, 0, output_size * sizeof(float)));
    // double conv3_time = run_conv3(group, padH, padW, strideH, strideW, dilationH, dilationW,
    //                               N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW,
    //                               d_input, d_weight, d_bias, d_output, ref_output.data());
    // std::cout << "GPU Convolution 3 (Vectorization) Time: " << conv3_time << " ms" << std::endl;

    // Run conv4: Winograd
    hipCheck(hipMemset(d_output, 0, output_size * sizeof(float)));
    double conv4_time = run_winograd_conv(4, group, padH, padW, strideH, strideW, dilationH, dilationW,
                                          N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW,
                                          d_input, weight.data(), d_bias, d_output, ref_output.data());
    std::cout << "GPU Convolution 4 (Winograd) Time: " << conv4_time << " ms" << std::endl;

    // Run conv5: Winograd with Shared Memory
    hipCheck(hipMemset(d_output, 0, output_size * sizeof(float)));
    double conv5_time = run_winograd_conv(5, group, padH, padW, strideH, strideW, dilationH, dilationW,
                                          N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW,
                                          d_input, weight.data(), d_bias, d_output, ref_output.data());
    std::cout << "GPU Convolution 5 (Winograd with Shared Memory) Time: " << conv5_time << " ms" << std::endl;

    // Run conv6: Winograd with many Optimizations
    hipCheck(hipMemset(d_output, 0, output_size * sizeof(float)));
    double conv6_time = run_winograd_conv(6, group, padH, padW, strideH, strideW, dilationH, dilationW,
                                          N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW,
                                          d_input, weight.data(), d_bias, d_output, ref_output.data());
    std::cout << "GPU Convolution 6 (Winograd with Many Optimizations) Time: " << conv6_time << " ms" << std::endl;

    // hipCheck(hipMemset(d_output, 0, output_size * sizeof(float)));
    // double miopen_time = run_miopen_conv(group, padH, padW, strideH, strideW, dilationH, dilationW,
    //                                      N, inC, inH, inW, M, C_per_group, kH, kW, outH, outW,
    //                                      d_input, d_weight, d_bias, d_output, ref_output.data());
    // std::cout << "MIOpen Convolution Time: " << miopen_time << " ms" << std::endl;

    // Free GPU memory
    hipCheck(hipFree(d_input));
    hipCheck(hipFree(d_weight));
    hipCheck(hipFree(d_bias));
    hipCheck(hipFree(d_output));

    return 0;
}
