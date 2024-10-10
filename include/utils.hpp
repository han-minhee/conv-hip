#ifndef UTILS_HPP
#define UTILS_HPP

#include <hip/hip_runtime.h>
// #include <miopen/miopen.h>

#define hipCheck(err)                                                                                   \
    if (err != hipSuccess)                                                                              \
    {                                                                                                   \
        fprintf(stderr, "Error: '%s'(%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        exit(err);                                                                                      \
    };

#define hipKernelCheck(kernelCall)    \
    kernelCall;                       \
    hipCheck(hipDeviceSynchronize()); \
    hipCheck(hipGetLastError());

#define CeilDiv(a, b) (((a) + (b) - 1) / (b))

// #define miopenCheck(status)                                                                                    \
//     if (status != miopenStatusSuccess)                                                                         \
//     {                                                                                                          \
//         std::cerr << "MIOpen error: " << miopenGetErrorString(status) << " at line " << __LINE__ << std::endl; \
//         exit(1);                                                                                               \
//     }

#endif // UTILS_HPP