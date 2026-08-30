#ifndef CUDA_UTILITIES_CUH
#define CUDA_UTILITIES_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// CUDA error checking
#define CUDA_CHECK(call)                                                        \
    do                                                                          \
    {                                                                           \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess)                                               \
        {                                                                       \
            fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",                    \
                    cudaGetErrorString(err__), (int)err__, __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#endif