#ifndef MHD_KERNELS_CUH
#define MHD_KERNELS_CUH
#include "memory.cuh"

namespace mhd
{
__global__ void initializeKernel(State state);
__global__ void advanceKernel(const __grid_constant__ State a, const __grid_constant__ State b);
}
#endif
