#include "kernels.cuh"
#include "initialization.cuh"

namespace mhd
{
__global__ void initializeKernel(State state)
{
    const label_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const label_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const label_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;
    initializeNode(x, y, z, state);
}

__global__ void advanceKernel(const __grid_constant__ State a, const __grid_constant__ State b)
{
    const label_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const label_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const label_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (interior(x, y, z)) return;
    advanceNode(x, y, z, a, b);
}
}
