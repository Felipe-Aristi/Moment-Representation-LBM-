#ifndef MHD_LAUNCH_CUH
#define MHD_LAUNCH_CUH

#include "kernels.cuh"
#include "../../src/utilities/cudaConfig.cuh"

namespace mhd
{
inline void initialize(const CudaConfig &cfg, const State &s)
{
    initializeKernel<<<cfg.grid, cfg.block>>>(s);
    CUDA_CHECK(cudaGetLastError());
}
inline void advance(const CudaConfig &cfg, const State &a, const State &b)
{
    advanceKernel<<<cfg.grid, cfg.block>>>(a, b);
    CUDA_CHECK(cudaGetLastError());
}
}
#endif
