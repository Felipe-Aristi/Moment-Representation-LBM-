#ifndef LAUNCH_CUH
#define LAUNCH_CUH

#include "kernels.cuh"
#include "memory.cuh"

#include "utilities/cudaConfig.cuh"
#include "utilities/cudaUtilities.cuh"

// ============================================================
// Taylor-Green initialization
// ============================================================

__host__ __forceinline__ void launchTaylorGreenInitialization(const CudaConfig &cfg, MomentState &state) noexcept
{
    initializeTaylorGreenKernel<<<cfg.grid, cfg.block>>>(state);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Jet initialization
// ============================================================

__host__ __forceinline__ void launchJetInitialization(const CudaConfig &cfg, MomentState &state) noexcept
{

    initializeJetBulkKernel<<<cfg.grid, cfg.block>>>(state);

    CUDA_CHECK(cudaGetLastError());

    initializeJetInletKernel<<<grid2D_xz(cfg), block2D_xz(cfg)>>>(state);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Jet boundary conditions
// ============================================================

__host__ __forceinline__ void launchJetBoundaryConditions(const CudaConfig &cfg, const MomentState &A, MomentState &B) noexcept
{
    jetBoundaryConditionsKernel<<<grid2D_xz(cfg), block2D_xz(cfg)>>>(A, B);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Channel-flow initialization
// ============================================================

__host__ __forceinline__ void launchChannelFlowInitialization(const CudaConfig &cfg, MomentState &state) noexcept
{
    initializeChannelFlowKernel<<<cfg.grid, cfg.block>>>(state);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Channel-flow boundary conditions
// ============================================================

__host__ __forceinline__ void launchChannelBoundaryConditions(const CudaConfig &cfg, const MomentState &A, MomentState &B) noexcept
{
    channelBoundaryConditionsKernel<<<grid2D_xz(cfg), block2D_xz(cfg)>>>(A, B);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// MR-LBM kernel launch
// ============================================================

__host__ __forceinline__ void launchLBM(const CudaConfig &cfg, const MomentState &A, MomentState &B) noexcept
{
    lbmKernel<<<cfg.grid, cfg.block>>>(A, B);

    CUDA_CHECK(cudaGetLastError());
}

__host__ __forceinline__ void launchLBMF(const CudaConfig &cfg, const MomentState &A, MomentState &B) noexcept
{
    lbmfKernel<<<cfg.grid, cfg.block>>>(A, B);

    CUDA_CHECK(cudaGetLastError());
}

#endif