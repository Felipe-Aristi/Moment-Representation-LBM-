#ifndef CUDACONFIG_CUH
#define CUDACONFIG_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>

#include "../constants.cuh"
#include "cudaUtilities.cuh"

// ============================================================
// Ceiling division
// ============================================================

__host__ __device__ [[nodiscard]] constexpr __forceinline__ std::size_t ceil_div(const std::size_t a, const std::size_t b) noexcept
{
    return (a + b - static_cast<std::size_t>(1)) / b;
}

// ============================================================
// CUDA configuration
// ============================================================

struct CudaConfig
{
    dim3 block;
    dim3 grid;
};

// ============================================================
// 3D configuration
// ============================================================

__host__ __forceinline__ CudaConfig make_cudaConfig() noexcept
{
    CudaConfig cfg{};

    cfg.block = dim3(blockX, blockY, blockZ);

    cfg.grid = dim3(ceil_div(static_cast<std::size_t>(NX), static_cast<std::size_t>(cfg.block.x)),
                    ceil_div(static_cast<std::size_t>(NY), static_cast<std::size_t>(cfg.block.y)),
                    ceil_div(static_cast<std::size_t>(NZ), static_cast<std::size_t>(cfg.block.z)));

    return cfg;
}

// ============================================================
// 2D XZ configuration
// ============================================================

// Uses block.x and block.y as the x and z dimensions.

__host__ __device__ __forceinline__ dim3 block2D_xz(const CudaConfig &cfg) noexcept
{
    return dim3(cfg.block.x, cfg.block.y, static_cast<unsigned int>(1));
}

__host__ __device__ __forceinline__ dim3 grid2D_xz(const CudaConfig &cfg) noexcept
{
    const dim3 block = block2D_xz(cfg);

    return dim3(ceil_div(static_cast<std::size_t>(NX), static_cast<std::size_t>(block.x)),
                ceil_div(static_cast<std::size_t>(NZ), static_cast<std::size_t>(block.y)),
                static_cast<unsigned int>(1));
}

// ============================================================
// GPU and mesh information
// ============================================================

inline CudaConfig print_device_and_make_config(const int deviceID)
{
    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceID));

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";

    CudaConfig cfg = make_cudaConfig();

    std::cout << "block=(" << cfg.block.x << "," << cfg.block.y << "," << cfg.block.z << ")\n";
    std::cout << "grid =(" << cfg.grid.x << "," << cfg.grid.y << "," << cfg.grid.z << ")\n";

    return cfg;
}

#endif