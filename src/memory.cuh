#ifndef MEMORY_CUH
#define MEMORY_CUH

#include <cuda_runtime.h>

#include "constants.cuh"
#include "utilities/types.cuh"
#include "utilities/cudaUtilities.cuh"

// ============================================================
// GPU moment state
// ============================================================

struct MomentState
{
    real_t *data = nullptr;

    real_t *rho = nullptr;

    real_t *ux = nullptr;
    real_t *uy = nullptr;
    real_t *uz = nullptr;

    real_t *pixx = nullptr;
    real_t *pixy = nullptr;
    real_t *pixz = nullptr;
    real_t *piyy = nullptr;
    real_t *piyz = nullptr;
    real_t *pizz = nullptr;
};

// ============================================================
// Host output state
// ============================================================

struct HostState
{
    real_t *data = nullptr;

    real_t *rho = nullptr;

    real_t *ux = nullptr;
    real_t *uy = nullptr;
    real_t *uz = nullptr;
};

// ============================================================
// Assign GPU SoA pointers
// ============================================================

inline void setMomentPointers(MomentState &state)
{
    state.rho = state.data + static_cast<std::size_t>(0) * fieldStride;

    state.ux = state.data + static_cast<std::size_t>(1) * fieldStride;
    state.uy = state.data + static_cast<std::size_t>(2) * fieldStride;
    state.uz = state.data + static_cast<std::size_t>(3) * fieldStride;

    state.pixx = state.data + static_cast<std::size_t>(4) * fieldStride;
    state.pixy = state.data + static_cast<std::size_t>(5) * fieldStride;
    state.pixz = state.data + static_cast<std::size_t>(6) * fieldStride;

    state.piyy = state.data + static_cast<std::size_t>(7) * fieldStride;
    state.piyz = state.data + static_cast<std::size_t>(8) * fieldStride;
    state.pizz = state.data + static_cast<std::size_t>(9) * fieldStride;
}

// ============================================================
// Assign host output pointers
// ============================================================

inline void setHostPointers(HostState &state)
{
    state.rho = state.data + static_cast<std::size_t>(0) * fieldStride;

    state.ux = state.data + static_cast<std::size_t>(1) * fieldStride;
    state.uy = state.data + static_cast<std::size_t>(2) * fieldStride;
    state.uz = state.data + static_cast<std::size_t>(3) * fieldStride;
}

// ============================================================
// Host allocation
// ============================================================

inline void allocateHostState(HostState &state)
{
    state.data = new real_t[static_cast<std::size_t>(NoutputFields) * fieldStride];

    setHostPointers(state);
}

// ============================================================
// Device allocation
// ============================================================

inline void allocateDeviceMoments(MomentState &state)
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.data), bytesMoments));

    setMomentPointers(state);
}

// ============================================================
// Host deallocation
// ============================================================

inline void freeHostState(HostState &state)
{
    delete[] state.data;

    state = {};
}

// ============================================================
// Device deallocation
// ============================================================

inline void freeDeviceMoments(MomentState &state)
{
    if (state.data != nullptr)
    {
        CUDA_CHECK(cudaFree(state.data));
    }

    state = {};
}

// ============================================================
// Device -> Host
// rho + ux + uy + uz only
// ============================================================

inline void copyDeviceToHost(HostState &host, const MomentState &device)
{
    CUDA_CHECK(cudaMemcpy(host.data, device.data, bytesOutput, cudaMemcpyDeviceToHost));
}

// ============================================================
// Complete memory structure
// ============================================================

struct Memory
{
    HostState host;

    MomentState d_A;
    MomentState d_B;
};

// ============================================================
// Allocate complete memory
// ============================================================

inline void allocateMemory(Memory &memory)
{
    allocateHostState(memory.host);

    allocateDeviceMoments(memory.d_A);
    allocateDeviceMoments(memory.d_B);
}

// ============================================================
// Free complete memory
// ============================================================

inline void freeMemory(Memory &memory)
{
    freeHostState(memory.host);

    freeDeviceMoments(memory.d_A);
    freeDeviceMoments(memory.d_B);
}

#endif // MEMORY_CUH