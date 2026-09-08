#ifndef MHD_MAGNETIC_MEMORY_CUH
#define MHD_MAGNETIC_MEMORY_CUH

#include <cuda_runtime.h>

#include "../../src/constants.cuh"
#include "../../src/utilities/types.cuh"
#include "../../src/utilities/cudaUtilities.cuh"

inline constexpr label_t NmagneticMoments = static_cast<label_t>(12);

inline constexpr label_t NmagneticOutputFields = static_cast<label_t>(3);

// ============================================================
// Magnetic memory sizes
// ============================================================

inline constexpr std::size_t bytesMagneticMoments = static_cast<std::size_t>(NmagneticMoments) * bytesField;

inline constexpr std::size_t bytesMagneticMomentsAB = static_cast<std::size_t>(2) * bytesMagneticMoments;

inline constexpr std::size_t bytesMagneticOutput = static_cast<std::size_t>(NmagneticOutputFields) * bytesField;

// ============================================================
// GPU magnetic state
// ============================================================

struct MagneticState
{
    real_t *data = nullptr;

    real_t *bx = nullptr;
    real_t *by = nullptr;
    real_t *bz = nullptr;

    real_t *axx = nullptr;
    real_t *axy = nullptr;
    real_t *axz = nullptr;

    real_t *ayx = nullptr;
    real_t *ayy = nullptr;
    real_t *ayz = nullptr;

    real_t *azx = nullptr;
    real_t *azy = nullptr;
    real_t *azz = nullptr;
};

// ============================================================
// Host magnetic output state
// ============================================================

struct HostMagneticState
{
    real_t *data = nullptr;

    real_t *bx = nullptr;
    real_t *by = nullptr;
    real_t *bz = nullptr;
};

// ============================================================
// Assign GPU SoA pointers
// ============================================================

inline void setMagneticPointers(MagneticState &state)
{
    state.bx = state.data + static_cast<std::size_t>(0) * fieldStride;
    state.by = state.data + static_cast<std::size_t>(1) * fieldStride;
    state.bz = state.data + static_cast<std::size_t>(2) * fieldStride;

    state.axx = state.data + static_cast<std::size_t>(3) * fieldStride;
    state.axy = state.data + static_cast<std::size_t>(4) * fieldStride;
    state.axz = state.data + static_cast<std::size_t>(5) * fieldStride;

    state.ayx = state.data + static_cast<std::size_t>(6) * fieldStride;
    state.ayy = state.data + static_cast<std::size_t>(7) * fieldStride;
    state.ayz = state.data + static_cast<std::size_t>(8) * fieldStride;

    state.azx = state.data + static_cast<std::size_t>(9) * fieldStride;
    state.azy = state.data + static_cast<std::size_t>(10) * fieldStride;
    state.azz = state.data + static_cast<std::size_t>(11) * fieldStride;
}

// ============================================================
// Assign host output pointers
// ============================================================

inline void setHostMagneticPointers(HostMagneticState &state)
{
    state.bx = state.data + static_cast<std::size_t>(0) * fieldStride;
    state.by = state.data + static_cast<std::size_t>(1) * fieldStride;
    state.bz = state.data + static_cast<std::size_t>(2) * fieldStride;
}

// ============================================================
// Host allocation
// ============================================================

inline void allocateHostMagneticState(HostMagneticState &state)
{
    state.data =
        new real_t[static_cast<std::size_t>(NmagneticOutputFields) *
                   fieldStride];

    setHostMagneticPointers(state);
}

// ============================================================
// Device allocation
// ============================================================

inline void allocateDeviceMagneticMoments(MagneticState &state)
{
    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void **>(&state.data),
            bytesMagneticMoments));

    setMagneticPointers(state);
}

// ============================================================
// Host deallocation
// ============================================================

inline void freeHostMagneticState(HostMagneticState &state)
{
    delete[] state.data;

    state = {};
}

// ============================================================
// Device deallocation
// ============================================================

inline void freeDeviceMagneticMoments(MagneticState &state)
{
    if (state.data != nullptr)
    {
        CUDA_CHECK(cudaFree(state.data));
    }

    state = {};
}

// ============================================================
// Device -> Host
//
// Bx + By + Bz only
// ============================================================

inline void copyMagneticDeviceToHost(
    HostMagneticState &host,
    const MagneticState &device)
{
    CUDA_CHECK(
        cudaMemcpy(
            host.data,
            device.data,
            bytesMagneticOutput,
            cudaMemcpyDeviceToHost));
}

// ============================================================
// Complete magnetic memory
// ============================================================

struct MagneticMemory
{
    HostMagneticState host;

    MagneticState d_A;
    MagneticState d_B;
};

// ============================================================
// Allocate complete magnetic memory
// ============================================================

inline void allocateMagneticMemory(MagneticMemory &memory)
{
    allocateHostMagneticState(memory.host);

    allocateDeviceMagneticMoments(memory.d_A);
    allocateDeviceMagneticMoments(memory.d_B);
}

// ============================================================
// Free complete magnetic memory
// ============================================================

inline void freeMagneticMemory(MagneticMemory &memory)
{
    freeHostMagneticState(memory.host);

    freeDeviceMagneticMoments(memory.d_A);
    freeDeviceMagneticMoments(memory.d_B);
}

#endif