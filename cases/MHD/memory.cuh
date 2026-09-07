#ifndef MHD_MEMORY_CUH
#define MHD_MEMORY_CUH

#include "../../src/memory.cuh"
#include "stencils/d3q7.cuh"

namespace mhd
{
// Per time level: 10 fluid fields + 3 cached B fields + 21 magnetic populations.
inline constexpr label_t magneticFields = 3 + 3 * D3Q7::q;
inline constexpr std::size_t magneticBytes = magneticFields * bytesField;

struct State
{
    MomentState fluid;
    real_t *data = nullptr;
    real_t *bx = nullptr;
    real_t *by = nullptr;
    real_t *bz = nullptr;
    real_t *g = nullptr;
};

inline void setPointers(State &s)
{
    s.bx = s.data;
    s.by = s.data + fieldStride;
    s.bz = s.data + 2 * fieldStride;
    s.g = s.data + 3 * fieldStride;
}

template <label_t I, label_t A>
__host__ __device__ constexpr std::size_t populationIndex(const label_t id) noexcept
{
    static_assert(I < D3Q7::q && A < 3);
    return (A * D3Q7::q + I) * fieldStride + id;
}

inline void allocate(State &s)
{
    allocateDeviceMoments(s.fluid);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&s.data), magneticBytes));
    setPointers(s);
    CUDA_CHECK(cudaMemset(s.fluid.data, 0, bytesMoments));
    CUDA_CHECK(cudaMemset(s.data, 0, magneticBytes));
}

inline void release(State &s)
{
    freeDeviceMoments(s.fluid);
    if (s.data) CUDA_CHECK(cudaFree(s.data));
    s = {};
}
}
#endif
