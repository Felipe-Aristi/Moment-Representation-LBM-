#ifndef STENCILS_BOUNDSD3Q7_CUH
#define STENCILS_BOUNDSD3Q7_CUH

#include "d3q7.cuh"

#include "../src/constants.cuh"
#include "../src/utilities/indexing.cuh"
#include "../src/utilities/types.cuh"

// ============================================================
// D3Q7 periodic pull coordinates
// ============================================================

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullxD3Q7Peri(const label_t x) noexcept
{
    constexpr int cx = D3Q7::cx<I>();

    if constexpr (cx == 0)
        return x;
    else if constexpr (cx == 1)
        return x == static_cast<label_t>(1) ? NX - static_cast<label_t>(2) : x - static_cast<label_t>(1);
    else
        return x == NX - static_cast<label_t>(2) ? static_cast<label_t>(1) : x + static_cast<label_t>(1);
}

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullyD3Q7Peri(const label_t y) noexcept
{
    constexpr int cy = D3Q7::cy<I>();

    if constexpr (cy == 0)
        return y;
    else if constexpr (cy == 1)
        return y == static_cast<label_t>(1) ? NY - static_cast<label_t>(2) : y - static_cast<label_t>(1);
    else
        return y == NY - static_cast<label_t>(2) ? static_cast<label_t>(1) : y + static_cast<label_t>(1);
}

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullzD3Q7Peri(const label_t z) noexcept
{
    constexpr int cz = D3Q7::cz<I>();

    if constexpr (cz == 0)
        return z;
    else if constexpr (cz == 1)
        return z == static_cast<label_t>(1) ? NZ - static_cast<label_t>(2) : z - static_cast<label_t>(1);
    else
        return z == NZ - static_cast<label_t>(2) ? static_cast<label_t>(1) : z + static_cast<label_t>(1);
}

// ============================================================
// D3Q7 periodic XYZ pull index
// ============================================================

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullidD3Q7PeriXYZ(const label_t x, const label_t y, const label_t z) noexcept
{
    return idx(pullxD3Q7Peri<I>(x), pullyD3Q7Peri<I>(y), pullzD3Q7Peri<I>(z));
}

#endif