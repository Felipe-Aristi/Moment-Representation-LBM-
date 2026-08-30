#ifndef BOUNDS_CUH
#define BOUNDS_CUH

#include "../constants.cuh"
#include "../stencil.cuh"
#include "indexing.cuh"
#include "types.cuh"

// ============================================================
// Physical domain
// ============================================================

// Returns true for nodes outside the physical fluid interior,
// including the ghost/boundary layers.

__host__ __device__ [[nodiscard]] constexpr __forceinline__ bool interior(const label_t x, const label_t y, const label_t z) noexcept
{
    return (x >= NX || y >= NY || z >= NZ ||
            x == static_cast<label_t>(0) || x == NX - static_cast<label_t>(1) ||
            y == static_cast<label_t>(0) || y == NY - static_cast<label_t>(1) ||
            z == static_cast<label_t>(0) || z == NZ - static_cast<label_t>(1));
}

// Used for inlet/outlet planes when only x and z determine
// whether the node belongs to the physical fluid cross-section.

__host__ __device__ [[nodiscard]] constexpr __forceinline__ bool inlet_outlet_interior(const label_t x, const label_t z) noexcept
{
    return (x >= NX || z >= NZ ||
            x == static_cast<label_t>(0) || x == NX - static_cast<label_t>(1) ||
            z == static_cast<label_t>(0) || z == NZ - static_cast<label_t>(1));
}

// ============================================================
// Periodic boundary conditions
// ============================================================

__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t wrapx(const label_t x) noexcept
{
    if (x == static_cast<label_t>(0))
    {
        return NX - static_cast<label_t>(2);
    }

    if (x == NX - static_cast<label_t>(1))
    {
        return static_cast<label_t>(1);
    }

    return x;
}

__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t wrapy(const label_t y) noexcept
{
    if (y == static_cast<label_t>(0))
    {
        return NY - static_cast<label_t>(2);
    }

    if (y == NY - static_cast<label_t>(1))
    {
        return static_cast<label_t>(1);
    }

    return y;
}

__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t wrapz(const label_t z) noexcept
{
    if (z == static_cast<label_t>(0))
    {
        return NZ - static_cast<label_t>(2);
    }

    if (z == NZ - static_cast<label_t>(1))
    {
        return static_cast<label_t>(1);
    }

    return z;
}

// ============================================================
// Coordinates from linear index
// ============================================================

__host__ __device__ constexpr __forceinline__ void coordinates(const label_t id, label_t &x, label_t &y, label_t &z) noexcept
{
    z = id / NXNY;

    const label_t rem = id - z * NXNY;

    y = rem / NX;
    x = rem - y * NX;
}

// ============================================================
// Pull coordinates
// ============================================================

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullx(const label_t x) noexcept
{
    return static_cast<label_t>(static_cast<int>(x) - D3Q27::cx<I>());
}

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pully(const label_t y) noexcept
{
    return static_cast<label_t>(static_cast<int>(y) - D3Q27::cy<I>());
}

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullz(const label_t z) noexcept
{
    return static_cast<label_t>(static_cast<int>(z) - D3Q27::cz<I>());
}

// ============================================================
// Compile-time periodic pull coordinates
// ============================================================

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullxPeri(const label_t x) noexcept
{
    constexpr int cx = D3Q27::cx<I>();

    if constexpr (cx == 0)
    {
        return x;
    }
    else if constexpr (cx == 1)
    {
        return x == static_cast<label_t>(1) ? NX - static_cast<label_t>(2) : x - static_cast<label_t>(1);
    }
    else
    {
        return x == NX - static_cast<label_t>(2) ? static_cast<label_t>(1) : x + static_cast<label_t>(1);
    }
}

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullyPeri(const label_t y) noexcept
{
    constexpr int cy = D3Q27::cy<I>();

    if constexpr (cy == 0)
    {
        return y;
    }
    else if constexpr (cy == 1)
    {
        return y == static_cast<label_t>(1) ? NY - static_cast<label_t>(2) : y - static_cast<label_t>(1);
    }
    else
    {
        return y == NY - static_cast<label_t>(2) ? static_cast<label_t>(1) : y + static_cast<label_t>(1);
    }
}

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullzPeri(const label_t z) noexcept
{
    constexpr int cz = D3Q27::cz<I>();

    if constexpr (cz == 0)
    {
        return z;
    }
    else if constexpr (cz == 1)
    {
        return z == static_cast<label_t>(1) ? NZ - static_cast<label_t>(2) : z - static_cast<label_t>(1);
    }
    else
    {
        return z == NZ - static_cast<label_t>(2) ? static_cast<label_t>(1) : z + static_cast<label_t>(1);
    }
}

// ============================================================
// Pull index
// Channel flow
// ============================================================

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullidPeriXY(const label_t x, const label_t y, const label_t z) noexcept
{
    return idx(pullxPeri<I>(x), pullyPeri<I>(y), pullz<I>(z));
}

// ============================================================
// Pull index - periodic XYZ
// Taylor-Green vortex
// ============================================================

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullidPeriXYZ(const label_t x, const label_t y, const label_t z) noexcept
{
    return idx(pullxPeri<I>(x), pullyPeri<I>(y), pullzPeri<I>(z));
}

// ============================================================
// Pull index - periodic XZ
// turbulent jet
// ============================================================

template <label_t I>
__host__ __device__ [[nodiscard]] constexpr __forceinline__ label_t pullidPeriXZ(const label_t x, const label_t y, const label_t z) noexcept
{
    return idx(pullxPeri<I>(x), pully<I>(y), pullzPeri<I>(z));
}

#endif