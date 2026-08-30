#ifndef STENCIL_CUH
#define STENCIL_CUH

#include "constants.cuh"
#include "utilities/types.cuh"

// ============================================================
// D3Q27 constants
// ============================================================

namespace d3q27
{

    inline constexpr int cx[Q] =
        {
            0,
            1, -1, 0, 0, 0, 0,
            1, -1, 1, -1,
            1, -1, 1, -1,
            0, 0, 0, 0,
            1, -1, 1, -1, 1, -1, 1, -1};

    inline constexpr int cy[Q] =
        {
            0,
            0, 0, 1, -1, 0, 0,
            1, -1, -1, 1,
            0, 0, 0, 0,
            1, -1, 1, -1,
            1, -1, 1, -1, -1, 1, -1, 1};

    inline constexpr int cz[Q] =
        {
            0,
            0, 0, 0, 0, 1, -1,
            0, 0, 0, 0,
            1, -1, -1, 1,
            1, -1, -1, 1,
            1, -1, -1, 1, 1, -1, -1, 1};

    inline constexpr real_t w0 = static_cast<real_t>(8.0 / 27.0);
    inline constexpr real_t w1 = static_cast<real_t>(2.0 / 27.0);
    inline constexpr real_t w2 = static_cast<real_t>(1.0 / 54.0);
    inline constexpr real_t w3 = static_cast<real_t>(1.0 / 216.0);

} // namespace d3q27

// ============================================================
// D3Q27 compile-time interface
// ============================================================

struct D3Q27
{
    template <label_t I>
    __host__ __device__ static constexpr int cx() noexcept
    {
        static_assert(I < Q);

        return d3q27::cx[I];
    }

    template <label_t I>
    __host__ __device__ static constexpr int cy() noexcept
    {
        static_assert(I < Q);

        return d3q27::cy[I];
    }

    template <label_t I>
    __host__ __device__ static constexpr int cz() noexcept
    {
        static_assert(I < Q);

        return d3q27::cz[I];
    }

    // ========================================================
    // |c_i|^2
    // ========================================================

    template <label_t I>
    __host__ __device__ static constexpr int c2() noexcept
    {
        constexpr int cxi = cx<I>();
        constexpr int cyi = cy<I>();
        constexpr int czi = cz<I>();

        return cxi * cxi + cyi * cyi + czi * czi;
    }

    // ========================================================
    // Lattice weights
    // ========================================================

    template <label_t I>
    __host__ __device__ static constexpr real_t w() noexcept
    {
        constexpr int c2i = c2<I>();

        if constexpr (c2i == 0)
        {
            return d3q27::w0;
        }
        else if constexpr (c2i == 1)
        {
            return d3q27::w1;
        }
        else if constexpr (c2i == 2)
        {
            return d3q27::w2;
        }
        else
        {
            return d3q27::w3;
        }
    }

    // ========================================================
    // Second-order Hermite polynomials
    // ========================================================

    template <label_t I>
    __host__ __device__ static constexpr real_t Hxx() noexcept
    {
        constexpr real_t cxi = static_cast<real_t>(cx<I>());

        return cxi * cxi - cs2;
    }

    template <label_t I>
    __host__ __device__ static constexpr real_t Hxy() noexcept
    {
        return static_cast<real_t>(cx<I>() * cy<I>());
    }

    template <label_t I>
    __host__ __device__ static constexpr real_t Hxz() noexcept
    {
        return static_cast<real_t>(cx<I>() * cz<I>());
    }

    template <label_t I>
    __host__ __device__ static constexpr real_t Hyy() noexcept
    {
        constexpr real_t cyi = static_cast<real_t>(cy<I>());

        return cyi * cyi - cs2;
    }

    template <label_t I>
    __host__ __device__ static constexpr real_t Hyz() noexcept
    {
        return static_cast<real_t>(cy<I>() * cz<I>());
    }

    template <label_t I>
    __host__ __device__ static constexpr real_t Hzz() noexcept
    {
        constexpr real_t czi = static_cast<real_t>(cz<I>());

        return czi * czi - cs2;
    }
};

#endif