#ifndef STENCILS_D3Q7_CUH
#define STENCILS_D3Q7_CUH

#include "../src/utilities/types.cuh"

// ============================================================
// D3Q7 constants
// ============================================================

namespace d3q7
{

    inline constexpr label_t Q = static_cast<label_t>(7);

    inline constexpr int cx[Q] =
        {
            0,
            1, -1,
            0, 0,
            0, 0};

    inline constexpr int cy[Q] =
        {
            0,
            0, 0,
            1, -1,
            0, 0};

    inline constexpr int cz[Q] =
        {
            0,
            0, 0,
            0, 0,
            1, -1};

    inline constexpr real_t w0 = static_cast<real_t>(1.0 / 4.0);
    inline constexpr real_t w1 = static_cast<real_t>(1.0 / 8.0);

    inline constexpr real_t cs2 = static_cast<real_t>(1.0 / 4.0);
    inline constexpr real_t inv_cs2 = static_cast<real_t>(1) / cs2;

} // namespace d3q7

// ============================================================
// D3Q7 compile-time interface
// ============================================================

struct D3Q7
{
    static constexpr label_t Q = d3q7::Q;

    template <label_t I>
    __host__ __device__ static constexpr int cx() noexcept
    {
        static_assert(I < Q);

        return d3q7::cx[I];
    }

    template <label_t I>
    __host__ __device__ static constexpr int cy() noexcept
    {
        static_assert(I < Q);

        return d3q7::cy[I];
    }

    template <label_t I>
    __host__ __device__ static constexpr int cz() noexcept
    {
        static_assert(I < Q);

        return d3q7::cz[I];
    }

    template <label_t I>
    __host__ __device__ static constexpr int c2() noexcept
    {
        constexpr int cxi = cx<I>();
        constexpr int cyi = cy<I>();
        constexpr int czi = cz<I>();

        return cxi * cxi + cyi * cyi + czi * czi;
    }

    template <label_t I>
    __host__ __device__ static constexpr real_t w() noexcept
    {
        if constexpr (I == static_cast<label_t>(0))
        {
            return d3q7::w0;
        }
        else
        {
            return d3q7::w1;
        }
    }
};

// ============================================================
// Compile-time consistency checks
// ============================================================

static_assert(D3Q7::Q == static_cast<label_t>(7));

static_assert(D3Q7::cx<1>() == 1);
static_assert(D3Q7::cx<2>() == -1);

static_assert(D3Q7::cy<3>() == 1);
static_assert(D3Q7::cy<4>() == -1);

static_assert(D3Q7::cz<5>() == 1);
static_assert(D3Q7::cz<6>() == -1);

static_assert(
    d3q7::w0 +
        static_cast<real_t>(6) * d3q7::w1 ==
    static_cast<real_t>(1));

static_assert(
    static_cast<real_t>(2) * d3q7::w1 ==
    d3q7::cs2);

#endif