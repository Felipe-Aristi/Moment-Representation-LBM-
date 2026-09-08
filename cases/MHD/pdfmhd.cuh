#ifndef MHD_PDFMHD_CUH
#define MHD_PDFMHD_CUH

#include "constantsmhd.cuh"

#include "../../stencils/d3q7.cuh"
#include "../../src/stencil.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// Magnetic vector population
// ============================================================

struct MagneticPDF
{
    real_t x;
    real_t y;
    real_t z;
};

// ============================================================
// c_i . u
// ============================================================

template <label_t I>
__device__ __forceinline__ real_t magneticCu(const real_t ux, const real_t uy, const real_t uz) noexcept
{
    constexpr int cx = D3Q7::cx<I>();
    constexpr int cy = D3Q7::cy<I>();
    constexpr int cz = D3Q7::cz<I>();

    real_t cu = static_cast<real_t>(0);

    if constexpr (cx == 1)
    {
        cu += ux;
    }
    else if constexpr (cx == -1)
    {
        cu -= ux;
    }

    if constexpr (cy == 1)
    {
        cu += uy;
    }
    else if constexpr (cy == -1)
    {
        cu -= uy;
    }

    if constexpr (cz == 1)
    {
        cu += uz;
    }
    else if constexpr (cz == -1)
    {
        cu -= uz;
    }

    return cu;
}

// ============================================================
// c_i . B
// ============================================================

template <label_t I>
__device__ __forceinline__ real_t magneticCB(const real_t bx, const real_t by, const real_t bz) noexcept
{
    constexpr int cx = D3Q7::cx<I>();
    constexpr int cy = D3Q7::cy<I>();
    constexpr int cz = D3Q7::cz<I>();

    real_t cB = static_cast<real_t>(0);

    if constexpr (cx == 1)
    {
        cB += bx;
    }
    else if constexpr (cx == -1)
    {
        cB -= bx;
    }

    if constexpr (cy == 1)
    {
        cB += by;
    }
    else if constexpr (cy == -1)
    {
        cB -= by;
    }

    if constexpr (cz == 1)
    {
        cB += bz;
    }
    else if constexpr (cz == -1)
    {
        cB -= bz;
    }

    return cB;
}

// ============================================================
// Magnetic equilibrium distribution
// g_i_beta^eq =  w_i [ B_beta + (1 / cs_m^2) ((c_i . u) B_beta - (c_i . B) u_beta)]
// ============================================================

template <label_t I>
__device__ __forceinline__ MagneticPDF geqmhd(const real_t ux, const real_t uy, const real_t uz, const real_t bx, const real_t by, const real_t bz, const real_t cu, const real_t cB) noexcept
{
    constexpr real_t wi = D3Q7::w<I>();

    MagneticPDF g{};

    g.x = wi * (bx + d3q7::inv_cs2 * (cu * bx - cB * ux));
    g.y = wi * (by + d3q7::inv_cs2 * (cu * by - cB * uy));
    g.z = wi * (bz + d3q7::inv_cs2 * (cu * bz - cB * uz));

    return g;
}

// ============================================================
// Regularized magnetic non-equilibrium distribution
// g_i_beta^neq,R = w_i / cs_m^2 * c_i_alpha A_alpha_beta^neq
// ============================================================

template <label_t I>
__device__ __forceinline__ MagneticPDF gneqrmhd(const real_t axx, const real_t axy, const real_t axz,
                                                const real_t ayx, const real_t ayy, const real_t ayz,
                                                const real_t azx, const real_t azy, const real_t azz) noexcept
{
    constexpr int cx = D3Q7::cx<I>();
    constexpr int cy = D3Q7::cy<I>();
    constexpr int cz = D3Q7::cz<I>();

    constexpr real_t wi = D3Q7::w<I>();

    real_t cAx = static_cast<real_t>(0);
    real_t cAy = static_cast<real_t>(0);
    real_t cAz = static_cast<real_t>(0);

    // ========================================================
    // x lattice direction
    //
    // c_x A_x_beta
    // ========================================================

    if constexpr (cx == 1)
    {
        cAx += axx;
        cAy += axy;
        cAz += axz;
    }
    else if constexpr (cx == -1)
    {
        cAx -= axx;
        cAy -= axy;
        cAz -= axz;
    }

    // ========================================================
    // y lattice direction
    //
    // c_y A_y_beta
    // ========================================================

    if constexpr (cy == 1)
    {
        cAx += ayx;
        cAy += ayy;
        cAz += ayz;
    }
    else if constexpr (cy == -1)
    {
        cAx -= ayx;
        cAy -= ayy;
        cAz -= ayz;
    }

    // ========================================================
    // z lattice direction
    //
    // c_z A_z_beta
    // ========================================================

    if constexpr (cz == 1)
    {
        cAx += azx;
        cAy += azy;
        cAz += azz;
    }
    else if constexpr (cz == -1)
    {
        cAx -= azx;
        cAy -= azy;
        cAz -= azz;
    }

    constexpr real_t factor = wi * d3q7::inv_cs2;

    MagneticPDF g{};

    g.x = factor * cAx;
    g.y = factor * cAy;
    g.z = factor * cAz;

    return g;
}

// ========================================================
// fieqmhd = w_i / (2 cs_f^4 mu_0) [ 0.5 B^2 (c_i^2 - cs_f^2) - (c_i . B)^2 ]
// ========================================================

template <label_t I>
__device__ __forceinline__ real_t feqmhd(const real_t ux, const real_t uy, const real_t uz,
                                         const real_t bx, const real_t by, const real_t bz,
                                         const real_t cu, const real_t cB) noexcept
{
    constexpr real_t wi = D3Q27::w<I>();
    constexpr real_t c2 = static_cast<real_t>(D3Q27::c2<I>());

    const real_t B2 = bx * bx + by * by + bz * bz;
    const real_t uB = ux * bx + uy * by + uz * bz;

    const real_t A2mhd = mhd_inv_mu0 * (static_cast<real_t>(0.5) * B2 * (c2 - cs2) - cB * cB);
    const real_t cMu = mhd_inv_mu0 * (static_cast<real_t>(0.5) * B2 * cu - cB * uB);
    const real_t A3mhd = cu * A2mhd - static_cast<real_t>(2) * cs2 * cMu;

    return wi * (inv_2cs4 * A2mhd + inv_2cs6 * A3mhd);
}

#endif