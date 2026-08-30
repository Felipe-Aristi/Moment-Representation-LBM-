#ifndef PDF_CUH
#define PDF_CUH

#include "constants.cuh"
#include "stencil.cuh"
#include "utilities/types.cuh"

// ============================================================
// c_i . u
// ============================================================

template <label_t I>
__device__ __forceinline__ real_t latticeCu(const real_t ux, const real_t uy, const real_t uz) noexcept
{
    constexpr int cx = D3Q27::cx<I>();
    constexpr int cy = D3Q27::cy<I>();
    constexpr int cz = D3Q27::cz<I>();

    real_t cu = static_cast<real_t>(0);

    if constexpr (cx == 1)
        cu += ux;
    else if constexpr (cx == -1)
        cu -= ux;

    if constexpr (cy == 1)
        cu += uy;
    else if constexpr (cy == -1)
        cu -= uy;

    if constexpr (cz == 1)
        cu += uz;
    else if constexpr (cz == -1)
        cu -= uz;

    return cu;
}

// ============================================================
// Equilibrium distribution
// ============================================================

template <label_t I>
__device__ __forceinline__ real_t feq(const real_t rho, const real_t ux, const real_t uy, const real_t uz, const real_t cu) noexcept
{
    constexpr real_t wi = D3Q27::w<I>();

    const real_t usq = ux * ux + uy * uy + uz * uz;
    const real_t cu2 = cu * cu;

    const real_t A12eq = cu * inv_cs2 - usq * inv_2cs2 + cu2 * inv_2cs4;
    const real_t A3eq = cu * (cu2 * inv_6cs6 - usq * inv_2cs4);

    return wi * rho * (static_cast<real_t>(1) + A12eq + A3eq);
}

// ============================================================
// Recursive-regularized non-equilibrium distribution
// ============================================================

template <label_t I>
__device__ __forceinline__ real_t fneqr(const real_t ux, const real_t uy, const real_t uz,
                                        const real_t pixx, const real_t pixy, const real_t pixz,
                                        const real_t piyy, const real_t piyz, const real_t pizz,
                                        const real_t cu) noexcept
{
    constexpr int cx = D3Q27::cx<I>();
    constexpr int cy = D3Q27::cy<I>();
    constexpr int cz = D3Q27::cz<I>();
    constexpr real_t wi = D3Q27::w<I>();

    // ========================================================
    // q = Pi . c
    // ========================================================

    real_t qx = static_cast<real_t>(0);
    real_t qy = static_cast<real_t>(0);
    real_t qz = static_cast<real_t>(0);

    if constexpr (cx == 1)
    {
        qx += pixx;
        qy += pixy;
        qz += pixz;
    }
    else if constexpr (cx == -1)
    {
        qx -= pixx;
        qy -= pixy;
        qz -= pixz;
    }

    if constexpr (cy == 1)
    {
        qx += pixy;
        qy += piyy;
        qz += piyz;
    }
    else if constexpr (cy == -1)
    {
        qx -= pixy;
        qy -= piyy;
        qz -= piyz;
    }

    if constexpr (cz == 1)
    {
        qx += pixz;
        qy += piyz;
        qz += pizz;
    }
    else if constexpr (cz == -1)
    {
        qx -= pixz;
        qy -= piyz;
        qz -= pizz;
    }

    // ========================================================
    // c . Pi . c
    // ========================================================

    real_t cPiC = static_cast<real_t>(0);

    if constexpr (cx == 1)
        cPiC += qx;
    else if constexpr (cx == -1)
        cPiC -= qx;

    if constexpr (cy == 1)
        cPiC += qy;
    else if constexpr (cy == -1)
        cPiC -= qy;

    if constexpr (cz == 1)
        cPiC += qz;
    else if constexpr (cz == -1)
        cPiC -= qz;

    // ========================================================
    // Second-order contribution
    //
    // A2neq = H^(2) : Pi
    // ========================================================

    const real_t tracePi = pixx + piyy + pizz;
    const real_t A2neq = cPiC - cs2 * tracePi;

    // ========================================================
    // c . Pi . u
    // ========================================================

    const real_t cPiU = qx * ux + qy * uy + qz * uz;

    // ========================================================
    // Third-order recursive contribution
    //
    // A3neq = cu A2neq - 2 cs2 c.Pi.u
    // ========================================================

    const real_t A3neq = cu * A2neq - static_cast<real_t>(2) * cs2 * cPiU;

    // ========================================================
    // Regularized non-equilibrium distribution
    // ========================================================

    return wi * (inv_2cs4 * A2neq + inv_2cs6 * A3neq);
}

#endif