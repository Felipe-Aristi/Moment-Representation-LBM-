#ifndef FORCE_CUH
#define FORCE_CUH

#include "constants.cuh"
#include "stencil.cuh"
#include "utilities/types.cuh"

template <label_t I>
__device__ __forceinline__ real_t force(const real_t ux, const real_t uy, const real_t uz,
                                        const real_t Fx, const real_t Fy, const real_t Fz,
                                        const real_t cu) noexcept
{
    constexpr int cx = D3Q27::cx<I>();
    constexpr int cy = D3Q27::cy<I>();
    constexpr int cz = D3Q27::cz<I>();
    constexpr real_t wi = D3Q27::w<I>();

    // ========================================================
    // c . F
    // ========================================================

    real_t cF = static_cast<real_t>(0);

    if constexpr (cx == 1)
        cF += Fx;
    else if constexpr (cx == -1)
        cF -= Fx;

    if constexpr (cy == 1)
        cF += Fy;
    else if constexpr (cy == -1)
        cF -= Fy;

    if constexpr (cz == 1)
        cF += Fz;
    else if constexpr (cz == -1)
        cF -= Fz;

    const real_t uF = ux * Fx + uy * Fy + uz * Fz;

    const real_t usq = ux * ux + uy * uy + uz * uz;

    const real_t A1force = cF;

    // ========================================================
    // Second-order forcing contribution
    // A2force = H^(2) : (uF + Fu) = 2 [ cu cF - cs2 uF ]
    // ========================================================

    const real_t A2force = static_cast<real_t>(2) * (cu * cF - cs2 * uF);

    // ========================================================
    // Third-order forcing contribution
    // A3force = H^(3) : (uuF + uFu + Fuu) = 3 [cu^2 cF - cs2 (2 cu uF + usq cF)]
    // ========================================================

    const real_t A3force = static_cast<real_t>(3) *
                           (cu * cu * cF - cs2 * (static_cast<real_t>(2) * cu * uF + usq * cF));

    // ========================================================
    // Third-order forcing reconstruction
    // ========================================================

    return wi * (inv_cs2 * A1force + inv_2cs4 * A2force + inv_6cs6 * A3force);
}

#endif