#ifndef MHD_LBMMHD_CUH
#define MHD_LBMMHD_CUH

#include "constantsmhd.cuh"
#include "magneticMemory.cuh"
#include "pdfmhd.cuh"

#include "../../src/lbm.cuh"
#include "../../src/memory.cuh"
#include "../../src/pdf.cuh"

#include "../../src/utilities/bounds.cuh"
#include "../../src/utilities/constexprFor.cuh"
#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// Magnetic moment accumulator
// ============================================================

struct MagneticMomentAccumulator
{
    real_t bx = static_cast<real_t>(0);
    real_t by = static_cast<real_t>(0);
    real_t bz = static_cast<real_t>(0);

    real_t axx = static_cast<real_t>(0);
    real_t axy = static_cast<real_t>(0);
    real_t axz = static_cast<real_t>(0);

    real_t ayx = static_cast<real_t>(0);
    real_t ayy = static_cast<real_t>(0);
    real_t ayz = static_cast<real_t>(0);

    real_t azx = static_cast<real_t>(0);
    real_t azy = static_cast<real_t>(0);
    real_t azz = static_cast<real_t>(0);
};

// ============================================================
// Local magnetic state
// ============================================================

struct LocalMagneticState
{
    real_t bx;
    real_t by;
    real_t bz;

    real_t axx;
    real_t axy;
    real_t axz;

    real_t ayx;
    real_t ayy;
    real_t ayz;

    real_t azx;
    real_t azy;
    real_t azz;
};

template <label_t I>
__device__ __forceinline__ real_t fstarMHD(const real_t rho, const real_t ux, const real_t uy, const real_t uz,
                                           const real_t pixx, const real_t pixy, const real_t pixz,
                                           const real_t piyy, const real_t piyz, const real_t pizz,
                                           const real_t bx, const real_t by, const real_t bz,
                                           const real_t cu, const real_t cB) noexcept
{
    const real_t fieq = feq<I>(rho, ux, uy, uz, cu);
    const real_t fieqmhd = feqmhd<I>(ux, uy, uz, bx, by, bz, cu, cB);
    const real_t fineq = fneqr<I>(ux, uy, uz, pixx, pixy, pixz, piyy, piyz, pizz, cu);

    return fieq + fieqmhd + mhd_omsF  * fineq;
}


// ============================================================
// Magnetic non-equilibrium source reconstruction
// ============================================================

template <label_t I>
__device__ __forceinline__ MagneticPDF magneticGneqSource(const MagneticState &state, const label_t id) noexcept
{
    constexpr int cx = D3Q7::cx<I>();
    constexpr int cy = D3Q7::cy<I>();
    constexpr int cz = D3Q7::cz<I>();

    if constexpr (cx != 0)
    {
        return gneqrmhd<I>(state.axx[id], state.axy[id], state.axz[id],
                           static_cast<real_t>(0), static_cast<real_t>(0), static_cast<real_t>(0),
                           static_cast<real_t>(0), static_cast<real_t>(0), static_cast<real_t>(0));
    }
    else if constexpr (cy != 0)
    {
        return gneqrmhd<I>(static_cast<real_t>(0), static_cast<real_t>(0), static_cast<real_t>(0),
                           state.ayx[id], state.ayy[id], state.ayz[id],
                           static_cast<real_t>(0), static_cast<real_t>(0), static_cast<real_t>(0));
    }
    else if constexpr (cz != 0)
    {
        return gneqrmhd<I>(static_cast<real_t>(0), static_cast<real_t>(0), static_cast<real_t>(0),
                           static_cast<real_t>(0), static_cast<real_t>(0), static_cast<real_t>(0),
                           state.azx[id], state.azy[id], state.azz[id]);
    }
    else
    {
        return {};
    }
}


template <label_t I>
__device__ __forceinline__ MagneticPDF gstarMHD(const real_t ux, const real_t uy, const real_t uz,
                                                const real_t bx, const real_t by, const real_t bz,
                                                const real_t cu, const real_t cB,
                                                const MagneticState &magneticA,
                                                const label_t pullID) noexcept
{
    const MagneticPDF geq = geqmhd<I>(ux, uy, uz, bx, by, bz, cu, cB);
    const MagneticPDF gneq = magneticGneqSource<I>(magneticA, pullID);

    MagneticPDF g{};

    g.x = geq.x + mhd_omsM  * gneq.x;
    g.y = geq.y + mhd_omsM  * gneq.y;
    g.z = geq.z + mhd_omsM  * gneq.z;

    return g;
}

// ============================================================
// Accumulate magnetic population
// ============================================================

template <label_t I>
__device__ __forceinline__ void accumulateMagnetic(const MagneticPDF &g, MagneticMomentAccumulator &m) noexcept
{
    constexpr int cx = D3Q7::cx<I>();
    constexpr int cy = D3Q7::cy<I>();
    constexpr int cz = D3Q7::cz<I>();

    m.bx += g.x;
    m.by += g.y;
    m.bz += g.z;

    if constexpr (cx == 1)
    {
        m.axx += g.x;
        m.axy += g.y;
        m.axz += g.z;
    }
    else if constexpr (cx == -1)
    {
        m.axx -= g.x;
        m.axy -= g.y;
        m.axz -= g.z;
    }

    if constexpr (cy == 1)
    {
        m.ayx += g.x;
        m.ayy += g.y;
        m.ayz += g.z;
    }
    else if constexpr (cy == -1)
    {
        m.ayx -= g.x;
        m.ayy -= g.y;
        m.ayz -= g.z;
    }

    if constexpr (cz == 1)
    {
        m.azx += g.x;
        m.azy += g.y;
        m.azz += g.z;
    }
    else if constexpr (cz == -1)
    {
        m.azx -= g.x;
        m.azy -= g.y;
        m.azz -= g.z;
    }
}

// ============================================================
// Recover fluid moments including Maxwell stress
//
// Pi^eq = rho uu + rho cs^2 I + M
//
// M = (1 / mu0) [ 0.5 B^2 I - BB ]
// ============================================================

__device__ __forceinline__ LocalState recoverFluidMomentsMHD(const MomentAccumulator &m, const real_t bx, const real_t by, const real_t bz) noexcept
{
    LocalState state{};

    state.rho = m.rho;

    const real_t invrho = static_cast<real_t>(1) / state.rho;

    state.ux = m.rhoux * invrho;
    state.uy = m.rhouy * invrho;
    state.uz = m.rhouz * invrho;

    const real_t B2 = bx * bx + by * by + bz * bz;
    const real_t halfB2 = static_cast<real_t>(0.5) * B2;

    const real_t mxx = mhd_inv_mu0 * (halfB2 - bx * bx);
    const real_t mxy = -mhd_inv_mu0 * bx * by;
    const real_t mxz = -mhd_inv_mu0 * bx * bz;

    const real_t myy = mhd_inv_mu0 * (halfB2 - by * by);
    const real_t myz = -mhd_inv_mu0 * by * bz;

    const real_t mzz = mhd_inv_mu0 * (halfB2 - bz * bz);

    state.pixx = m.pxx - m.rhoux * state.ux - state.rho * cs2 - mxx;
    state.pixy = m.pxy - m.rhoux * state.uy - mxy;
    state.pixz = m.pxz - m.rhoux * state.uz - mxz;

    state.piyy = m.pyy - m.rhouy * state.uy - state.rho * cs2 - myy;
    state.piyz = m.pyz - m.rhouy * state.uz - myz;

    state.pizz = m.pzz - m.rhouz * state.uz - state.rho * cs2 - mzz;

    return state;
}

// ============================================================
// Recover magnetic moments
//
// A_alpha_beta^neq = A_alpha_beta - A_alpha_beta^eq
//
// A_alpha_beta^eq = u_alpha B_beta - B_alpha u_beta
// ============================================================

__device__ __forceinline__ LocalMagneticState recoverMagneticMomentsMHD(const MagneticMomentAccumulator &m, const real_t ux, const real_t uy, const real_t uz) noexcept
{
    LocalMagneticState state{};

    state.bx = m.bx;
    state.by = m.by;
    state.bz = m.bz;

    const real_t axyEq = ux * state.by - state.bx * uy;
    const real_t axzEq = ux * state.bz - state.bx * uz;
    const real_t ayzEq = uy * state.bz - state.by * uz;

    state.axx = m.axx;
    state.axy = m.axy - axyEq;
    state.axz = m.axz - axzEq;

    state.ayx = m.ayx + axyEq;
    state.ayy = m.ayy;
    state.ayz = m.ayz - ayzEq;

    state.azx = m.azx + axzEq;
    state.azy = m.azy + ayzEq;
    state.azz = m.azz;

    return state;
}

// ============================================================
// MR-MHD reconstruct-collide-stream
// ============================================================

__device__ __forceinline__ void lbmmhd(const label_t x, const label_t y, const label_t z,
                                       const MomentState &fluidA, const MomentState &fluidB,
                                       const MagneticState &magneticA, const MagneticState &magneticB) noexcept
{
    const label_t id = idx(x, y, z);

    MomentAccumulator fluidMoments{};
    MagneticMomentAccumulator magneticMoments{};

    constexpr_for<0, Q>([&](auto I)
                        {
        constexpr label_t i = decltype(I)::value;

        const label_t pullID = pullidPeriXYZ<i>(x, y, z);


        const real_t rho_s = fluidA.rho[pullID];

        const real_t ux_s = fluidA.ux[pullID];
        const real_t uy_s = fluidA.uy[pullID];
        const real_t uz_s = fluidA.uz[pullID];

        const real_t bx_s = magneticA.bx[pullID];
        const real_t by_s = magneticA.by[pullID];
        const real_t bz_s = magneticA.bz[pullID];

        const real_t cu = latticeCu<i>(ux_s, uy_s, uz_s);
        const real_t cB = latticeCu<i>(bx_s, by_s, bz_s);



        const real_t fi = fstarMHD<i>(rho_s, ux_s, uy_s, uz_s,
                                      fluidA.pixx[pullID], fluidA.pixy[pullID], fluidA.pixz[pullID],
                                      fluidA.piyy[pullID], fluidA.piyz[pullID], fluidA.pizz[pullID],
                                      bx_s, by_s, bz_s, cu, cB);

        accumulate<i>(fi, fluidMoments);

        if constexpr (i < D3Q7::Q)
        {
            const MagneticPDF gi = gstarMHD<i>(ux_s, uy_s, uz_s, bx_s, by_s, bz_s, cu, cB, magneticA, pullID);

            accumulateMagnetic<i>(gi, magneticMoments);
        } });

    const real_t bx = magneticMoments.bx;
    const real_t by = magneticMoments.by;
    const real_t bz = magneticMoments.bz;

    const LocalState fluidState = recoverFluidMomentsMHD(fluidMoments, bx, by, bz);

    const LocalMagneticState magneticState = recoverMagneticMomentsMHD(magneticMoments, fluidState.ux, fluidState.uy, fluidState.uz);

    fluidB.rho[id] = fluidState.rho;

    fluidB.ux[id] = fluidState.ux;
    fluidB.uy[id] = fluidState.uy;
    fluidB.uz[id] = fluidState.uz;

    fluidB.pixx[id] = fluidState.pixx;
    fluidB.pixy[id] = fluidState.pixy;
    fluidB.pixz[id] = fluidState.pixz;

    fluidB.piyy[id] = fluidState.piyy;
    fluidB.piyz[id] = fluidState.piyz;
    fluidB.pizz[id] = fluidState.pizz;

    magneticB.bx[id] = magneticState.bx;
    magneticB.by[id] = magneticState.by;
    magneticB.bz[id] = magneticState.bz;

    magneticB.axx[id] = magneticState.axx;
    magneticB.axy[id] = magneticState.axy;
    magneticB.axz[id] = magneticState.axz;

    magneticB.ayx[id] = magneticState.ayx;
    magneticB.ayy[id] = magneticState.ayy;
    magneticB.ayz[id] = magneticState.ayz;

    magneticB.azx[id] = magneticState.azx;
    magneticB.azy[id] = magneticState.azy;
    magneticB.azz[id] = magneticState.azz;
}

#endif