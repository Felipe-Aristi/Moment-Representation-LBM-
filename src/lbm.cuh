#ifndef LBM_CUH
#define LBM_CUH

#include "constants.cuh"
#include "stencil.cuh"
#include "pdf.cuh"
#include "memory.cuh"

#include "utilities/types.cuh"
#include "utilities/indexing.cuh"
#include "utilities/bounds.cuh"
#include "utilities/constexprFor.cuh"

// ============================================================
// Local moment accumulator
// ============================================================

struct MomentAccumulator
{
    real_t rho = static_cast<real_t>(0);

    real_t rhoux = static_cast<real_t>(0);
    real_t rhouy = static_cast<real_t>(0);
    real_t rhouz = static_cast<real_t>(0);

    real_t pxx = static_cast<real_t>(0);
    real_t pxy = static_cast<real_t>(0);
    real_t pxz = static_cast<real_t>(0);
    real_t pyy = static_cast<real_t>(0);
    real_t pyz = static_cast<real_t>(0);
    real_t pzz = static_cast<real_t>(0);
};

// ============================================================
// Local MR-LBM state
// ============================================================

struct LocalState
{
    real_t rho;

    real_t ux;
    real_t uy;
    real_t uz;

    real_t pixx;
    real_t pixy;
    real_t pixz;
    real_t piyy;
    real_t piyz;
    real_t pizz;
};

// ============================================================
// Reconstruct and collide population
// ============================================================

template <label_t I>
__device__ __forceinline__ real_t fstar(const real_t rho, const real_t ux, const real_t uy, const real_t uz,
                                        const real_t pixx, const real_t pixy, const real_t pixz,
                                        const real_t piyy, const real_t piyz, const real_t pizz) noexcept
{
    const real_t cu = latticeCu<I>(ux, uy, uz);

    const real_t fieq = feq<I>(rho, ux, uy, uz, cu);
    const real_t fineq = fneqr<I>(ux, uy, uz, pixx, pixy, pixz, piyy, piyz, pizz, cu);

    return fieq + oms * fineq;
}

// ============================================================
// Accumulate population contribution
// ============================================================

template <label_t I>
__device__ __forceinline__ void accumulate(const real_t fi, MomentAccumulator &m) noexcept
{
    constexpr int cx = D3Q27::cx<I>();
    constexpr int cy = D3Q27::cy<I>();
    constexpr int cz = D3Q27::cz<I>();

    constexpr int cxy = cx * cy;
    constexpr int cxz = cx * cz;
    constexpr int cyz = cy * cz;

    m.rho += fi;

    if constexpr (cx == 1)
    {
        m.rhoux += fi;
    }
    else if constexpr (cx == -1)
    {
        m.rhoux -= fi;
    }

    if constexpr (cy == 1)
    {
        m.rhouy += fi;
    }
    else if constexpr (cy == -1)
    {
        m.rhouy -= fi;
    }

    if constexpr (cz == 1)
    {
        m.rhouz += fi;
    }
    else if constexpr (cz == -1)
    {
        m.rhouz -= fi;
    }

    if constexpr (cx != 0)
    {
        m.pxx += fi;
    }

    if constexpr (cy != 0)
    {
        m.pyy += fi;
    }

    if constexpr (cz != 0)
    {
        m.pzz += fi;
    }

    if constexpr (cxy == 1)
    {
        m.pxy += fi;
    }
    else if constexpr (cxy == -1)
    {
        m.pxy -= fi;
    }

    if constexpr (cxz == 1)
    {
        m.pxz += fi;
    }
    else if constexpr (cxz == -1)
    {
        m.pxz -= fi;
    }

    if constexpr (cyz == 1)
    {
        m.pyz += fi;
    }
    else if constexpr (cyz == -1)
    {
        m.pyz -= fi;
    }
}

// ============================================================
// Recover MR-LBM state
// ============================================================

__device__ __forceinline__ LocalState recoverMoments(const MomentAccumulator &m) noexcept
{
    LocalState state{};

    state.rho = m.rho;

    const real_t invrho = static_cast<real_t>(1) / state.rho;

    state.ux = m.rhoux * invrho;
    state.uy = m.rhouy * invrho;
    state.uz = m.rhouz * invrho;

    state.pixx = m.pxx - m.rhoux * state.ux - state.rho * cs2;
    state.pixy = m.pxy - m.rhoux * state.uy;
    state.pixz = m.pxz - m.rhoux * state.uz;

    state.piyy = m.pyy - m.rhouy * state.uy - state.rho * cs2;
    state.piyz = m.pyz - m.rhouy * state.uz;

    state.pizz = m.pzz - m.rhouz * state.uz - state.rho * cs2;

    return state;
}

// ============================================================
// MR-LBM
// ============================================================

__device__ __forceinline__ void lbm(const label_t x, const label_t y, const label_t z,
                                    const MomentState &A, const MomentState &B) noexcept
{
    const label_t id = idx(x, y, z);

    const real_t *__restrict__ rhoA = A.rho;

    const real_t *__restrict__ uxA = A.ux;
    const real_t *__restrict__ uyA = A.uy;
    const real_t *__restrict__ uzA = A.uz;

    const real_t *__restrict__ pixxA = A.pixx;
    const real_t *__restrict__ pixyA = A.pixy;
    const real_t *__restrict__ pixzA = A.pixz;
    const real_t *__restrict__ piyyA = A.piyy;
    const real_t *__restrict__ piyzA = A.piyz;
    const real_t *__restrict__ pizzA = A.pizz;

    real_t *__restrict__ rhoB = B.rho;

    real_t *__restrict__ uxB = B.ux;
    real_t *__restrict__ uyB = B.uy;
    real_t *__restrict__ uzB = B.uz;

    real_t *__restrict__ pixxB = B.pixx;
    real_t *__restrict__ pixyB = B.pixy;
    real_t *__restrict__ pixzB = B.pixz;
    real_t *__restrict__ piyyB = B.piyy;
    real_t *__restrict__ piyzB = B.piyz;
    real_t *__restrict__ pizzB = B.pizz;

    MomentAccumulator moments{};

    // ========================================================
    // Reconstruct, collide and Stream
    // ========================================================

    constexpr_for<0, Q>([&](auto I)
                        {
        constexpr label_t i = decltype(I)::value;

        // ====================================================
        // Pull-streaming index
        // ====================================================

        // Jet: periodic x and z
        // const label_t pullID = pullidPeriXZ<i>(x, y, z);

        // Taylor-Green vortex: periodic x, y and z
        const label_t pullID = pullidPeriXYZ<i>(x, y, z);



        const real_t rho_s = rhoA[pullID];

        const real_t ux_s = uxA[pullID];
        const real_t uy_s = uyA[pullID];
        const real_t uz_s = uzA[pullID];

        const real_t pixx_s = pixxA[pullID];
        const real_t pixy_s = pixyA[pullID];
        const real_t pixz_s = pixzA[pullID];

        const real_t piyy_s = piyyA[pullID];
        const real_t piyz_s = piyzA[pullID];
        const real_t pizz_s = pizzA[pullID];



        const real_t fi = fstar<i>(rho_s, ux_s, uy_s, uz_s,
                                   pixx_s, pixy_s, pixz_s,
                                   piyy_s, piyz_s, pizz_s);



        accumulate<i>(fi, moments); });

    const LocalState state = recoverMoments(moments);

    rhoB[id] = state.rho;

    uxB[id] = state.ux;
    uyB[id] = state.uy;
    uzB[id] = state.uz;

    pixxB[id] = state.pixx;
    pixyB[id] = state.pixy;
    pixzB[id] = state.pixz;

    piyyB[id] = state.piyy;
    piyzB[id] = state.piyz;
    pizzB[id] = state.pizz;
}

#endif