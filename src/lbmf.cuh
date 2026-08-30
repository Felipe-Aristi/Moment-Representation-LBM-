#ifndef LBMF_CUH
#define LBMF_CUH

#include "lbm.cuh"
#include "force.cuh"

template <label_t I>
__device__ __forceinline__ real_t fstarForce(const real_t rho, const real_t ux, const real_t uy, const real_t uz,
                                             const real_t pixx, const real_t pixy, const real_t pixz,
                                             const real_t piyy, const real_t piyz, const real_t pizz,
                                             const real_t Fx, const real_t Fy, const real_t Fz) noexcept
{
    const real_t cu = latticeCu<I>(ux, uy, uz);

    const real_t fieq = feq<I>(rho, ux, uy, uz, cu);
    const real_t fineq = fneqr<I>(ux, uy, uz, pixx, pixy, pixz, piyy, piyz, pizz, cu);
    const real_t Fi = force<I>(ux, uy, uz, Fx, Fy, Fz, cu);

    return fieq + oms * fineq + static_cast<real_t>(0.5) * Fi;
}

// ============================================================
// Accumulate population contribution with force
// ============================================================

template <label_t I>
__device__ __forceinline__ void accumulateForce(const real_t fi, MomentAccumulator &m) noexcept
{
    accumulate<I>(fi, m);
}

// ============================================================
// Recover MR-LBM state with force
// rho u = sum_i fi ci + F/2
// Pi_neq,R = Pi - rho uu - rho cs2 I + 1/2 (uF + Fu)
// ============================================================

__device__ __forceinline__ LocalState recoverMomentsForce(const MomentAccumulator &m,
                                                          const real_t Fx, const real_t Fy, const real_t Fz) noexcept
{
    LocalState state{};

    state.rho = m.rho;

    const real_t invrho = static_cast<real_t>(1) / state.rho;

    state.ux = (m.rhoux + static_cast<real_t>(0.5) * Fx) * invrho;
    state.uy = (m.rhouy + static_cast<real_t>(0.5) * Fy) * invrho;
    state.uz = (m.rhouz + static_cast<real_t>(0.5) * Fz) * invrho;

    state.pixx = m.pxx - state.rho * state.ux * state.ux - state.rho * cs2 + state.ux * Fx;

    state.pixy = m.pxy - state.rho * state.ux * state.uy +
                 static_cast<real_t>(0.5) * (state.ux * Fy + state.uy * Fx);

    state.pixz = m.pxz - state.rho * state.ux * state.uz +
                 static_cast<real_t>(0.5) * (state.ux * Fz + state.uz * Fx);

    state.piyy = m.pyy - state.rho * state.uy * state.uy - state.rho * cs2 + state.uy * Fy;

    state.piyz = m.pyz - state.rho * state.uy * state.uz +
                 static_cast<real_t>(0.5) * (state.uy * Fz + state.uz * Fy);

    state.pizz = m.pzz - state.rho * state.uz * state.uz - state.rho * cs2 + state.uz * Fz;

    return state;
}

// ============================================================
// Forced MR-LBM
// ============================================================

__device__ __forceinline__ void lbmf(const label_t x, const label_t y, const label_t z,
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
    // Reconstruct, collide, force and Stream
    // ========================================================

    constexpr_for<0, Q>([&](auto I)
                        {constexpr label_t i = decltype(I)::value;

        // ====================================================
        // Pull-streaming index
        //
        // Channel:
        // x -> periodic
        // y -> walls
        // z -> periodic
        // ====================================================

        const label_t pullID = pullidPeriXZ<i>(x, y, z);

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

        // ====================================================
        // Force density at source node  F = rho g
        // ====================================================

        const real_t Fx_s = rho_s * channel_g;

        constexpr real_t Fy_s = static_cast<real_t>(0);
        constexpr real_t Fz_s = static_cast<real_t>(0);


        const real_t fi = fstarForce<i>(rho_s, ux_s, uy_s, uz_s, pixx_s, pixy_s, pixz_s, piyy_s, piyz_s, pizz_s,
                                        Fx_s, Fy_s, Fz_s);


        accumulateForce<i>(fi, moments); });

    // ========================================================
    // Force density at destination node F = rho g
    // ========================================================

    const real_t Fx = moments.rho * channel_g;

    constexpr real_t Fy = static_cast<real_t>(0);
    constexpr real_t Fz = static_cast<real_t>(0);

    const LocalState state = recoverMomentsForce(moments, Fx, Fy, Fz);

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