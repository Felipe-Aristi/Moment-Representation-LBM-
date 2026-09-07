#ifndef MHD_LBM_CUH
#define MHD_LBM_CUH

#include "../../src/lbm.cuh"
#include "memory.cuh"
#include "pdfmhd.cuh"

namespace mhd
{
__device__ __forceinline__ Vector velocityAt(const State &s, const label_t id) noexcept
{
    return {s.fluid.ux[id], s.fluid.uy[id], s.fluid.uz[id]};
}

__device__ __forceinline__ Vector magneticAt(const State &s, const label_t id) noexcept
{
    return {s.bx[id], s.by[id], s.bz[id]};
}

template <label_t I>
__device__ __forceinline__ label_t magneticPullId(const label_t x, const label_t y, const label_t z) noexcept
{
    return idx(wrapx(static_cast<label_t>(static_cast<int>(x) - D3Q7::cx<I>())),
               wrapy(static_cast<label_t>(static_cast<int>(y) - D3Q7::cy<I>())),
               wrapz(static_cast<label_t>(static_cast<int>(z) - D3Q7::cz<I>())));
}

template <label_t I>
__device__ __forceinline__ real_t fluidPostCollision(const State &s, const label_t id) noexcept
{
    const MomentState &f = s.fluid;
    return fstar<I>(f.rho[id], f.ux[id], f.uy[id], f.uz[id], f.pixx[id], f.pixy[id], f.pixz[id], f.piyy[id], f.piyz[id], f.pizz[id])
        + magneticEquilibrium<I>(magneticAt(s, id));
}

__device__ __forceinline__ void storeFluid(const LocalState &f, const MomentState &s, const label_t id) noexcept
{
    s.rho[id] = f.rho;
    s.ux[id] = f.ux; s.uy[id] = f.uy; s.uz[id] = f.uz;
    s.pixx[id] = f.pixx; s.pixy[id] = f.pixy; s.pixz[id] = f.pixz;
    s.piyy[id] = f.piyy; s.piyz[id] = f.piyz; s.pizz[id] = f.pizz;
}

// A is the complete pre-collision state at t; B becomes the state at t+1.
// Every source read is from A. No synchronization or ghost copying is needed.
__device__ __forceinline__ void advanceNode(const label_t x, const label_t y, const label_t z, const State &a, const State &b) noexcept
{
    const label_t id = idx(x, y, z);
    real_t field[3] = {};
    constexpr_for<0, D3Q7::q>([&](auto I)
    {
        constexpr label_t i = decltype(I)::value;
        const label_t source = magneticPullId<i>(x, y, z);
        const Vector u = velocityAt(a, source);
        const Vector magnetic = magneticAt(a, source);
        constexpr_for<0, 3>([&](auto A)
        {
            constexpr label_t alpha = decltype(A)::value;
            const real_t g = magneticOms * a.g[populationIndex<i, alpha>(source)] + magneticOmega * geq<i, alpha>(u, magnetic);
            b.g[populationIndex<i, alpha>(id)] = g;
            field[alpha] += g;
        });
    });
    const Vector magnetic = {field[0], field[1], field[2]};
    b.bx[id] = magnetic.x; b.by[id] = magnetic.y; b.bz[id] = magnetic.z;

    MomentAccumulator moments{};
    constexpr_for<0, Q>([&](auto I)
    {
        constexpr label_t i = decltype(I)::value;
        accumulate<i>(fluidPostCollision<i>(a, pullidPeriXYZ<i>(x, y, z)), moments);
    });

    LocalState fluid = recoverMoments(moments);
    // Pi^neq = Pi - rho u u - rho cs2 I - M(B), using B at t+1.
    // R3 uses this corrected tensor in the existing recursive reconstruction.
    const MaxwellStress m = maxwellStress(magnetic);
    fluid.pixx -= m.xx; fluid.pixy -= m.xy; fluid.pixz -= m.xz;
    fluid.piyy -= m.yy; fluid.piyz -= m.yz; fluid.pizz -= m.zz;
    storeFluid(fluid, b.fluid, id);
}
}
#endif
