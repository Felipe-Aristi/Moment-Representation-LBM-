#ifndef MHD_PDFMHD_CUH
#define MHD_PDFMHD_CUH

#include "../../src/pdf.cuh"
#include "config.cuh"
#include "stencils/d3q7.cuh"

namespace mhd
{
struct Vector
{
    real_t x, y, z;
    template <label_t A> __host__ __device__ constexpr real_t component() const noexcept
    {
        static_assert(A < 3);
        if constexpr (A == 0) return x;
        if constexpr (A == 1) return y;
        return z;
    }
};

// Conservative magnetic momentum flux M = (B.B / 2) I - B B.
struct MaxwellStress
{
    real_t xx, xy, xz, yy, yz, zz;
};

__host__ __device__ inline MaxwellStress maxwellStress(const Vector b) noexcept
{
    const real_t pressure = static_cast<real_t>(0.5) * (b.x * b.x + b.y * b.y + b.z * b.z);
    return {pressure - b.x * b.x, -b.x * b.y, -b.x * b.z, pressure - b.y * b.y, -b.y * b.z, pressure - b.z * b.z};
}

// Hermite projection guarantees zero mass/momentum and second moment M,
// including the nonzero trace of M in three dimensions.
template <label_t I>
__device__ __forceinline__ real_t magneticEquilibrium(const Vector b) noexcept
{
    const MaxwellStress m = maxwellStress(b);
    const real_t contraction = D3Q27::Hxx<I>() * m.xx + D3Q27::Hyy<I>() * m.yy + D3Q27::Hzz<I>() * m.zz
        + static_cast<real_t>(2) * (D3Q27::Hxy<I>() * m.xy + D3Q27::Hxz<I>() * m.xz + D3Q27::Hyz<I>() * m.yz);
    return D3Q27::w<I>() * inv_2cs4 * contraction;
}

template <label_t I>
__device__ __forceinline__ real_t feqMHD(const real_t rho, const Vector u, const Vector b) noexcept
{
    return feq<I>(rho, u.x, u.y, u.z, latticeCu<I>(u.x, u.y, u.z)) + magneticEquilibrium<I>(b);
}

// First moment (streaming index beta): u_beta B_alpha - B_beta u_alpha.
// The stretching term is essential: B is not three passive scalars.
template <label_t I, label_t A>
__device__ __forceinline__ real_t geq(const Vector u, const Vector b) noexcept
{
    const real_t cu = D3Q7::cx<I>() * u.x + D3Q7::cy<I>() * u.y + D3Q7::cz<I>() * u.z;
    const real_t cb = D3Q7::cx<I>() * b.x + D3Q7::cy<I>() * b.y + D3Q7::cz<I>() * b.z;
    return D3Q7::w<I>() * (b.component<A>() + (cu * b.component<A>() - cb * u.component<A>()) / magneticCs2);
}
}
#endif
