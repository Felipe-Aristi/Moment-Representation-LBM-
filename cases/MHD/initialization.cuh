#ifndef MHD_INITIALIZATION_CUH
#define MHD_INITIALIZATION_CUH

#include <cmath>
#include "lbm.cuh"

namespace mhd
{
// Classical planar Orszag-Tang pattern, at low Mach, repeated along z.
// Equilibrium start: Pi^neq = 0 and g_i = g_i^eq.
__device__ __forceinline__ void initializeNode(const label_t x, const label_t y, const label_t z, const State &s) noexcept
{
    const label_t id = idx(x, y, z);
    const real_t X = k * static_cast<real_t>(wrapx(x) - 1);
    const real_t Y = k * static_cast<real_t>(wrapy(y) - 1);
    const Vector u = {-velocity * sinf(Y), velocity * sinf(X), 0};
    const Vector b = {-magneticAmplitude * sinf(Y), magneticAmplitude * sinf(static_cast<real_t>(2) * X), 0};
    const real_t b2 = b.x * b.x + b.y * b.y;
    // Leading incompressible pressure Poisson solution, with zero mean.
    const real_t pressure = -rho0 * velocity * velocity * cosf(X) * cosf(Y)
        + static_cast<real_t>(0.8) * magneticAmplitude * magneticAmplitude * cosf(static_cast<real_t>(2) * X) * cosf(Y)
        - static_cast<real_t>(0.5) * (b2 - magneticAmplitude * magneticAmplitude);
    LocalState fluid{};
    fluid.rho = rho0 + pressure * inv_cs2;
    fluid.ux = u.x; fluid.uy = u.y; fluid.uz = u.z;
    storeFluid(fluid, s.fluid, id);
    s.bx[id] = b.x; s.by[id] = b.y; s.bz[id] = b.z;
    constexpr_for<0, D3Q7::q>([&](auto I)
    {
        constexpr label_t i = decltype(I)::value;
        constexpr_for<0, 3>([&](auto A)
        {
            constexpr label_t alpha = decltype(A)::value;
            s.g[populationIndex<i, alpha>(id)] = geq<i, alpha>(u, b);
        });
    });
}
}
#endif
