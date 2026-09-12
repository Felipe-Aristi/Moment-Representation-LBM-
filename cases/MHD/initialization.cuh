#ifndef MHD_INITIALIZATION_CUH
#define MHD_INITIALIZATION_CUH

#include "constantsmhd.cuh"
#include "magneticMemory.cuh"

#include "../../src/memory.cuh"
#include "../../src/utilities/bounds.cuh"
#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// Orszag-Tang vortex initialization
// ============================================================

__device__ __forceinline__ void initializeOrszagTang(const label_t x, const label_t y, const label_t z, MomentState &fluid, MagneticState &magnetic) noexcept
{
    const label_t id = idx(x, y, z);

    const label_t xp = wrapx(x);
    const label_t yp = wrapy(y);

    const real_t X = ot_kx * static_cast<real_t>(xp - static_cast<label_t>(1));
    const real_t Y = ot_ky * static_cast<real_t>(yp - static_cast<label_t>(1));

    const real_t sinX = sin(X);
    const real_t sinY = sin(Y);

    const real_t cosX = cos(X);
    const real_t cosY = cos(Y);
    const real_t cos2X = cos(static_cast<real_t>(2) * X);

    // ========================================================
    // Fluid fields
    // u = (-U0 sin(Y), U0 sin(X), 0)
    // ========================================================

    const real_t rho = rho0;

    const real_t ux = -ot_U0 * sinY;
    const real_t uy = ot_U0 * sinX;
    const real_t uz = static_cast<real_t>(0);

    // ========================================================
    // Magnetic fields
    // B = (-B0 sin(Y), B0 sin(2X), 0)
    // ========================================================

    const real_t bx = -ot_B0 * sinY;
    const real_t by = ot_B0 * sin(static_cast<real_t>(2) * X);
    const real_t bz = static_cast<real_t>(0);

    // ========================================================
    // Fluid velocity gradients
    // ========================================================

    const real_t duxdx = static_cast<real_t>(0);
    const real_t duxdy = -ot_U0 * ot_ky * cosY;

    const real_t duydx = ot_U0 * ot_kx * cosX;
    const real_t duydy = static_cast<real_t>(0);

    // ========================================================
    // Initial fluid non-equilibrium stress
    // ========================================================

    const real_t stressFactor = -rho * cs2 * mhd_tauF;

    const real_t pixx = static_cast<real_t>(2) * stressFactor * duxdx;
    const real_t pixy = stressFactor * (duxdy + duydx);
    const real_t pixz = static_cast<real_t>(0);

    const real_t piyy = static_cast<real_t>(2) * stressFactor * duydy;
    const real_t piyz = static_cast<real_t>(0);

    const real_t pizz = static_cast<real_t>(0);

    // ========================================================
    // Magnetic field gradients
    // ========================================================

    const real_t dbxdx = static_cast<real_t>(0);
    const real_t dbxdy = -ot_B0 * ot_ky * cosY;

    const real_t dbydx = static_cast<real_t>(2) * ot_B0 * ot_kx * cos2X;
    const real_t dbydy = static_cast<real_t>(0);

    // ========================================================
    // Initial magnetic non-equilibrium first moment
    // A_alpha_beta^neq = -tauM cs_m^2 d_alpha B_beta
    // ========================================================

    const real_t magneticFactor = -mhd_tauM * mhd_cs2;

    const real_t axx = magneticFactor * dbxdx;
    const real_t axy = magneticFactor * dbydx;
    const real_t axz = static_cast<real_t>(0);

    const real_t ayx = magneticFactor * dbxdy;
    const real_t ayy = magneticFactor * dbydy;
    const real_t ayz = static_cast<real_t>(0);

    const real_t azx = static_cast<real_t>(0);
    const real_t azy = static_cast<real_t>(0);
    const real_t azz = static_cast<real_t>(0);

    // ========================================================
    // Store fluid moment representation
    // ========================================================

    fluid.rho[id] = rho;

    fluid.ux[id] = ux;
    fluid.uy[id] = uy;
    fluid.uz[id] = uz;

    fluid.pixx[id] = pixx;
    fluid.pixy[id] = pixy;
    fluid.pixz[id] = pixz;

    fluid.piyy[id] = piyy;
    fluid.piyz[id] = piyz;
    fluid.pizz[id] = pizz;

    // ========================================================
    // Store magnetic moment representation
    // ========================================================

    magnetic.bx[id] = bx;
    magnetic.by[id] = by;
    magnetic.bz[id] = bz;

    magnetic.axx[id] = axx;
    magnetic.axy[id] = axy;
    magnetic.axz[id] = axz;

    magnetic.ayx[id] = ayx;
    magnetic.ayy[id] = ayy;
    magnetic.ayz[id] = ayz;

    magnetic.azx[id] = azx;
    magnetic.azy[id] = azy;
    magnetic.azz[id] = azz;
}

#endif