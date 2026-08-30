#ifndef TAYLORGREEN_INITIALIZATION_CUH
#define TAYLORGREEN_INITIALIZATION_CUH

#include "../../src/constants.cuh"
#include "../../src/memory.cuh"
#include "../../src/utilities/bounds.cuh"
#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/types.cuh"

__device__ __forceinline__ void initializeTaylorGreen(const label_t x, const label_t y, const label_t z, MomentState &state) noexcept
{
    const label_t id = idx(x, y, z);

    const label_t xp = wrapx(x);
    const label_t yp = wrapy(y);
    const label_t zp = wrapz(z);

    const real_t X = tgv_kx * static_cast<real_t>(xp - static_cast<label_t>(1));
    const real_t Y = tgv_ky * static_cast<real_t>(yp - static_cast<label_t>(1));
    const real_t Z = tgv_kz * static_cast<real_t>(zp - static_cast<label_t>(1));

    const real_t sinX = sin(X);
    const real_t sinY = sin(Y);
    const real_t sinZ = sin(Z);

    const real_t cosX = cos(X);
    const real_t cosY = cos(Y);
    const real_t cosZ = cos(Z);

    const real_t ux = tgv_U0 * sinX * cosY * cosZ;
    const real_t uy = -tgv_U0 * cosX * sinY * cosZ;
    const real_t uz = static_cast<real_t>(0);

    const real_t pressure = rho0 * tgv_U0 * tgv_U0 * static_cast<real_t>(1.0 / 16.0) * (cos(static_cast<real_t>(2) * X) + cos(static_cast<real_t>(2) * Y)) * (cos(static_cast<real_t>(2) * Z) + static_cast<real_t>(2));

    const real_t rho = rho0 + pressure * inv_cs2;

    const real_t duxdx = tgv_U0 * tgv_kx * cosX * cosY * cosZ;
    const real_t duxdy = -tgv_U0 * tgv_ky * sinX * sinY * cosZ;
    const real_t duxdz = -tgv_U0 * tgv_kz * sinX * cosY * sinZ;

    const real_t duydx = tgv_U0 * tgv_kx * sinX * sinY * cosZ;
    const real_t duydy = -tgv_U0 * tgv_ky * cosX * cosY * cosZ;
    const real_t duydz = tgv_U0 * tgv_kz * cosX * sinY * sinZ;

    const real_t stressFactor = -rho * cs2 * tau;

    const real_t pixx = static_cast<real_t>(2) * stressFactor * duxdx;
    const real_t pixy = stressFactor * (duxdy + duydx);
    const real_t pixz = stressFactor * duxdz;

    const real_t piyy = static_cast<real_t>(2) * stressFactor * duydy;
    const real_t piyz = stressFactor * duydz;

    const real_t pizz = static_cast<real_t>(0);

    state.rho[id] = rho;

    state.ux[id] = ux;
    state.uy[id] = uy;
    state.uz[id] = uz;

    state.pixx[id] = pixx;
    state.pixy[id] = pixy;
    state.pixz[id] = pixz;
    state.piyy[id] = piyy;
    state.piyz[id] = piyz;
    state.pizz[id] = pizz;
}

#endif