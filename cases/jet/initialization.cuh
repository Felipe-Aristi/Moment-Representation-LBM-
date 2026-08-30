#ifndef JET_INITIALIZATION_CUH
#define JET_INITIALIZATION_CUH

#include "../../src/constants.cuh"
#include "../../src/memory.cuh"

#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/types.cuh"
#include "../../src/utilities/mathUtilities.cuh"

// ============================================================
// Jet bulk initialization
// ============================================================

__device__ __forceinline__ void initializeJetBulk(const label_t x, const label_t y, const label_t z, MomentState &state) noexcept
{
    const label_t id = idx(x, y, z);

    state.rho[id] = rho0;

    state.ux[id] = static_cast<real_t>(0);
    state.uy[id] = static_cast<real_t>(0);
    state.uz[id] = static_cast<real_t>(0);

    state.pixx[id] = static_cast<real_t>(0);
    state.pixy[id] = static_cast<real_t>(0);
    state.pixz[id] = static_cast<real_t>(0);
    state.piyy[id] = static_cast<real_t>(0);
    state.piyz[id] = static_cast<real_t>(0);
    state.pizz[id] = static_cast<real_t>(0);
}

// ============================================================
// Jet inlet initialization
// ============================================================

__device__ __forceinline__ void initializeJetInlet(const label_t x, const label_t z, MomentState &state) noexcept
{
    constexpr label_t yB = static_cast<label_t>(0);

    const label_t idB = idx(x, yB, z);

    const real_t uyB = isJet(x, z) * jet_velocity;

    state.rho[idB] = rho0;

    state.ux[idB] = static_cast<real_t>(0);
    state.uy[idB] = uyB;
    state.uz[idB] = static_cast<real_t>(0);

    state.pixx[idB] = static_cast<real_t>(0);
    state.pixy[idB] = static_cast<real_t>(0);
    state.pixz[idB] = static_cast<real_t>(0);
    state.piyy[idB] = static_cast<real_t>(0);
    state.piyz[idB] = static_cast<real_t>(0);
    state.pizz[idB] = static_cast<real_t>(0);
}

#endif