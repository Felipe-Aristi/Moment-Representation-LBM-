#ifndef CHANNEL_INITIALIZATION_CUH
#define CHANNEL_INITIALIZATION_CUH

#include <cmath>

#include "../../src/constants.cuh"
#include "../../src/memory.cuh"
#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/types.cuh"

// ============================================================
// Initial mean channel velocity
// ============================================================

__device__ __forceinline__ real_t channelMeanVelocity(const label_t y) noexcept
{
    const real_t yr = static_cast<real_t>(y);

    const real_t wallDistance =
        yr <= channel_halfHeight
            ? yr
            : channel_height - yr;

    const real_t yPlus = wallDistance * channel_uTau / channel_nu;

    if (yPlus <= static_cast<real_t>(0))
    {
        return static_cast<real_t>(0);
    }

    const real_t uPlus =
        yPlus <= channel_yPlusTransition
            ? yPlus
            : static_cast<real_t>(1) / channel_kappa * logf(yPlus) + channel_B;

    return channel_uTau * uPlus;
}

// ============================================================
// Channel-flow initialization
// ============================================================

__device__ __forceinline__ void initializeChannelFlow(const label_t x, const label_t y, const label_t z,
                                                      MomentState &state) noexcept
{
    const label_t id = idx(x, y, z);

    // ========================================================
    // Mean streamwise velocity
    // ========================================================

    const real_t uxMean = channelMeanVelocity(y);

    // ========================================================
    // Divergence-free cross-plane perturbation
    //
    // psi = A/kz * s(y) sin(kx x) sin(kz z)
    //
    // uy =  d psi / dz
    // uz = -d psi / dy
    //
    // therefore:
    //
    // d uy / dy + d uz / dz = 0
    // ========================================================

    const real_t xr = static_cast<real_t>(x) - static_cast<real_t>(1);
    const real_t zr = static_cast<real_t>(z) - static_cast<real_t>(1);
    const real_t yr = static_cast<real_t>(y);

    const real_t phaseX = channel_kx * xr;
    const real_t phaseZ = channel_kz * zr;
    const real_t phaseY = pi * yr / channel_height;

    const real_t sinX = sinf(phaseX);
    const real_t sinZ = sinf(phaseZ);
    const real_t cosZ = cosf(phaseZ);

    const real_t sinY = sinf(phaseY);

    const real_t wallEnvelope = sinY * sinY;

    const real_t wallEnvelopeDerivative =
        pi / channel_height *
        sinf(static_cast<real_t>(2) * phaseY);

    const real_t uyPerturbation =
        channel_perturbation *
        wallEnvelope *
        sinX *
        cosZ;

    const real_t uzPerturbation =
        -channel_perturbation *
        wallEnvelopeDerivative / channel_kz *
        sinX *
        sinZ;

    // ========================================================
    // Macroscopic fields
    // ========================================================

    state.rho[id] = rho0;

    state.ux[id] = uxMean;
    state.uy[id] = uyPerturbation;
    state.uz[id] = uzPerturbation;

    state.pixx[id] = static_cast<real_t>(0);
    state.pixy[id] = static_cast<real_t>(0);
    state.pixz[id] = static_cast<real_t>(0);

    state.piyy[id] = static_cast<real_t>(0);
    state.piyz[id] = static_cast<real_t>(0);
    state.pizz[id] = static_cast<real_t>(0);
}

#endif