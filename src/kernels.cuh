#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "memory.cuh"
#include "statistics/statisticsMemory.cuh"
#include "../cases/MHD/magneticMemory.cuh"

// ============================================================
// MR
// ============================================================

__global__ void initializeTaylorGreenKernel(MomentState state);

__global__ void kineticEnergyPartialsKernel(const __grid_constant__ MomentState state, statistics_t *partials);

__global__ void enstrophyPartialsKernel(const __grid_constant__ MomentState state, statistics_t *partials);

__global__ void statisticsFinalReductionKernel(const statistics_t *partials, statistics_t *result);

__global__ void initializeJetBulkKernel(MomentState state);

__global__ void initializeJetInletKernel(MomentState state);

__global__ void jetBoundaryConditionsKernel(const __grid_constant__ MomentState A, const __grid_constant__ MomentState B);

__global__ void initializeChannelFlowKernel(MomentState state);

__global__ void channelBoundaryConditionsKernel(const __grid_constant__ MomentState A, const __grid_constant__ MomentState B);

__global__ void lbmKernel(const __grid_constant__ MomentState A, const __grid_constant__ MomentState B);

__global__ void lbmfKernel(const __grid_constant__ MomentState A, const __grid_constant__ MomentState B);

// ============================================================
// MR-MHD
// ============================================================

__global__ void initializeOrszagTangKernel(MomentState fluid, MagneticState magnetic);

__global__ void lbmmhdKernel(const __grid_constant__ MomentState fluidA, const __grid_constant__ MomentState fluidB,
                             const __grid_constant__ MagneticState magneticA, const __grid_constant__ MagneticState magneticB);

#endif
