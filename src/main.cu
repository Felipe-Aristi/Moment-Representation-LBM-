#include <cuda_runtime.h>

#include <iostream>
#include <utility>

#include "constants.cuh"
#include "memory.cuh"
#include "launch.cuh"

#include "utilities/cudaConfig.cuh"
#include "utilities/cudaUtilities.cuh"
#include "utilities/types.cuh"
#include "utilities/mlups_count.cuh"

#include "../post/taylorgreen/vti.cuh"
#include "../post/jet/vti.cuh"
#include "../post/channelflow/vti.cuh"

int main()
{
    constexpr int deviceID = 0;

    CUDA_CHECK(cudaSetDevice(deviceID));

    const CudaConfig cfg = print_device_and_make_config(deviceID);

    Memory memory;

    allocateMemory(memory);

    MlupsCounter mlupsCounter;

    initializeMlupsCounter(mlupsCounter);

    printMlupsHeader();

    // ========================================================
    // Initialization
    // ========================================================

    // launchTaylorGreenInitialization(cfg, memory.d_A);

    // launchJetInitialization(cfg, memory.d_A);

    launchChannelFlowInitialization(cfg, memory.d_A);

    // ========================================================
    // A-B flip-flop pointers
    // ========================================================

    MomentState *A = &memory.d_A;
    MomentState *B = &memory.d_B;

    // Write the initialized state at the actual simulation time t = 0.
    copyDeviceToHost(memory.host, *A);
    writeChannelFlowVTI(memory.host, static_cast<label_t>(0));

    // ========================================================
    // Time evolution
    // ========================================================

    for (label_t step = static_cast<label_t>(0); step < Steps; ++step)
    {
        if (step % mlupsEvery == static_cast<label_t>(0))
        {
            startMlupsCounter(mlupsCounter);
        }

        // launchLBM(cfg, *A, *B);

        // launchJetBoundaryConditions(cfg, *A, *B);

        launchLBMF(cfg, *A, *B);

        launchChannelBoundaryConditions(cfg, *A, *B);

        std::swap(A, B);

        if ((step + static_cast<label_t>(1)) % mlupsEvery == static_cast<label_t>(0))
        {
            printMlups(mlupsCounter, step + static_cast<label_t>(1));
        }

        if ((step + static_cast<label_t>(1)) % output == static_cast<label_t>(0))
        {
            copyDeviceToHost(memory.host, *A);

            // writeTaylorGreenVTI(memory.host, step + static_cast<label_t>(1));
            // writeJetVTI(memory.host, step + static_cast<label_t>(1));

            writeChannelFlowVTI(memory.host, step + static_cast<label_t>(1));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    destroyMlupsCounter(mlupsCounter);

    freeMemory(memory);

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
