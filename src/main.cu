#include <cuda_runtime.h>

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>

#include "constants.cuh"
#include "memory.cuh"
#include "launch.cuh"

#include "statistics/statisticsMemory.cuh"
#include "statistics/sampling.cuh"

#include "utilities/cudaConfig.cuh"
#include "utilities/cudaUtilities.cuh"
#include "utilities/types.cuh"
#include "utilities/mlups_count.cuh"

#include "../post/taylorgreen/vti.cuh"
#include "../post/jet/vti.cuh"
#include "../post/channelflow/vti.cuh"

#include "../cases/taylorgreen/statistics.cuh"

int main()
{
    constexpr int deviceID = 0;

    CUDA_CHECK(cudaSetDevice(deviceID));

    const CudaConfig cfg = print_device_and_make_config(deviceID);

    Memory memory;

    allocateMemory(memory);

    StatisticsMemory statisticsMemory;

    allocateStatisticsMemory(statisticsMemory);

    MlupsCounter mlupsCounter;

    initializeMlupsCounter(mlupsCounter);

    printMlupsHeader();

    int exitCode = EXIT_SUCCESS;

    try
    {
        TaylorGreenStatisticsOutput statisticsOutput;

        initializeTaylorGreenStatisticsOutput(statisticsOutput);

        // ========================================================
        // Initialization
        // ========================================================

        launchTaylorGreenInitialization(cfg, memory.d_A);

        // launchJetInitialization(cfg, memory.d_A);

        // launchChannelFlowInitialization(cfg, memory.d_A);

        // ========================================================
        // A-B flip-flop pointers
        // ========================================================

        MomentState *A = &memory.d_A;
        MomentState *B = &memory.d_B;

        // ========================================================
        // Taylor-Green statistics at t = 0
        // ========================================================

        const statistics_t initialKineticEnergySum = launchKineticEnergyStatistics(*A, statisticsMemory);
        const statistics_t initialKineticEnergyStar = taylorGreenKineticEnergyStar(initialKineticEnergySum);
        const statistics_t initialRelativeError = taylorGreenInitialKineticEnergyRelativeError(initialKineticEnergyStar);

        writeTaylorGreenKineticEnergy(statisticsOutput, static_cast<label_t>(0), initialKineticEnergySum);

        std::cout << std::setprecision(std::numeric_limits<statistics_t>::max_digits10)
                  << "Taylor-Green step-0 kinetic-energy validation\n"
                  << "SK        = " << initialKineticEnergySum << "\n"
                  << "SK exact  = " << tgvExpectedKineticEnergySum << "\n"
                  << "KStar     = " << initialKineticEnergyStar << "\n"
                  << "rel error = " << initialRelativeError << "\n";

        // Write the initialized state at the actual simulation time t = 0.
        copyDeviceToHost(memory.host, *A);
        writeTaylorGreenVTI(memory.host, static_cast<label_t>(0));

        // ========================================================
        // Time evolution
        // ========================================================

        for (label_t step = static_cast<label_t>(0); step < Steps; ++step)
        {
            if (step % mlupsEvery == static_cast<label_t>(0))
            {
                startMlupsCounter(mlupsCounter);
            }

            launchLBM(cfg, *A, *B);

            // launchJetBoundaryConditions(cfg, *A, *B);

            // launchLBMF(cfg, *A, *B);

            // launchChannelBoundaryConditions(cfg, *A, *B);

            std::swap(A, B);

            const label_t completedStep = step + static_cast<label_t>(1);

            if (completedStep % mlupsEvery == static_cast<label_t>(0))
            {
                printMlups(mlupsCounter, completedStep);
            }

            if (statisticsSampleDue<statisticsStart, statisticsEvery>(completedStep))
            {
                const statistics_t kineticEnergySum = launchKineticEnergyStatistics(*A, statisticsMemory);

                writeTaylorGreenKineticEnergy(statisticsOutput, completedStep, kineticEnergySum);
            }

            if (completedStep % output == static_cast<label_t>(0))
            {
                copyDeviceToHost(memory.host, *A);

                writeTaylorGreenVTI(memory.host, completedStep);

                // writeJetVTI(memory.host, completedStep);

                // writeChannelFlowVTI(memory.host, completedStep);
            }
        }

        closeTaylorGreenStatisticsOutput(statisticsOutput);
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << "\n";
        exitCode = EXIT_FAILURE;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    destroyMlupsCounter(mlupsCounter);

    freeStatisticsMemory(statisticsMemory);

    freeMemory(memory);

    CUDA_CHECK(cudaDeviceReset());

    return exitCode;
}
