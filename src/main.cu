
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
#include "../cases/taylorgreen/statistics.cuh"

int main()
{
    constexpr int deviceID = 0;

    CUDA_CHECK(cudaSetDevice(deviceID));

    const CudaConfig cfg = print_device_and_make_config(deviceID);

    // ============================================================
    // Solver memory
    // ============================================================

    Memory memory;
    allocateMemory(memory);

    // ============================================================
    // Statistics memory
    // ============================================================

    StatisticsMemory statisticsMemory;
    allocateStatisticsMemory(statisticsMemory);

    // ============================================================
    // MLUPS counter
    // ============================================================

    MlupsCounter mlupsCounter;
    initializeMlupsCounter(mlupsCounter);
    printMlupsHeader();

    int exitCode = EXIT_SUCCESS;

    try
    {
        // ========================================================
        // Taylor-Green statistics output
        // ========================================================

        TaylorGreenStatisticsOutput statisticsOutput;
        initializeTaylorGreenStatisticsOutput(statisticsOutput);

        // ========================================================
        // Taylor-Green initialization
        // ========================================================

        launchTaylorGreenInitialization(cfg, memory.d_A);

        // ========================================================
        // A-B flip-flop pointers
        // ========================================================

        MomentState *A = &memory.d_A;
        MomentState *B = &memory.d_B;

        // ========================================================
        // Initial kinetic energy
        // ========================================================

        const statistics_t initialKineticEnergySum = launchKineticEnergyStatistics(*A, statisticsMemory);
        const statistics_t initialKineticEnergyStar = taylorGreenKineticEnergyStar(initialKineticEnergySum);
        const statistics_t initialKineticEnergyRelativeError = taylorGreenInitialKineticEnergyRelativeError(initialKineticEnergyStar);

        writeTaylorGreenKineticEnergy(statisticsOutput, static_cast<label_t>(0), initialKineticEnergySum);

        std::cout << std::setprecision(std::numeric_limits<statistics_t>::max_digits10)
                  << "Taylor-Green step-0 kinetic-energy validation\n"
                  << "SK        = " << initialKineticEnergySum << "\n"
                  << "SK exact  = " << tgvExpectedKineticEnergySum << "\n"
                  << "KStar     = " << initialKineticEnergyStar << "\n"
                  << "rel error = " << initialKineticEnergyRelativeError << "\n";

        // ========================================================
        // Initial enstrophy
        // ========================================================

        const statistics_t initialEnstrophySum = launchEnstrophyStatistics(*A, statisticsMemory);
        const statistics_t initialEnstrophyStar = taylorGreenEnstrophyStar(initialEnstrophySum);
        const statistics_t expectedInitialEnstrophySum = taylorGreenExpectedInitialEnstrophySumFD();
        const statistics_t expectedInitialEnstrophyStar = taylorGreenExpectedInitialEnstrophyStarFD();
        const statistics_t initialEnstrophyRelativeError = taylorGreenInitialEnstrophyRelativeError(initialEnstrophyStar);

        writeTaylorGreenEnstrophy(statisticsOutput, static_cast<label_t>(0), initialEnstrophySum);

        std::cout << std::setprecision(std::numeric_limits<statistics_t>::max_digits10)
                  << "Taylor-Green step-0 enstrophy validation\n"
                  << "Szeta         = " << initialEnstrophySum << "\n"
                  << "Szeta exact   = " << expectedInitialEnstrophySum << "\n"
                  << "ZetaStar      = " << initialEnstrophyStar << "\n"
                  << "ZetaStar FD   = " << expectedInitialEnstrophyStar << "\n"
                  << "ZetaStar cont = " << static_cast<statistics_t>(3.0 / 8.0) << "\n"
                  << "rel error     = " << initialEnstrophyRelativeError << "\n";

        // ========================================================
        // Initial VTI output
        // ========================================================

        copyDeviceToHost(memory.host, *A);
        writeTaylorGreenVTI(memory.host, static_cast<label_t>(0));

        // ========================================================
        // Time evolution
        // ========================================================

        for (label_t step = static_cast<label_t>(0); step < Steps; ++step)
        {
            // ====================================================
            // MLUPS timer
            // ====================================================

            if (step % mlupsEvery == static_cast<label_t>(0))
            {
                startMlupsCounter(mlupsCounter);
            }

            // ====================================================
            // MR-LBM evolution
            // ====================================================

            launchLBM(cfg, *A, *B);

            // ====================================================
            // A-B swap
            // ====================================================

            std::swap(A, B);

            const label_t completedStep = step + static_cast<label_t>(1);

            // ====================================================
            // MLUPS output
            // ====================================================

            if (completedStep % mlupsEvery == static_cast<label_t>(0))
            {
                printMlups(mlupsCounter, completedStep);
            }

            // ====================================================
            // Taylor-Green statistics
            // ====================================================

            if (statisticsSampleDue<statisticsStart, statisticsEvery>(completedStep))
            {
                const statistics_t kineticEnergySum = launchKineticEnergyStatistics(*A, statisticsMemory);
                const statistics_t enstrophySum = launchEnstrophyStatistics(*A, statisticsMemory);

                writeTaylorGreenKineticEnergy(statisticsOutput, completedStep, kineticEnergySum);
                writeTaylorGreenEnstrophy(statisticsOutput, completedStep, enstrophySum);
            }

            // ====================================================
            // VTI output
            // ====================================================

            if (completedStep % output == static_cast<label_t>(0))
            {
                copyDeviceToHost(memory.host, *A);
                writeTaylorGreenVTI(memory.host, completedStep);
            }
        }

        // ========================================================
        // Close statistics output
        // ========================================================

        closeTaylorGreenStatisticsOutput(statisticsOutput);
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << "\n";
        exitCode = EXIT_FAILURE;
    }

    // ============================================================
    // Cleanup
    // ============================================================

    CUDA_CHECK(cudaDeviceSynchronize());

    destroyMlupsCounter(mlupsCounter);

    freeStatisticsMemory(statisticsMemory);
    freeMemory(memory);

    CUDA_CHECK(cudaDeviceReset());

    return exitCode;
}
