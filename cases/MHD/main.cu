#include <cuda_runtime.h>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <utility>
#include "launch.cuh"
#include "output.cuh"

#ifndef MRLBM_MHD
#error "Compile the MHD executable with -DMRLBM_MHD."
#endif

int main(const int argc, char **argv)
{
    mhd::State a{}, b{};
    try
    {
        if (argc > 3) throw std::runtime_error("Usage: mr_lbm_mhd [device_id] [new_output_directory]");
        int device = 0;
        if (argc > 1)
        {
            const std::string_view arg(argv[1]);
            const auto result = std::from_chars(arg.data(), arg.data() + arg.size(), device);
            if (result.ec != std::errc{} || result.ptr != arg.data() + arg.size() || device < 0)
                throw std::runtime_error("GPU id must be a non-negative integer.");
        }
        const std::filesystem::path directory = argc > 2 ? argv[2] : "output/MHD/orszag_tang";
        if (std::filesystem::exists(directory) && !std::filesystem::is_empty(directory))
            throw std::runtime_error("Output directory is not empty. Pass a new directory as the second argument.");

        CUDA_CHECK(cudaSetDevice(device));
        const CudaConfig cfg = print_device_and_make_config(device);
        mhd::allocate(a); mhd::allocate(b);
        mhd::Snapshot snapshot;
        std::filesystem::create_directories(directory);
        mhd::writeMetadata(directory);
        std::ofstream csv(directory / "statistics.csv");
        if (!csv) throw std::runtime_error("Cannot open MHD statistics.");
        mhd::writeStatisticsHeader(csv);
        std::vector<label_t> snapshots;
        std::cout << "MHD: " << mhd::nx << 'x' << mhd::ny << 'x' << mhd::nz
                  << " physical nodes, Re=" << mhd::reynolds << ", Rm=" << mhd::magneticReynolds
                  << ", tau=" << tau << ", tauB=" << mhd::magneticTau << '\n'
                  << "GPU state: " << 2 * (bytesMoments + mhd::magneticBytes) / (1024.0 * 1024.0) << " MiB\n";
        mhd::initialize(cfg, a);
        auto save = [&](const label_t step, const bool fields)
        {
            mhd::copySnapshot(snapshot, a);
            const auto s = mhd::statistics(snapshot);
            mhd::writeStatistics(csv, step, s);
            if (fields)
            {
                mhd::writeVTI(directory, snapshot, step);
                snapshots.push_back(step);
                mhd::writePVD(directory, snapshots);
                std::cout << "step=" << step << " t*=" << mhd::timeStar(step)
                          << " K=" << s.kinetic << " M=" << s.magnetic
                          << " divB_rms=" << s.divRms << " maxMach=" << s.maxMach << '\n';
            }
        };
        save(0, true);
        const auto start = std::chrono::steady_clock::now();
        for (label_t step = 1; step <= Steps; ++step)
        {
            mhd::advance(cfg, a, b);
            std::swap(a, b);
            const bool fields = step % mhd::outputEvery == 0 || step == Steps;
            if (fields || step % mhd::sampleEvery == 0) save(step, fields);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "Completed in " << seconds << " seconds including diagnostics. Output: " << directory << '\n';
        mhd::release(a); mhd::release(b);
        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        mhd::release(a); mhd::release(b);
        return EXIT_FAILURE;
    }
}
