#ifndef MHD_D3Q7_CUH
#define MHD_D3Q7_CUH

#include "../../../src/utilities/types.cuh"

namespace mhd
{
struct D3Q7
{
    static constexpr label_t q = 7;
    template <label_t I> __host__ __device__ static constexpr int cx() noexcept
    {
        static_assert(I < q);
        return I == 1 ? 1 : I == 2 ? -1 : 0;
    }
    template <label_t I> __host__ __device__ static constexpr int cy() noexcept
    {
        static_assert(I < q);
        return I == 3 ? 1 : I == 4 ? -1 : 0;
    }
    template <label_t I> __host__ __device__ static constexpr int cz() noexcept
    {
        static_assert(I < q);
        return I == 5 ? 1 : I == 6 ? -1 : 0;
    }
    template <label_t I> __host__ __device__ static constexpr real_t w() noexcept
    {
        static_assert(I < q);
        return I == 0 ? static_cast<real_t>(0.25) : static_cast<real_t>(0.125);
    }
};
}
#endif
