#ifndef STATISTICS_SAMPLING_CUH
#define STATISTICS_SAMPLING_CUH

#include "../utilities/types.cuh"

// ============================================================
// Compile-time statistics sampling schedule
// ============================================================

template <label_t Start, label_t Every>
[[nodiscard]] constexpr __forceinline__ bool statisticsSampleDue(const label_t step) noexcept
{
    static_assert(Every > static_cast<label_t>(0));

    if constexpr (Start > static_cast<label_t>(0))
    {
        return step >= Start && (step - Start) % Every == static_cast<label_t>(0);
    }
    else
    {
        return step % Every == static_cast<label_t>(0);
    }
}

static_assert(statisticsSampleDue<static_cast<label_t>(0), static_cast<label_t>(1)>(static_cast<label_t>(0)));
static_assert(!statisticsSampleDue<static_cast<label_t>(3), static_cast<label_t>(2)>(static_cast<label_t>(2)));
static_assert(statisticsSampleDue<static_cast<label_t>(3), static_cast<label_t>(2)>(static_cast<label_t>(3)));
static_assert(statisticsSampleDue<static_cast<label_t>(3), static_cast<label_t>(2)>(static_cast<label_t>(5)));

#endif // STATISTICS_SAMPLING_CUH
