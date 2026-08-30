#ifndef CONSTEXPRFOR_CUH
#define CONSTEXPRFOR_CUH

#include <cstddef>
#include <cmath>
#include <type_traits>
#include <utility>
#include "types.cuh"

// Constexpr_for ---> for esquisito pra os amigos

template <typename T, T V>
using integralConstant = std::integral_constant<T, V>;

template <const label_t Start, const label_t End, typename F>
__device__ inline constexpr void constexpr_for(F &&f) noexcept
{
    if constexpr (Start < End)
    {
        f(integralConstant<label_t, Start>());
        if constexpr (Start + 1 < End)
        {
            constexpr_for<Start + 1, End>(std::forward<F>(f));
        }
    }
}

#endif