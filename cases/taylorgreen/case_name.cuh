#ifndef TAYLORGREEN_CASE_NAME_CUH
#define TAYLORGREEN_CASE_NAME_CUH

#include <string>
#include "../../src/constants.cuh"

[[nodiscard]] inline std::string taylorGreenCaseName()
{
    return "TG_RE_" + std::to_string(static_cast<label_t>(tgv_Re)) + "_"
        + std::to_string(NXPhysical) + "x" + std::to_string(NYPhysical)
        + "x" + std::to_string(NZPhysical);
}

#endif
