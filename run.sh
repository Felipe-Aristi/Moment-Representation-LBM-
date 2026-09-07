#!/bin/bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

case_name="${1:-taylorgreen}"
if (( $# > 0 )); then shift; fi
case "$case_name" in
    taylorgreen)
        sources=(src/main.cu src/kernels.cu)
        executable=mr_lbm
        definitions=()
        ;;
    mhd|MHD)
        sources=(cases/MHD/main.cu cases/MHD/kernels.cu)
        executable=mr_lbm_mhd
        definitions=(-DMRLBM_MHD)
        # Optional compile-time overrides, split into arguments without eval.
        read -r -a overrides <<< "${MHD_NVCC_FLAGS:-}"
        definitions+=("${overrides[@]}")
        ;;
    *)
        echo "Usage: bash run.sh [taylorgreen|mhd] [device_id] [new_output_directory]" >&2
        exit 2
        ;;
esac

"${NVCC:-nvcc}" "${sources[@]}" -o "$executable" \
    -O3 -std=c++20 --restrict --extended-lambda --expt-relaxed-constexpr \
    "-arch=${CUDA_ARCH:-sm_89}" -lineinfo \
    -Xptxas=-v,-warn-spills,-warn-lmem-usage "${definitions[@]}"

if [[ "${BUILD_ONLY:-0}" != 1 ]]; then
    "./$executable" "$@"
fi
