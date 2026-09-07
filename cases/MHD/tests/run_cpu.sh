#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
test_dir="$(mktemp -d)"
trap 'rm -rf -- "$test_dir"' EXIT
"${CXX:-g++}" -std=c++20 -O2 -Wall -Wextra -Wpedantic \
    -D__host__= -D__device__= -D__forceinline__=inline -DMHD_CPU_TEST -DMRLBM_MHD \
    -DMHD_NX=32 -DMHD_NY=32 "-DMHD_NZ=${MHD_TEST_NZ:-1}" -DMHD_RE=10 -DMHD_RM=10 \
    -Icases/MHD/tests/support cases/MHD/tests/numerics.cpp -o "$test_dir/numerics"
"$test_dir/numerics" "$test_dir/output"
python3 cases/MHD/tests/verify_output.py "$test_dir/output"
