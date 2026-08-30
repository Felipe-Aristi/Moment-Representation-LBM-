#!/bin/bash

set -e

nvcc src/main.cu src/kernels.cu \
    -o mr_lbm \
    -O3 \
    -std=c++20 \
    --restrict \
    --extended-lambda \
    --expt-relaxed-constexpr \
    -arch=sm_89 \
    -lineinfo \
    -Xptxas=-v,-warn-spills,-warn-lmem-usage

./mr_lbm