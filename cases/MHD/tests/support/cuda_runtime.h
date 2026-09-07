#ifndef MHD_CPU_TEST_CUDA_RUNTIME_H
#define MHD_CPU_TEST_CUDA_RUNTIME_H

#ifndef MHD_CPU_TEST
#error "This allocation shim is only for the CPU numerical tests, never for a CUDA build."
#endif
#include <cstdlib>
#include <cstring>

// Only allocation/copy APIs required by shared headers. No GPU or kernel emulation.
using cudaError_t = int;
inline constexpr cudaError_t cudaSuccess = 0;
enum cudaMemcpyKind { cudaMemcpyDeviceToHost };
inline cudaError_t cudaMalloc(void **p, std::size_t n) { *p = std::malloc(n); return *p ? 0 : 1; }
inline cudaError_t cudaFree(void *p) { std::free(p); return 0; }
inline cudaError_t cudaMemset(void *p, int value, std::size_t n) { std::memset(p, value, n); return 0; }
inline cudaError_t cudaMemcpy(void *to, const void *from, std::size_t n, cudaMemcpyKind) { std::memcpy(to, from, n); return 0; }
inline const char *cudaGetErrorString(cudaError_t) { return "CPU test allocation failure"; }
#endif
