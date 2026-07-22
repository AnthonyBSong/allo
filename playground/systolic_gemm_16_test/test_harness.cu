// Host-side correctness test for systolic_gemm_16_kernel against the golden
// PE-dataflow reference computed in golden.py (see that file for why the
// golden is a Python simulation rather than an Allo-executed one).
//
// Build & run on a CUDA machine (not run yet -- no CUDA on the machine this
// was written on):
//   nvcc -O3 -arch=sm_80 test_harness.cu -o test_harness && ./test_harness
//
// Regenerate systolic_gemm_16.cu / utils.cuh from the mesh compiler with:
//   mesh ../test_irs/systolic.ir --target sm_80 > systolic_gemm_16.cu
// (run from this directory so utils.cuh lands alongside it)
// Regenerate test_data.h (new random A/B) with: python3 golden.py

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#include "systolic_gemm_16.cu"
#include "test_data.h"

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err__ = (expr);                                             \
    if (err__ != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err__));                                    \
      exit(1);                                                              \
    }                                                                        \
  } while (0)

int main() {
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * TEST_M * TEST_K));
  CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * TEST_K * TEST_N));
  CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * TEST_M * TEST_N));

  CUDA_CHECK(cudaMemcpy(d_A, test_A, sizeof(float) * TEST_M * TEST_K,
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, test_B, sizeof(float) * TEST_K * TEST_N,
                         cudaMemcpyHostToDevice));

  Memref<float, TEST_M, TEST_K> A{d_A};
  Memref<float, TEST_K, TEST_N> B{d_B};
  Memref<float, TEST_M, TEST_N> C{d_C};

  systolic_gemm_16(A, B, C);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float h_C[TEST_M][TEST_N];
  CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(float) * TEST_M * TEST_N,
                         cudaMemcpyDeviceToHost));

  bool pass = true;
  float max_abs_diff = 0.0f;
  for (int i = 0; i < TEST_M; ++i) {
    for (int j = 0; j < TEST_N; ++j) {
      float got = h_C[i][j];
      float want = test_C_expected[i][j];
      float diff = std::fabs(got - want);
      max_abs_diff = std::max(max_abs_diff, diff);
      // float32 accumulation in the same k-order as golden.py -- expect
      // near-exact agreement, not just GEMM-tolerance closeness.
      float tol = 1e-3f + 1e-4f * std::fabs(want);
      if (diff > tol) {
        pass = false;
        printf("MISMATCH C[%d][%d]: got %.6f want %.6f (diff %.6g)\n", i, j,
               got, want, diff);
      }
    }
  }

  printf("\nC (from systolic_gemm_16_kernel):\n");
  for (int i = 0; i < TEST_M; ++i) {
    for (int j = 0; j < TEST_N; ++j) printf("%12.6f", h_C[i][j]);
    printf("\n");
  }
  printf("\nmax abs diff vs golden: %.6g\n", max_abs_diff);
  printf(pass ? "\nPASS\n" : "\nFAIL\n");

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return pass ? 0 : 1;
}
