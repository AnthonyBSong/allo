// Auto-raised from ../test_irs/systolic.ir by mesh
#pragma once
#include <cstdint>
#include "utils.cuh"

__global__ void systolic_gemm_16_kernel(
    Memref<float, 4, 4> arg0,
    Memref<float, 4, 4> arg1,
    Memref<float, 4, 4> arg2) {
  // @systolic_gemm_16.pe over grid [6, 6], packed to 9 warp(s) / 288 thread(s), 40 PE-to-PE edge(s)
  const int ROW[288] = { 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  const int COL[288] = { 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  const int wid_r = ROW[threadIdx.x];
  const int wid_c = COL[threadIdx.x];
  
  __shared__ Stream<float, 4> systolic_gemm_16_fifo_A[6][6];
  __shared__ Stream<float, 4> systolic_gemm_16_fifo_B[6][6];
  if (threadIdx.x == 0) {
    for (int i0 = 0; i0 < 6; ++i0) for (int i1 = 0; i1 < 6; ++i1) systolic_gemm_16_fifo_A[i0][i1].init();
    for (int i0 = 0; i0 < 6; ++i0) for (int i1 = 0; i1 < 6; ++i1) systolic_gemm_16_fifo_B[i0][i1].init();
  }
  __syncthreads();
  
  if (wid_r >= 0) {
    if (((wid_r == 0) && (wid_c == 0))) {
      0.000000e+00f;
    } else {
      if (((wid_r == 5) && (wid_c == 5))) {
        0.000000e+00f;
      } else {
        if ((wid_c == 0)) {
          if (((wid_r != 0) && (wid_r != 5))) {
            for (int arg3 = 0; arg3 < 4; ++arg3) {
              systolic_gemm_16_fifo_A[wid_r][wid_c + 1].put(arg0(wid_r - 1, arg3));
            }
          }
        } else {
          if ((wid_r == 0)) {
            if (((wid_c != 0) && (wid_c != 5))) {
              for (int arg3 = 0; arg3 < 4; ++arg3) {
                systolic_gemm_16_fifo_B[wid_r + 1][wid_c].put(arg1(arg3, wid_c - 1));
              }
            }
          } else {
            if ((wid_r == 5)) {
              if (((wid_c != 0) && (wid_c != 5))) {
                for (int arg3 = 0; arg3 < 4; ++arg3) {
                  systolic_gemm_16_fifo_B[wid_r][wid_c].get();
                }
              }
            } else {
              if ((wid_c == 5)) {
                if (((wid_r != 0) && (wid_r != 5))) {
                  for (int arg3 = 0; arg3 < 4; ++arg3) {
                    systolic_gemm_16_fifo_A[wid_r][wid_c].get();
                  }
                }
              } else {
                float acc = 0.000000e+00f;
                for (int arg3 = 0; arg3 < 4; ++arg3) {
                  float a = systolic_gemm_16_fifo_A[wid_r][wid_c].get();
                  float b = systolic_gemm_16_fifo_B[wid_r][wid_c].get();
                  float prod = (a * b);
                  acc = (acc + prod);
                  systolic_gemm_16_fifo_A[wid_r][wid_c + 1].put(a);
                  systolic_gemm_16_fifo_B[wid_r + 1][wid_c].put(b);
                }
                arg2(wid_r - 1, wid_c - 1) = acc;
              }
            }
          }
        }
      }
    }
  }
}

#ifdef __CUDACC__
__host__ inline void systolic_gemm_16(
    Memref<float, 4, 4> arg0,
    Memref<float, 4, 4> arg1,
    Memref<float, 4, 4> arg2) {
  systolic_gemm_16_kernel<<<1, 288>>>(arg0, arg1, arg2);
}
#endif // __CUDACC__

