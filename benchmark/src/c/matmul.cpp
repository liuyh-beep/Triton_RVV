#include "kernel/matmul.h"
#include "support/omp.h"
#include "support/support.h"

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32

void matmul(float *arg0, float *arg1, float *arg2, int M, int N, int K) {
  // Initialize output matrix to zero
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      arg2[i * N + j] = 0;
    }
  }

  for (int i = 0; i < M; i += BLOCK_SIZE_M) {
    int i_end = std::min(i + BLOCK_SIZE_M, M);
    for (int j = 0; j < N; j += BLOCK_SIZE_N) {
      int j_end = std::min(j + BLOCK_SIZE_N, N);
      
      for(int k = 0; k < K; k += BLOCK_SIZE_K){
        int k_end = std::min(k + BLOCK_SIZE_K, K);
        // Notice: i-j-k order, to be the same as Triton matmul
        for(int ii = i; ii < i_end; ++ii){
          for(int jj = j; jj < j_end; ++jj){
            for(int kk = k; kk < k_end; ++kk){
              arg2[ii * N + jj] += arg0[ii * K + kk] * arg1[kk * N + jj];
            }
          }
        }

      }
    }
  }
}
