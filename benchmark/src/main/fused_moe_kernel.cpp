#include "support/support.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#ifdef C_KERNEL_ENABLE
// #include "kernel/v0_moe_fused.h" // C++ kernel implementation would be declared here
#endif

#ifdef TRITON_KERNEL_ENABLE
#include "fused_moe_kernel_launcher.h"
#endif

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
// TODO: read files, malloc memory, run kernel, check accuracy
//       and free memory in a single function to avoid code duplication

int main(int argc, char *argv[]) {
    
    // A: [M, K]
    // B: [E, N, K]
    // C: [M, TopK, N]
    // sorted_token_ids: [1, total_padded_tokens]
    // expert_ids: [1, total_padded_tokens / block_size_m]
    // num_tokens_post_padded_ptr: [1] = total_padded_tokens
    // EM = total_padded_tokens

    int M = 1000, E = 8, K = 128, N = 512, TopK = 2;
    int total_padded_tokens = 2008, expert_ids_size = total_padded_tokens / fused_moe_kernel_BLOCK_SIZE_M, num_valid_tokens = M * TopK;
    int RUN_COUNT = 1;

    printf("MoE MatMul Data: M=%d, E=%d, K=%d, N=%d, TopK=%d, total_padded_tokens=%d, expert_ids_size=%d, RUN_COUNT=%d\n",
           M, E, K, N, TopK, total_padded_tokens, expert_ids_size, RUN_COUNT);

    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(E * N * K * sizeof(float));
    float* ref_output = (float*)malloc(M * TopK * N * sizeof(float));
    float* real_output = (float*)malloc(M * TopK * N * sizeof(float));

    int* sorted_token_ids = (int*)malloc(total_padded_tokens * sizeof(int));
    int* expert_ids = (int*)malloc(expert_ids_size * sizeof(int));
    int* num_tokens_post_padded = (int*)malloc(sizeof(int));
    *num_tokens_post_padded = total_padded_tokens;

    if(!A || !B /*|| !ref_output*/ || !real_output || !sorted_token_ids || !expert_ids || !num_tokens_post_padded) {
        printf("ERROR: Memory allocation failed.\n");
        return -1;
    }

    memset(real_output, 0, M * TopK * N * sizeof(float));

    // --- Data Initialization from .txt files ---
#ifdef CHECK_ACCURACY
    std::string file1 = getDB("fused_moe", std::to_string(M) + "x" + std::to_string(K), 1);
    if (!readMatrix(file1.c_str(), A, M, K)) {
        printf("Failed to read matrix A\n");
        return -1;
    }
    printf("Matrix A (%dx%d) loaded from %s\n", M, K, file1.c_str());

    std::string file2 = getDB("fused_moe", std::to_string(E) + "x" + std::to_string(N) + "x" + std::to_string(K), 2);
    int tmp_m = E * N, tmp_n = K;
    if (!readMatrix(file2.c_str(), B, tmp_m, tmp_n)) {
        printf("Failed to read matrix B\n");
        return -1;
    }
    printf("Matrix B (%dx%dx%d) loaded from %s\n", E, N, K, file2.c_str());

    std::string file3 = getDB("fused_moe", std::to_string(M) + "x" + std::to_string(TopK) + "x" + std::to_string(N), 3);
    tmp_m = M * TopK, tmp_n = N;
    if (!readMatrix(file3.c_str(), ref_output, tmp_m, tmp_n)) {
        printf("Failed to read matrix ref_output\n");
        return -1;
    }
    printf("Matrix ref_output (%dx%dx%d) loaded from %s\n", M, TopK, N, file3.c_str());


    std::string file4 = getDB("fused_moe", std::to_string(1) + "x" + std::to_string(total_padded_tokens), 4);
    tmp_m = 1;
    if (!readMatrix(file4.c_str(),  sorted_token_ids, tmp_m, total_padded_tokens)) {
        printf("Failed to read matrix sorted_token_ids from %s\n", file4.c_str());
        return -1;
    }
    printf("Matrix sorted_token_ids (1x%d) loaded from %s\n", total_padded_tokens, file4.c_str());

    std::string file5 = getDB("fused_moe", std::to_string(1) + "x" + std::to_string(expert_ids_size), 5);
    if (!readMatrix(file5.c_str(), expert_ids, tmp_m, expert_ids_size)) {
        printf("Failed to read matrix expert_ids from %s\n", file5.c_str());
        return -1;
    }
    printf("Matrix expert_ids (1x%d) loaded from %s\n", expert_ids_size, file5.c_str());
#endif

    printf("Memory ranges:\n");
    printf("A: %p to %p\n", A, A + M * K);
    for(int i=0; i<5; ++i)
        printf("A[%d]: %f\n", i, A[i]);

    printf("B: %p to %p\n", B, B + E * N * K);
    for(int i=0; i<5; ++i)
        printf("B[%d]: %f\n", i, B[i]);

    // printf("C: %p to %p\n", real_output, real_output + M * TopK * N);
    // for(int i=0; i<5; ++i)
    //     printf("C[%d]: %f\n", i, real_output[i]);

    printf("ref_output: %p to %p\n", ref_output, ref_output + M * TopK * N);
    for(int i=0; i<5; ++i)
        printf("ref_output[%d]: %f\n", i, ref_output[i]);
    printf("sorted_token_ids: %p to %p\n", sorted_token_ids, sorted_token_ids + total_padded_tokens);
    for(int i=0; i<5; ++i)
        printf("sorted_token_ids[%d]: %d\n", i, sorted_token_ids[i]);
    printf("expert_ids: %p to %p\n", expert_ids, expert_ids + expert_ids_size);
    for(int i=0; i<5; ++i)
        printf("expert_ids[%d]: %d\n", i, expert_ids[i]);

#ifdef C_KERNEL_ENABLE
    // C++ kernel execution would go here.
#endif

#ifdef TRITON_KERNEL_ENABLE
    printf("Executing Triton MoE MatMul Kernel %d times...\n", RUN_COUNT);
    int num_thread = 1;
#ifdef SINGLE_BLOCK
  num_thread= -1;
  printf("Not complete loop, actual thread num: %d\n", num_thread);
#endif

#ifdef SINGLE_ITERATION
  printf("Single iteration mode enabled, num_thread = %d\n", num_thread);
#endif
    high_resolution_clock::time_point beginTime = high_resolution_clock::now();
    for (int i = 0; i < RUN_COUNT; i++) {
        /*
            a_ptr, b_ptr, c_ptr,
            sorted_token_ids_ptr,
            expert_ids_ptr,
            num_tokens_post_padded_ptr,
            # Matrix dimensions
            N, K, EM,
            num_valid_tokens,
            stride_am, stride_ak,
            stride_be, stride_bk, stride_bn,
            stride_cm, stride_cn,
        */
        fused_moe_kernel_wrap(
            ceil(1.0 * total_padded_tokens / fused_moe_kernel_BLOCK_SIZE_M) * ceil(1.0 * N / fused_moe_kernel_BLOCK_SIZE_N), 1, 1, 
            num_thread, fused_moe_kernel, 
            A, B, real_output,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            N, K, total_padded_tokens,
            num_valid_tokens,
            K, 
            N * K, K, 
            N);
    }
  high_resolution_clock::time_point endTime = high_resolution_clock::now();
  milliseconds timeInterval =
      std::chrono::duration_cast<milliseconds>(endTime - beginTime);

  std::chrono::duration<double> triton_correlation_time_interval =
      endTime - beginTime;
  /// NOTE: Format running time to generate performance report easily
  PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL,
                            triton_correlation_time_interval.count())

#endif

#ifdef CHECK_ACCURACY
    printf("Checking accuracy for MoE MatMul...\n");
    check_tensor(ref_output, real_output, M * TopK * N, "out");
#endif

  char filename[256];
  bool success = true;

  snprintf(filename, sizeof(filename), "real_output_%dx%dx%d_1.txt", M, TopK, N);
  if (!writeMatrix(filename, real_output, M * TopK, N)) {
      printf("Failed to save input matrix arg0\n");
  }

  printf("saved real_out\n");

    free(A);
    //printf("A is freed\n");

    free(B);
    //printf("B is freed\n");

    free(sorted_token_ids);
    //printf("sorted_token_ids is freed\n");

    free(expert_ids);
    //printf("expert_ids is freed\n");

    free(real_output);
    // printf("real_output is freed\n");

    free(ref_output);
    // printf("ref_output is freed\n");

    return 0;
}