#ifdef C_KERNEL_ENABLE
#include "kernel/attention.h" // Assumes void attention_cpp(...) is declared
#endif

#ifdef TRITON_KERNEL_ENABLE
#include "atten_matmul_kernel_launcher.h"
#include "transpose_block_kernel_launcher.h"
#include "softmax_kernel_launcher.h"
#endif

#include "support/support.h"

#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <cmath> // For sqrtf

using namespace std;
using std::chrono::high_resolution_clock;

// --- Main Test Harness for Attention Forward Pass ---
int main(int argc, char *argv[]) {
    // Default parameters
    int Z = 4;    // Batch Size
    int H = 32;    // Number of Heads
    int N_CTX = 1024; // Sequence Length
    int D_HEAD = 64;   // Dimension of Head
    int RUN_COUNT = 5; // Reduced for faster testing

    // --- Argument Parsing ---
    // Example: "ZxHxN_CTX_QxN_CTX_KxD_HEADxRUN_COUNT" -> "1x2x128x128x64x3"
    if (argc >= 2) {
        std::vector<int> Shape = splitStringToInts(argv[1]);
        if (Shape.size()) {
            assert(Shape.size() == 6 && "Invalid shape: ZxHxN_CTXxD_HEADxRUN_COUNT\n");
            Z = Shape.at(0); H = Shape.at(1); N_CTX = Shape.at(2);
            D_HEAD = Shape.at(4); RUN_COUNT = Shape.at(5);
        }
    }
    
    
    printf("Attention Fwd Data: Z=%d, H=%d, N_CTX=%d, D_HEAD=%d, RUN_COUNT=%d\n",
           Z, H, N_CTX, D_HEAD, RUN_COUNT);

    float sm_scale = 1.0f / sqrtf(static_cast<float>(D_HEAD));

    // --- Memory Allocation ---
    size_t q_elements = (size_t)Z * H * N_CTX * D_HEAD;
    size_t k_elements = (size_t)Z * H * N_CTX * D_HEAD;
    size_t v_elements = (size_t)Z * H * N_CTX * D_HEAD;
    size_t out_elements = (size_t)Z * H * N_CTX * D_HEAD;
    size_t k_transposed_elements = (size_t)Z * H * D_HEAD * N_CTX; // K^T
    size_t scores_elements = (size_t)Z * H * N_CTX * N_CTX; // Q @ K^T
    size_t attn_weights_elements = (size_t)Z * H * N_CTX * N_CTX; // softmax(scores)

    float *q_ptr = (float *)malloc(q_elements * sizeof(float));
    float *k_ptr = (float *)malloc(k_elements * sizeof(float));
    float *v_ptr = (float *)malloc(v_elements * sizeof(float));
    float *ref_output_ptr = (float *)malloc(out_elements * sizeof(float));
    float *real_output_ptr = (float *)malloc(out_elements * sizeof(float));
    
    // Intermediate tensors for decomposed attention
    float *k_transposed_ptr = (float *)malloc(k_transposed_elements * sizeof(float));
    float *scores_ptr = (float *)malloc(scores_elements * sizeof(float));
    float *attn_weights_ptr = (float *)malloc(attn_weights_elements * sizeof(float));

    if (!q_ptr || !k_ptr || !v_ptr || !ref_output_ptr || !real_output_ptr || 
        !k_transposed_ptr || !scores_ptr || !attn_weights_ptr) {
        printf("ERROR: Memory allocation failed!\n");
        // Basic cleanup
        free(q_ptr); free(k_ptr); free(v_ptr); free(ref_output_ptr); free(real_output_ptr);
        free(k_transposed_ptr); free(scores_ptr); free(attn_weights_ptr);
        return -1;
    }
    
    memset(ref_output_ptr, 0, out_elements * sizeof(float));
    memset(real_output_ptr, 0, out_elements * sizeof(float));
    memset(k_transposed_ptr, 0, k_transposed_elements * sizeof(float));
    memset(scores_ptr, 0, scores_elements * sizeof(float));
    memset(attn_weights_ptr, 0, attn_weights_elements * sizeof(float));

    // --- Data Initialization ---
#ifdef CHECK_ACCURACY
    int tmp_m = Z * H * N_CTX, tmp_n = D_HEAD;
    std::string file1 = getDB("attention_fwd", std::to_string(Z) + "x" + std::to_string(H) + "x" + std::to_string(N_CTX) + "x" + std::to_string(D_HEAD), 1);
    if (!readMatrix(file1.c_str(), q_ptr, tmp_m, tmp_n)) {
        printf("Failed to read q_ptr from %s\n", file1.c_str());
        return -1;
    }
    printf("q_ptr (%dx%dx%dx%d) loaded from %s\n", Z, H, N_CTX, D_HEAD, file1.c_str());

    std::string file4 = getDB("attention_fwd", std::to_string(Z) + "x" + std::to_string(H) + "x" + std::to_string(N_CTX) + "x" + std::to_string(D_HEAD), 4);
    if (!readMatrix(file4.c_str(), ref_output_ptr, tmp_m, tmp_n)) {
        printf("Failed to read ref_output from %s\n", file4.c_str());
        return -1;
    }
    printf("ref_output_ptr (%dx%dx%dx%d) loaded from %s\n", Z, H, N_CTX, D_HEAD, file4.c_str());

    std::string file2 = getDB("attention_fwd", std::to_string(Z) + "x" + std::to_string(H) + "x" + std::to_string(N_CTX) + "x" + std::to_string(D_HEAD), 2);
    if (!readMatrix(file2.c_str(), k_ptr, tmp_m, tmp_n)) {
        printf("Failed to read k_ptr from %s\n", file2.c_str());
        return -1;
    }
    printf("k_ptr (%dx%dx%dx%d) loaded from %s\n", Z, H, N_CTX, D_HEAD, file2.c_str());

    std::string file3 = getDB("attention_fwd", std::to_string(Z) + "x" + std::to_string(H) + "x" + std::to_string(N_CTX) + "x" + std::to_string(D_HEAD), 3);
    if (!readMatrix(file3.c_str(), v_ptr, tmp_m, tmp_n)) {
        printf("Failed to read v_ptr from %s\n", file3.c_str());
        return -1;
    }
    printf("v_ptr (%dx%dx%dx%d) loaded from %s\n", Z, H, N_CTX, D_HEAD, file3.c_str());
#endif

#ifdef C_KERNEL_ENABLE
    printf("Executing C Attention Fwd Kernel %d times...\n", RUN_COUNT);
    auto c_fwd_begin_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUN_COUNT; i++) {
        attention_cpp(q_ptr, k_ptr, v_ptr, real_output_ptr,
                      Z, H, N_CTX, D_HEAD, sm_scale);
    }
    auto c_fwd_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> c_fwd_time_interval = c_fwd_end_time - c_fwd_begin_time;
    PRINT_KERNEL_RUNNING_TIME(C_KERNEL, c_fwd_time_interval.count() / RUN_COUNT)
#endif

#ifdef TRITON_KERNEL_ENABLE
    int num_thread = 1;
#ifdef SINGLE_BLOCK
    num_thread = -1;
    printf("Not complete loop, actual thread num: %d\n", num_thread);
#endif
#ifdef SINGLE_ITERATION
    printf("Single iteration mode enabled, num_thread = %d\n", num_thread);
#endif

    // Strides for contiguous tensors (Z, H, N_CTX, D_HEAD)
    int stride_qz = H * N_CTX * D_HEAD, stride_qh = N_CTX * D_HEAD, stride_qm = D_HEAD;
    int stride_kz = H * N_CTX * D_HEAD, stride_kh = N_CTX * D_HEAD, stride_kn = D_HEAD;
    // K^T: (Z, H, D_HEAD, N_CTX)
    int stride_ktz = H * D_HEAD * N_CTX, stride_kth = D_HEAD * N_CTX, stride_ktd = N_CTX;
    int stride_vz = H * N_CTX * D_HEAD, stride_vh = N_CTX * D_HEAD, stride_vn = D_HEAD;
    // Scores/Attention: (Z, H, N_CTX, N_CTX)
    int stride_sz = H * N_CTX * N_CTX, stride_sh = N_CTX * N_CTX, stride_sm = N_CTX;
    int stride_oz = H * N_CTX * D_HEAD, stride_oh = N_CTX * D_HEAD, stride_om = D_HEAD;

    // Step 1: Transpose K to get K^T
    // Grid from Triton: (triton.cdiv(N_CTX, BLOCK_SIZE_M) * Z * H, )
    int transpose_grid_dim0 = ((N_CTX + transpose_block_kernel_BLOCK_SIZE_M - 1) / transpose_block_kernel_BLOCK_SIZE_M) * Z * H;
    
    printf("Step 1: Transposing K...\n");
    transpose_block_kernel_wrap(
        transpose_grid_dim0, 1, 1,
        num_thread,
        transpose_block_kernel,
        k_ptr, k_transposed_ptr,
        Z, H, N_CTX, D_HEAD,
        stride_kz, stride_kh, stride_kn,  // K strides
        stride_ktz, stride_kth, stride_ktd  // K^T strides
    );

    // Step 2: Q @ K^T -> Scores
    // Grid from Triton: (triton.cdiv(N_CTX, BLOCK_SIZE_M) * triton.cdiv(N_CTX, BLOCK_SIZE_K), Z * H)
    int matmul1_grid_dim0 = ((N_CTX + matmul_kernel_BLOCK_SIZE_M - 1) / matmul_kernel_BLOCK_SIZE_M) * 
                           ((N_CTX + matmul_kernel_BLOCK_SIZE_K - 1) / matmul_kernel_BLOCK_SIZE_K);
    int matmul1_grid_dim1 = Z * H;

    printf("Step 2: Computing Q @ K^T...\n");
    atten_matmul_kernel_wrap(
        matmul1_grid_dim0, matmul1_grid_dim1, 1,
        num_thread,
        matmul_kernel,
        q_ptr, k_transposed_ptr, scores_ptr,
        N_CTX, D_HEAD, N_CTX,  // M, K, N dimensions
        Z, H,
        stride_qm,  // Q stride for last two dims (stride(2))
        stride_ktd,  // K^T stride for last two dims  
        stride_sm,   // S stride for last two dims
        true, sm_scale
    );

    // Step 3: Softmax on scores -> attention weights
    // Grid from Triton: (N_CTX, Z * H)
    int softmax_grid_dim0 = N_CTX;
    int softmax_grid_dim1 = Z * H;

    printf("Step 3: Computing softmax...\n");
    softmax_kernel_wrap(
        softmax_grid_dim0, softmax_grid_dim1, 1,
        num_thread,
        softmax_kernel,
        scores_ptr, attn_weights_ptr,
        Z, H, N_CTX,
        stride_sm,  // S stride for last two dims
        stride_sm   // P stride for last two dims (same as S)
    );

    // Step 4: Attention weights @ V -> Output
    // Grid from Triton: (triton.cdiv(N_CTX, BLOCK_SIZE_M) * triton.cdiv(D_HEAD, BLOCK_SIZE_N), Z * H)
    int matmul2_grid_dim0 = ((N_CTX + matmul_kernel_BLOCK_SIZE_M - 1) / matmul_kernel_BLOCK_SIZE_M) * 
                           ((D_HEAD + matmul_kernel_BLOCK_SIZE_N - 1) / matmul_kernel_BLOCK_SIZE_N);
    int matmul2_grid_dim1 = Z * H;

    printf("Step 4: Computing Attention @ V...\n");
    atten_matmul_kernel_wrap(
        matmul2_grid_dim0, matmul2_grid_dim1, 1,
        num_thread,
        matmul_kernel,
        attn_weights_ptr, v_ptr, real_output_ptr,
        N_CTX, N_CTX, D_HEAD,  // M, K, N dimensions
        Z, H,
        stride_sm,  // P stride for last two dims
        stride_vn,  // V stride for last two dims
        stride_om,   // Output stride for last two dims
        false, sm_scale
    );

    // --- Accuracy Check ---
#ifdef CHECK_ACCURACY
    printf("Check correctness after first run...\n");
    check_tensor(ref_output_ptr, real_output_ptr, out_elements, "attention_fwd_output");
#endif
    memset(real_output_ptr, 0, out_elements * sizeof(float));
    memset(k_transposed_ptr, 0, k_transposed_elements * sizeof(float));
    memset(scores_ptr, 0, scores_elements * sizeof(float));
    memset(attn_weights_ptr, 0, attn_weights_elements * sizeof(float));

    printf("Executing Triton Attention Fwd Kernel %d times...\n", RUN_COUNT);
    auto triton_fwd_begin_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUN_COUNT; i++) {
        // Step 1: Transpose K
        transpose_block_kernel_wrap(
            transpose_grid_dim0, 1, 1,
            num_thread,
            transpose_block_kernel,
            k_ptr, k_transposed_ptr,
            Z, H, N_CTX, D_HEAD,
            stride_kz, stride_kh, stride_kn,
            stride_ktz, stride_kth, stride_ktd
        );

        // Step 2: Q @ K^T
        matmul_kernel_wrap(
            matmul1_grid_dim0, matmul1_grid_dim1, 1,
            num_thread,
            matmul_kernel,
            q_ptr, k_transposed_ptr, scores_ptr,
            N_CTX, D_HEAD, N_CTX,
            Z, H,
            stride_qm,
            stride_ktd,
            stride_sm,
            true, sm_scale
        );

        // Step 3: Softmax
        softmax_kernel_wrap(
            softmax_grid_dim0, softmax_grid_dim1, 1,
            num_thread,
            softmax_kernel,
            scores_ptr, attn_weights_ptr,
            Z, H, N_CTX,
            stride_sm,
            stride_sm
        );

        // Step 4: Attention @ V
        matmul_kernel_wrap(
            matmul2_grid_dim0, matmul2_grid_dim1, 1,
            num_thread,
            matmul_kernel,
            attn_weights_ptr, v_ptr, real_output_ptr,
            N_CTX, N_CTX, D_HEAD,
            Z, H,
            stride_sm,
            stride_vn,
            stride_om,
            false, sm_scale
        );
    }
    auto triton_fwd_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> triton_fwd_time_interval = triton_fwd_end_time - triton_fwd_begin_time;
    PRINT_KERNEL_RUNNING_TIME(TRITON_KERNEL, triton_fwd_time_interval.count() / RUN_COUNT)
#endif

    // --- Save Test Data ---
#ifdef KEEP_TEST_DATA
    printf("Mode: KEEP_TEST_DATA. Saving data for Attention (not fully implemented here)...\n");
#endif

    // --- Cleanup ---
    free(q_ptr); 
    free(k_ptr); 
    free(v_ptr);
    free(ref_output_ptr);
    free(real_output_ptr);
    free(k_transposed_ptr);
    free(scores_ptr);
    free(attn_weights_ptr);

    return 0;
}