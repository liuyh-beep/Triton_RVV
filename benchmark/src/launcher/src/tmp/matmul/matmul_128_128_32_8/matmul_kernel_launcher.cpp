
#include "matmul_kernel_launcher.h"
#include "support/omp.h"
#include "support/support.h"
#include <algorithm>
#include <optional>
#include <stdio.h>

const int matmul_kernel_BLOCK_SIZE_M = 128;
const int matmul_kernel_BLOCK_SIZE_N = 128;
const int matmul_kernel_BLOCK_SIZE_K = 32;
const int matmul_kernel_GROUP_SIZE_M = 8;


void matmul_kernel_wrap(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_threads, matmul_kernel_kernel_ptr_t kernel_ptr , void* arg0, void* arg1, void* arg2, int32_t arg3, int32_t arg4, int32_t arg5, int32_t arg6, int32_t arg8, int32_t arg10) {
    // TODO: Consider using omp collapse(3) clause for simplicity?
    size_t N = gridX * gridY * gridZ;
    if (N == 1) {
        (*kernel_ptr)(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg8, arg10,  0, 0, 0, 1, 1, 1);
        return;
    }
    auto all_grids = get_all_grids(gridX, gridY, gridZ);
    int omp_max_threads = 1;
#ifdef _OPENMP
    omp_max_threads = omp_get_max_threads();
#endif // _OPENMP
    int max_threads = (num_threads > 0) ? num_threads : omp_max_threads;

    // Don't pay OMP overhead price when a single thread is used.
    if (max_threads == 1) {
        for (size_t i = 0; i < N; ++i) {
        const auto [x, y, z] = all_grids[i];
        (*kernel_ptr)(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg8, arg10,  x, y, z, gridX, gridY, gridZ);
        }
        return;
    }

    // For now, use the default chunk size, total iterations / max_threads.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(max_threads)
#endif // _OPENMP
    for (size_t i = 0; i < N; ++i) {
        const auto [x, y, z] = all_grids[i];
        (*kernel_ptr)(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg8, arg10,  x, y, z, gridX, gridY, gridZ);
    }
}
    