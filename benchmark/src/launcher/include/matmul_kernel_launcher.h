
#include <stdint.h>
#include <cstddef>
using matmul_kernel_kernel_ptr_t = void(*)(void*, void*, void*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C"{
// Pointer type (=Memref) becomes int64_t + MemRef struct
// FIXME: understand what this int64_t is used for.
void(matmul_kernel)(void*, void*, void*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
}

extern const int matmul_kernel_BLOCK_SIZE_M;
extern const int matmul_kernel_BLOCK_SIZE_N;
extern const int matmul_kernel_BLOCK_SIZE_K;
extern const int matmul_kernel_GROUP_SIZE_M;


void matmul_kernel_wrap(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_threads,
                        matmul_kernel_kernel_ptr_t kernel_ptr , void* arg0, void* arg1, void* arg2, int32_t arg3, int32_t arg4, int32_t arg5, int32_t arg6, int32_t arg8, int32_t arg10);
    