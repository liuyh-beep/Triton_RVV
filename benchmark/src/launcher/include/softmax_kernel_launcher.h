
#include <stdint.h>
#include <cstddef>
using softmax_kernel_kernel_ptr_t = void(*)(void*, void*, int32_t, int32_t, int32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C"{
// Pointer type (=Memref) becomes int64_t + MemRef struct
// FIXME: understand what this int64_t is used for.
void(softmax_kernel)(void*, void*, int32_t, int32_t, int32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
}

extern const int softmax_kernel_BLOCK_SIZE;


void softmax_kernel_wrap(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_threads,
                        softmax_kernel_kernel_ptr_t kernel_ptr , void* arg0, void* arg1, int32_t arg2, int32_t arg3, int32_t arg4);
    