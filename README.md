# Introduction

This branch enables auto-tuning for `BLOCK_SIZE` parameters which are used in kernel functions. After writting a new triton kernel, we can use the Triton's auto-tuning function to config all META data(like `BLOCK_SIZE: tl.constexpr`). And all `BLOCK` parameters would be writtern to its kernel header(e.g. `benchmark/src/launcher/include/matmul_kernel_launcher.h`).

# Workflow

4 steps: auto-tuning on host, transfer ELFs/configs to remote, run perf at remote and finally return perf report data to host.

## Auto-tuning

In order to get a lot of ELFs with different block configurations, we need to call `triton.runtime.Autotuner` in a Triton kernel. Currently modified Triton CPU can set different block sizes for auto tuning, but it would ignore all non-integer `tl.constexpr` in a kernel function if it is configurable. One example to set block sizes:

```python
'''benchmark/src/triton/matmul.py'''
# Do not need to add @triton.jit
def matmul_kernel(...):
    ...

# We can set BLOCK candicates here
def get_matmul_kernel_autotune_config(num_threads=0):
    configs = []
        
    block_sizes_M = [4, 8]
    block_sizes_N = [8, 16]
    block_sizes_K = [8]

    # Generate unique total block sizes from all combinations
    total_block_sizes = set()
    for block_m in block_sizes_M:
        for block_n in block_sizes_N:
            for block_k in block_sizes_K:
                total_block_sizes.add(block_m * block_n * block_k)
    
    # Sort the total block sizes for systematic exploration
    total_block_sizes = sorted(list(total_block_sizes))
    
    # For each total block size, find all combinations that achieve it
    for total_size in total_block_sizes:
        for block_m in block_sizes_M:
            for block_n in block_sizes_N:
                # Calculate the required block_k
                if total_size % (block_m * block_n) == 0:
                    block_k = total_size // (block_m * block_n)
                    # Only include if block_k is a valid block size
                    if block_k in block_sizes_K:
                        #print(f"Config: BLOCK_SIZE_M={block_m}, BLOCK_SIZE_N={block_n}, BLOCK_SIZE_K={block_k}")
                        configs.append(
                            triton.Config({
                                # Here all `BLOCK` should be the same as the ones in matmul_kernel()
                                'BLOCK_SIZE_M': block_m,
                                'BLOCK_SIZE_N': block_n,
                                'BLOCK_SIZE_K': block_k,
                                'GROUP_SIZE_M': GROUP_SIZE_M,
                                'USE_BLOCK_POINTERS': USE_BLOCK_POINTERS
                            }, num_threads=num_threads)
                        )
    
    return configs


def benchmark_triton(shape, a, b, parallel=False):

    fn = matmul_kernel
    fn_jit = triton.jit(fn)
    fn_jit_tuned = triton.runtime.Autotuner(fn_jit, fn_jit.arg_names, 
        reset_to_zero=None, 
        restore_value=None,
        configs=get_matmul_kernel_autotune_config(0 if parallel else 1),
        key=[],
    )

    M, N, K = shape
    c = torch.empty((M, N), dtype=torch.float32, device="cpu")
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    def run_triton_kernel():
        # don't need to include the Meta parameters in the call
        # to the kernel, they are already included in the config
        fn_jit_tuned[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1)
        )

    run_triton_kernel() # generate IR for all configs
```

With the correct main cpp(see *Code conventions*), we can run `shell/build_kernel.py`, and the way to run it remains the same(e.g. `python3 shell/build_kernel.py -t benchmark/src/triton/matmul.py -j benchmark/config.json`), it would run auto-tuning defined in `benchmark/src/triton/matmul.py:get_matmul_kernel_autotune_config`, and produce many subdirectories under `benchmark/src/launcher/src` with different `BLOCK` configurations. For example:

```bash
benchmark/src/launcher/src/
`-- matmul
    |-- matmul_4_16_8_8
    |   |-- blk_constants.json # all BLOCK parameters would be saved to a json
    |   |-- matmul_kernel.llir
    |   |-- matmul_kernel.ttcir
    |   |-- matmul_kernel.tttcir
    |   `-- matmul_kernel_launcher.cpp
    |-- matmul_4_8_8_8
    |   |-- blk_constants.json
    |   |-- matmul_kernel.llir
    |   |-- matmul_kernel.ttcir
    |   |-- matmul_kernel.tttcir
    |   `-- matmul_kernel_launcher.cpp
    |-- matmul_8_16_8_8
    |   |-- blk_constants.json
    |   |-- matmul_kernel.llir
    |   |-- matmul_kernel.ttcir
    |   |-- matmul_kernel.tttcir
    |   `-- matmul_kernel_launcher.cpp
    `-- matmul_8_8_8_8
        |-- blk_constants.json
        |-- matmul_kernel.llir
        |-- matmul_kernel.ttcir
        |-- matmul_kernel.tttcir
        `-- matmul_kernel_launcher.cpp
```

After that, it would read each subdirectory under `benchmark/src/launcher/src/{kernel_name}` to find all BLOCK configurations, and compile them to produce corresponding ELF, dumpped assembly code and a `configs.json` which records paths to read ELF and paths to write perf results under `benchmark/auto-tuner/{kernel_name}`. Like:

```bash
benchmark/auto-tuner/matmul
|-- dump
|   |-- VL256_matmul_4_16_8_g_static_O2.elf.s
|   |-- VL256_matmul_4_16_8_kernel_src.s
|   |-- VL256_matmul_4_8_8_g_static_O2.elf.s
|   |-- VL256_matmul_4_8_8_kernel_src.s
|   |-- VL256_matmul_8_16_8_g_static_O2.elf.s
|   |-- VL256_matmul_8_16_8_kernel_src.s
|   |-- VL256_matmul_8_8_8_g_static_O2.elf.s
|   `-- VL256_matmul_8_8_8_kernel_src.s
`-- run
    |-- bin
    |   |-- VL256_matmul_4_16_8_g_static_O2.elf
    |   |-- VL256_matmul_4_8_8_g_static_O2.elf
    |   |-- VL256_matmul_8_16_8_g_static_O2.elf
    |   `-- VL256_matmul_8_8_8_g_static_O2.elf
    |-- configs.json
    |-- perf
    |   |-- perf_data
    |   `-- perf_stats.csv
    `-- test_data # This directory is necessary, 
    # which should save test data into TXT files, 
    # and above ELF would read from these TXT when running
```

## Transfer to remote

The `shell/build_kernel.py` would not transfer any directory by default due to possible long auto-tuning and compilation. So there is a script `benchmark/auto-tuner/transfer_to_remote.py` to transfer all necessary files to remote. Usage: `python3 benchmark/auto-tuner/transfer_to_remote.py <kernel_name>`, and it would transfer `auto-tuner/{kernel_name}/run` to our laptop. One thing to notice is, `auto-tuner/{kernel_name}/run/test_data` is necessary, and **we need to generate test data and save them into that path manually so far**.

## Laptop side

After receiving `auto-tuner/{kernel_name}/run`, we can run `auto-tuner/run_perf.py` to get perf data, and it would produce a csv under `auto-tuner/{kernel_name}/run/perf/perf_stats.csv` on our laptop. And that csv file would record the kernel running time, some perf events(i.e. micro operations, cache, vector instructions) and perf record for L2 cache events. Also, `auto-tuner/run_perf.py` would not transfer this csv to host by default due to possible long running time, and we can use `auto-tuner/report_to_host.py` to copy that csv to host, usage: `python3 auto-tuner/transfer_to_remote.py <kernel_name>`.




# Perf metrics correlation analysis scripts(under development)


There is a script(`benchmark/auto-tuner/analysis_correlation.py`) to analyse perf metrics correlation between perf events and kernel running time, and it needs more tests.

# Code conventions

1. Kernel naming

    The kernel file name should be the same as the name of kernel function. And the name of kernel function should have a suffix of `_kernel`, like `flash_attention_fwd_kernel`, `matmul_kernel`.

2. Main cpp

    Current modified Triton CPU would write `BLOCK` parameters into kernel launcher headers(e.g. `benchmark/src/launcher/include/matmul_kernel_launcher.h`), and it would be named as `{kernel_name}_kernel_{BLOCK_para_name}`. So we can use them in main cpp to calculate the grid dimensions, and it is adaptive to auto-tuning. One example:

    ```c++

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
    ```


    Therefore, in its main cpp, we can define the grid as:

    ```c++
    #ifdef TRITON_KERNEL_ENABLE
    high_resolution_clock::time_point beginTime = high_resolution_clock::now();
    gridX = ceil(1.0 * M / matmul_kernel_BLOCK_SIZE_M) * ceil(1.0 * N / matmul_kernel_BLOCK_SIZE_N);
    for (int i = 0; i < RUN_COUNT; i++) {
        matmul_kernel_wrap(ceil(1.0 * M / matmul_kernel_BLOCK_SIZE_M) *
                            ceil(1.0 * N / matmul_kernel_BLOCK_SIZE_N),
                        1, 1, 1, matmul_kernel, arg0, arg1, real_out, M, N, K, K,
                        N, N);
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
    ```

