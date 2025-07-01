# Common auto-tuning

In general, we can set a range for each `BLOCK_SIZE` parameter and run `build_kernel.py` to get ELFs with different `BLOCK_SIZE`, for instance, in the `benchmark/src/triton/flash_attention_fwd.py`, we have:

```py

def get_flash_attention_fwd_kernel_autotune_config(DMODEL, num_threads=0):
    # Because BLOCK_SIZE_DMODEL must be D_HEAD, and D_HEAD is one of the dimensions of input, 
    # we should not change BLOCK_SIZE_DMODEL.
    configs = []
    
    block_sizes_M = [4, 8, 16, 32, 64]
    block_sizes_N = [8, 16, 32, 64]

    for block_m in block_sizes_M:
        for block_n in block_sizes_N:
            configs.append(
                triton.Config({
                            'BLOCK_SIZE_M': block_m,
                            'BLOCK_SIZE_N': block_n,
                            'BLOCK_SIZE_DMODEL': DMODEL,
                            'IS_CAUSAL': False
                        }, num_threads=num_threads)
                )
            configs.append(
                triton.Config({
                            'BLOCK_SIZE_M': block_m,
                            'BLOCK_SIZE_N': block_n,
                            'BLOCK_SIZE_DMODEL': DMODEL,
                            'IS_CAUSAL': False
                        }, num_threads=num_threads)
                )
    return configs

def benchmark_triton(q, k, v, sm_scale, DMODEL, parallel=False):

    fn = flash_attention_fwd_kernel # Here is the kernel name
    fn_jit = triton.jit(fn)
    fn_jit_tuned = triton.runtime.Autotuner(fn_jit, fn_jit.arg_names, 
        reset_to_zero=None, 
        restore_value=None,
        configs=get_flash_attention_fwd_kernel_autotune_config(DMODEL, num_threads=0 if parallel else 1),
        key=[],
    )

    # Ensure inputs are 4D: (Batch, Heads, SeqLen, HeadDim)
    assert q.dim() == k.dim() == v.dim() == 4

    Z,   H,   N_CTX_q, D_HEAD   = q.shape
    Z_k, H_k, N_CTX_k, D_HEAD_k = k.shape
    Z_v, H_v, N_CTX_v, D_HEAD_v = v.shape

    # Check that dimensions match where they should
    assert Z == Z_k == Z_v and H == H_k == H_v and D_HEAD == D_HEAD_k == D_HEAD_v
    assert N_CTX_k == N_CTX_v  # SeqLen_K == SeqLen_V

    # Convert to Triton-compatible device and dtype
    q = q.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    k = k.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    v = v.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)

    output = torch.empty_like(q)
    # M_LogSumExp is used for numerical stability checks or backward pass
    M_LogSumExp = torch.empty((Z, H, N_CTX_q), device=DEVICE, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(N_CTX_q, META["BLOCK_SIZE_M"]), Z * H,
    )

    def run_triton_kernel():
        # don't need to include the Meta parameters in the call
        # to the kernel, they are already included in the config
        fn_jit_tuned[grid](
            q, k, v, output, M_LogSumExp,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            H, N_CTX_q, D_HEAD,
            sm_scale
        )

    run_triton_kernel() # generate IR for all configs

```

By running `python3 shell/build_kernel.py -t benchmark/src/triton/flash_attention_fwd.py -j benchmark/config.json`, it would generate different kernel IR with corresponding `BLOCK_SIZE` parameters firstly, then compile kernel IR and main cpp(under `benchmark/src/main/fused_moe_kernel.cpp`) and link them together to generate a complete ELF.

And for the main cpp, it can read all `BLOCK_SIZE` parameters to control how to launch Triton kernel, because it would calculate how many grids we need to launch, also read input and expected output data from TXT files. Usually, the number of grids is dependent on `BLOCK_SIZE_M`.

# Fused MoE kernel auto-tuning

Due to the particularity of MoE, the `BLOCK_SIZE_M` would influence the computation of MoE kernel (the function `moe_align_block_size` in `benchmark/src/triton/fused_moe.py`). Because it needs to be padded before computating, and the size to pad is decided by `BLOCK_SIZE_M`, for instance, for the tokens like [7, 5, 2, 2] and the `BLOCK_SIZE_M` of 4, it would be padded to [8, 8, 4, 4] to make each expert have the tokens of multiple of `BLOCK_SIZE_M`, while for the same size of tokens of [4, 4, 4, 4], it does not need to pad, but they have the different output.

In order to generate correct ELFs with different `BLOCK_SIZE_M`, we have to prepare different input and output data, and change the way to launch kernel, because we have to decide how many grids we need, but in MoE, its grids number is dependent on the padded data size and `BLOCK_SIZE_M` simultaneously, so it is not convenient to write general main cpp.

One good thing is, we have all input and output data files, and there are no too many variants. For example, `matrix_BLOCK_SIZE_M_16_1000x2x512_3.txt`, `matrix_BLOCK_SIZE_M_16_1x2064_4.txt` `matrix_BLOCK_SIZE_M_16_1x129_5.txt` are respectively expected output data, `sorted_token_ids` and `expert_ids` when the `BLOCK_SIZE_M` is 16.

Our task is:

- to write correct main cpp to fit different `BLOCK_SIE_M` and run `build_kernel.py` to get complete ELFs. 

- run all ELFs on RISC-V laptop and get perf statistic results(see `T_RVV/README.md`).
