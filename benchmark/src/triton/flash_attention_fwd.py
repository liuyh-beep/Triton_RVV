import torch
import triton
import triton.language as tl
import math
import numpy as np
import os

DEVICE = 'cpu'
DTYPE_TRITON_KERNEL = tl.float32 # Kernel internal computation dtype
DTYPE_TORCH_INPUT = torch.float32  # Dtype for torch tensors
triton.runtime.driver.set_active_to_cpu()


'''The name of the kernel must match the file name for Triton to find it.'''

def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, # Tensor pointers
    M_LogSumExp_ptr, # Pointer to store m_i and l_i (combined for stability)
    
    stride_qz, stride_qh, stride_qm, stride_qk, # Strides for Q
    stride_kz, stride_kh, stride_km, stride_kk, # Strides for K (km for N_CTX)
    stride_vz, stride_vh, stride_vm, stride_vk, # Strides for V (vm for N_CTX)
    stride_oz, stride_oh, stride_om, stride_ok, # Strides for Out

    H, N_CTX, D_HEAD, # Dimensions: Batch, Heads, SeqLenQ, SeqLenK, HeadDim
    sm_scale, # Softmax scale factor (usually 1/sqrt(D_HEAD))
    
    BLOCK_SIZE_M: tl.constexpr, # Tile size for Q sequence length dimension
    BLOCK_SIZE_N: tl.constexpr, # Tile size for K sequence length dimension
    BLOCK_SIZE_DMODEL: tl.constexpr, # Head dimension (must be == D_HEAD)
    IS_CAUSAL: tl.constexpr
):
    # Program IDs
    start_m_idx = tl.program_id(axis=0) # Index of the current BLOCK_SIZE_M tile along N_CTX
    off_zh_idx = tl.program_id(axis=1)  # Combined batch and head index

    off_z = off_zh_idx // H
    off_h = off_zh_idx % H

    # Calculate offsets for Q, K, V, Out based on batch and head
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    out_offset = off_z * stride_oz + off_h * stride_oh
    m_logsumexp_offset = off_z * H * N_CTX + off_h * N_CTX # Assuming M_LogSumExp is [Z, H, N_CTX]

    # Pointers to Q, K, V, Out for the current batch/head
    Q_batch_head_ptr = Q_ptr + q_offset
    K_batch_head_ptr = K_ptr + k_offset
    V_batch_head_ptr = V_ptr + v_offset
    Out_batch_head_ptr = Out_ptr + out_offset
    M_LogSumExp_batch_head_ptr = M_LogSumExp_ptr + m_logsumexp_offset


    # Initialize accumulators and stats for the current Q_block
    # tl.zeros needs a shape, so if BLOCK_SIZE_M or BLOCK_SIZE_DMODEL is 1, handle carefully.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_DMODEL), dtype=DTYPE_TRITON_KERNEL)
    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=DTYPE_TRITON_KERNEL) # Running max
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=DTYPE_TRITON_KERNEL)              # Running sum of exps

    # Offsets for the current Q block (BLOCK_SIZE_M rows of queries)
    offs_m_q = start_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) # Rows in Q to process
    offs_d = tl.arange(0, BLOCK_SIZE_DMODEL)                 # Columns (head dimension)

    # Load current Q block
    # Q_block_ptr: base, shape, strides, offsets, block_shape, order
    q_ptrs = Q_batch_head_ptr + offs_m_q[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Mask for Q if N_CTX is not a multiple of BLOCK_SIZE_M
    mask_q_rows = offs_m_q < N_CTX
    q_block = tl.load(q_ptrs, mask=mask_q_rows[:, None] & (offs_d[None, :] < D_HEAD), other=0.0)
    q_block = q_block.to(DTYPE_TRITON_KERNEL)


    # Loop over K and V blocks (N_CTX dimension)
    # N_CTX is the sequence length of K and V
    for start_n_idx in range(0, tl.cdiv(N_CTX, BLOCK_SIZE_N)):
        offs_m_k = start_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # Rows in K/V to process in this block

        # Load K block (transposed for dot product with Q)
        # K is [D_HEAD, N_CTX] effectively for this head after transposing for QK^T
        # K_block will be [BLOCK_SIZE_DMODEL, BLOCK_SIZE_N]
        k_ptrs = K_batch_head_ptr + offs_d[:, None] * stride_kk + offs_m_k[None, :] * stride_km
        # Mask for K if N_CTX is not a multiple of BLOCK_SIZE_N
        mask_k_cols = offs_m_k < N_CTX
        k_block = tl.load(k_ptrs, mask=(offs_d[:, None] < D_HEAD) & mask_k_cols[None, :], other=0.0)
        k_block = k_block.to(DTYPE_TRITON_KERNEL)

        # Compute QK^T score for the block: (BLOCK_SIZE_M, D_MODEL) @ (D_MODEL, BLOCK_SIZE_N) -> (BLOCK_SIZE_M, BLOCK_SIZE_N)
        qk_scores = tl.dot(q_block, k_block) * sm_scale
        qk_scores = qk_scores.to(DTYPE_TRITON_KERNEL)

        # Apply causal mask if needed
        if IS_CAUSAL:
            # offs_m_q are global query indices for current Q block
            # offs_m_k are global key indices for current K block
            # For causality, q_idx >= k_idx
            causal_mask = offs_m_q[:, None] >= offs_m_k[None, :]
            qk_scores += tl.where(causal_mask, 0, float("-inf"))

        # Online softmax update
        m_i_prev = m_i
        m_i_block_max = tl.max(qk_scores, axis=1) # Max score for each query in Q_block against current K_block
        m_i = tl.maximum(m_i, m_i_block_max)    # m_i([BLOCK_SIZE_M,]) Update running max

        # Numerator: p_ij = exp(qk_scores - m_i_new)
        # Corrected for Triton: p_ij is actually exp(scores_ij - m_ij)
        # where m_ij is the max of current qk_scores and m_i_old
        # Let's use the standard FlashAttention update:
        p_block = tl.exp(qk_scores - m_i[:, None]) # exp([BLOCKS_SIZE_M, BLOCK_SIZE_N])
        p_block = p_block.to(DTYPE_TRITON_KERNEL)

        # Denominator update: l_i_new = l_i_old * exp(m_i_old - m_i_new) + sum(p_block_new_m, axis=1)
        alpha = tl.exp(m_i_prev - m_i) # m_i_prev is (BLOCK_SIZE_M,), m_i is (BLOCK_SIZE_M,)
        alpha = alpha.to(DTYPE_TRITON_KERNEL)
        
        acc = acc * alpha[:, None] # Scale existing accumulator
        l_i = l_i * alpha          # Scale existing l_i

        # Load V block
        # V_block will be [BLOCK_SIZE_N, BLOCK_SIZE_DMODEL]
        v_ptrs = V_batch_head_ptr + offs_m_k[:, None] * stride_vm + offs_d[None, :] * stride_vk
        mask_v_rows = offs_m_k < N_CTX
        v_block = tl.load(v_ptrs, mask=mask_v_rows[:, None] & (offs_d[None, :] < D_HEAD), other=0.0)
        v_block = v_block.to(DTYPE_TRITON_KERNEL) # Or Q.dtype.element_ty for dot product type

        # Update accumulator: acc += P_block @ V_block
        # p_block is (BLOCK_SIZE_M, BLOCK_SIZE_N), v_block is (BLOCK_SIZE_N, BLOCK_SIZE_DMODEL)
        acc = tl.dot(p_block.to(v_block.dtype), v_block, acc) # Add to scaled accumulator
        
        l_i += tl.sum(p_block, axis=1) # Update l_i with sum of current p_block; l_i([BLOCK_SIZE_M,])

    # Final normalization
    acc = acc / l_i[:, None]
    acc = acc.to(Q_ptr.dtype.element_ty) # Cast to output type

    # Store output block
    out_ptrs = Out_batch_head_ptr + offs_m_q[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=mask_q_rows[:, None] & (offs_d[None, :] < D_HEAD))

    # Store m_i (max for stability) and l_i (logsumexp for backward) if needed
    # Here, storing m_i + log(l_i) which is the log of the true denominator (logsumexp)
    # For simplicity if M_LogSumExp_ptr is only for m_i for stability for a fwd-only test,
    # we can just store m_i. If it's for backward, logsumexp is needed.
    # The original Triton code stores m_i + tl.math.log2(l_i) when using exp2.
    # For tl.exp, it would be m_i + tl.log(l_i)
    final_logsumexp = m_i + tl.log(l_i)
    m_logsumexp_store_ptrs = M_LogSumExp_batch_head_ptr + offs_m_q
    tl.store(m_logsumexp_store_ptrs, final_logsumexp, mask=mask_q_rows)


'''
Test data saving, check output, read test data, auto-tuning, modify llvm ir
'''



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

    # # Generate unique total block sizes from all combinations
    # total_block_sizes = set()
    # for block_m in block_sizes_M:
    #     for block_n in block_sizes_N:
    #         total_block_sizes.add(block_m * block_n)
    
    # # Sort the total block sizes for systematic exploration
    # total_block_sizes = sorted(list(total_block_sizes))
    
    # # For each total block size, find all combinations that achieve it
    # for total_size in total_block_sizes:
    #     for block_m in block_sizes_M:
    #         if total_size % block_m == 0:
    #             block_n = total_size // block_m 
    #             configs.append(
    #                     triton.Config({
    #                         'BLOCK_SIZE_M': block_m,
    #                         'BLOCK_SIZE_N': block_n,
    #                         'BLOCK_SIZE_DMODEL': 32,
    #                         'IS_CAUSAL': False
    #                     }, num_threads=num_threads)
    #                 )
    #             configs.append(
    #                     triton.Config({
    #                         'BLOCK_SIZE_M': block_m,
    #                         'BLOCK_SIZE_N': block_n,
    #                         'BLOCK_SIZE_DMODEL': 32,
    #                         'IS_CAUSAL': True
    #                     }, num_threads=num_threads)
    #                 )
    
    return configs


def benchmark_triton(q, k, v, sm_scale, DMODEL, parallel=False):

    fn = flash_attention_fwd_kernel
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


def flash_attention_fwd(q, k, v, sm_scale, BLOCK_M, BLOCK_N, is_causal=False):
    # Ensure inputs are 4D: (Batch, Heads, SeqLen, HeadDim)
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] # D_HEAD matches
    assert k.shape[2] == v.shape[2] # SeqLen_K == SeqLen_V

    Z, H, N_CTX_Q, D_HEAD = q.shape
    _Z_k, _H_k, _, _D_HEAD_k = k.shape
    _Z_v, _H_v, _, _D_HEAD_v = v.shape

    assert Z == _Z_k == _Z_v and H == _H_k == _H_v and D_HEAD == _D_HEAD_k == _D_HEAD_v

    # Convert to Triton-compatible device and dtype
    q = q.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    k = k.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    v = v.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)

    output = torch.empty_like(q)
    # M_LogSumExp is used for numerical stability checks or backward pass
    M_LogSumExp = torch.empty((Z, H, N_CTX_Q), device=DEVICE, dtype=torch.float32)

    # # Block sizes - these need tuning for CPU.
    # # BLOCK_M and BLOCK_N are tile sizes along sequence lengths.
    # # !! BLOCK_SIZE_DMODEL must be D_HEAD
    # BLOCK_M = 64  # Tile size for Q sequence length
    # BLOCK_N = 64  # Tile size for K sequence length
    # if D_HEAD > 128: # Example simple heuristic
    #     BLOCK_M = 32
    #     BLOCK_N = 32
    # elif D_HEAD <=32:
    #     BLOCK_M = 128
    #     BLOCK_N = 128


    # Grid: (num_q_blocks, num_batch_heads)
    grid = (triton.cdiv(N_CTX_Q, BLOCK_M), Z * H)

    kernel = triton.jit(flash_attention_fwd_kernel)

    print(f"Grid: {grid}, BLOCK_SIZE_M: {BLOCK_M}, BLOCK_SIZE_N: {BLOCK_N}")
    print(f"Q: {q.shape}, K: {k.shape}, V: {v.shape}, Out: {output.shape}")
    print(f"Strides Q: {q.stride()}, K: {k.stride()}, V: {v.stride()}, Out: {output.stride()}")


    kernel[grid](
        q, k, v, output, M_LogSumExp,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        H, N_CTX_Q, D_HEAD,
        sm_scale,
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_DMODEL=D_HEAD,
        IS_CAUSAL=is_causal
        # num_warps and num_stages are GPU-specific, not used for CPU
    )
    return output, M_LogSumExp


# PyTorch reference implementation for testing
def pytorch_attention(q, k, v, sm_scale, is_causal=False):
    # q, k, v: (Batch, Heads, SeqLen, HeadDim)
    # Transpose K for matmul: (B, H, D_HEAD, SeqLen_K)
    k_t = k.transpose(-2, -1)
    # Scores: (B, H, SeqLen_Q, SeqLen_K)
    scores = (q @ k_t) * sm_scale

    if is_causal:
        # Create a causal mask
        # Mask elements where key_idx > query_idx
        mask_value = -float('inf') if scores.dtype == torch.float32 else -1e4 # Approx -inf for float16
        rows = q.size(2)
        cols = k.size(2)
        causal_mask = torch.ones(rows, cols, dtype=torch.bool, device=q.device).tril(diagonal=0)
        # Expand mask to match scores shape (B, H, Sq, Sk)
        # For tril, if Sq != Sk, it applies to the min(Sq, Sk) x min(Sq,Sk) bottom-left submatrix
        # Here we want to mask where key_pos > query_pos
        # query_pos is indexed by M (rows of scores), key_pos by N (cols of scores)
        # So we want M_indices < N_indices to be masked. tril(0) gives M_indices >= N_indices
        scores = scores.masked_fill(~causal_mask[None, None, :, :], mask_value)

    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ v
    return output, attn_weights # Return weights for potential inspection


def run_and_verify(q_torch, k_torch, v_torch, sm_scale_test, is_causal):
        
    triton_output_non_causal, _ = flash_attention_fwd(
        q_torch.clone(), k_torch.clone(), v_torch.clone(), sm_scale_test, is_causal=is_causal
    )
    pytorch_output_non_causal, _ = pytorch_attention(
        q_torch.clone(), k_torch.clone(), v_torch.clone(), sm_scale_test, is_causal=is_causal
    )

    atol = 1e-5
    rtol = 1e-3 # Online softmax can accumulate small errors
    if DTYPE_TORCH_INPUT == torch.float16: atol=1e-3; rtol=1e-2

    print(f"Triton out (non-causal, first sample, first head, first 2 queries, first 5 features):\n{triton_output_non_causal[0,0,:2,:5]}")
    print(f"PyTorch out (non-causal, first sample, first head, first 2 queries, first 5 features):\n{pytorch_output_non_causal[0,0,:2,:5]}")
    
    all_close_non_causal = torch.allclose(triton_output_non_causal, pytorch_output_non_causal, atol=atol, rtol=rtol)
    
    is_match = False
    if all_close_non_causal:
        print("✅ Non-Causal: Triton and PyTorch match.")
        is_match = True
    else:
        print("❌ Non-Causal: Triton and PyTorch differ.")
        print(f"   Max diff: {torch.max(torch.abs(triton_output_non_causal - pytorch_output_non_causal))}")

    return triton_output_non_causal, is_match


def save_matrices_to_txt(*matrices, 
                        output_dir: str = ".", 
                        precision: int = 6,
                        dtype: str = "float",
                        prefix: str = "matrix",
                        delimiter: str = " ",
                        create_manifest: bool = False):
    """
    Save multiple torch tensors or numpy arrays to txt files with intelligent flattening.
    
    Args:
        *matrices: Variable number of torch tensors or numpy arrays
        output_dir: Directory to save the txt files (default: current directory)
        precision: Number of decimal places for floating point numbers (default: 6)
        dtype: Data type for formatting ("float", "int", "scientific", "auto") (default: "float")
        prefix: Prefix for output filenames (default: "matrix")
        delimiter: Delimiter between values in the same row (default: " ")
        create_manifest: Whether to create a manifest file with metadata (default: True)
    
    Returns:
        List of saved file paths
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    manifest_data = []
    
    for idx, matrix in enumerate(matrices, 1):
        # Convert to numpy if it's a torch tensor
        if isinstance(matrix, torch.Tensor):
            data = matrix.detach().cpu().numpy()
            original_type = "torch.Tensor"
            original_device = str(matrix.device)
            original_dtype = str(matrix.dtype)
        elif isinstance(matrix, np.ndarray):
            data = matrix
            original_type = "numpy.ndarray"
            original_device = "cpu"
            original_dtype = str(matrix.dtype)
        else:
            raise TypeError(f"Parameter {idx} must be a torch.Tensor or numpy.ndarray, "
                          f"got {type(matrix)}")
        
        # Get original shape for filename and metadata
        original_shape = data.shape
        
        # Handle shape string generation properly for 1D arrays
        if len(original_shape) == 1:
            # For 1D arrays like (768,), make it (1, 768) in the filename
            shape_str = f"1x{original_shape[0]}"
            display_shape = f"(1, {original_shape[0]})"
        else:
            shape_str = "x".join(map(str, original_shape))
            display_shape = str(original_shape)

        # Flatten all dimensions except the last one
        if len(data.shape) > 1:
            # Reshape to (product_of_all_except_last, last_dimension)
            new_shape = (-1, data.shape[-1])
            flattened_data = data.reshape(new_shape)
        else:
            # If 1D, reshape to (1, length) for consistency - 1 row, M columns
            flattened_data = data.reshape(1, -1)
        
        rows, cols = flattened_data.shape
        
        # Generate filename
        filename = f"{prefix}_{shape_str}_{idx}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Auto-detect data type if specified
        if dtype.lower() == "auto":
            if np.issubdtype(data.dtype, np.integer):
                format_str = "%d"
                detected_dtype = "int"
            elif np.issubdtype(data.dtype, np.floating):
                # Check if values are very large or very small
                abs_data = np.abs(data[np.isfinite(data)])
                if len(abs_data) > 0:
                    max_val = np.max(abs_data)
                    min_val = np.min(abs_data[abs_data > 0]) if np.any(abs_data > 0) else 0
                    if max_val > 10**6 or (min_val > 0 and min_val < 10**(-4)):
                        format_str = f"%.{precision}e"
                        detected_dtype = "scientific"
                    else:
                        format_str = f"%.{precision}f"
                        detected_dtype = "float"
                else:
                    format_str = f"%.{precision}f"
                    detected_dtype = "float"
            else:
                format_str = f"%.{precision}f"
                detected_dtype = "float"
        else:
            # Use specified dtype
            if dtype.lower() == "int":
                format_str = "%d"
                detected_dtype = "int"
            elif dtype.lower() == "scientific":
                format_str = f"%.{precision}e"
                detected_dtype = "scientific"
            else:  # default to float
                format_str = f"%.{precision}f"
                detected_dtype = "float"
        
        # Save to file
        with open(filepath, 'w') as f:
            # Write dimensions as first line
            f.write(f"{rows}{delimiter}{cols}\n")
            
            # Write data
            for row in flattened_data:
                if len(row) == 1:
                    # Single column
                    f.write(format_str % row[0] + "\n")
                else:
                    # Multiple columns
                    row_str = delimiter.join([format_str % val for val in row])
                    f.write(row_str + "\n")
        
        saved_files.append(filepath)
        
        # Collect metadata for manifest
        manifest_entry = {
            'index': idx,
            'filename': filename,
            'original_shape': original_shape,
            'flattened_shape': (rows, cols),
            'original_type': original_type,
            'original_dtype': original_dtype,
            'original_device': original_device,
            'saved_dtype': detected_dtype,
            'precision': precision,
            'delimiter': delimiter
        }
        manifest_data.append(manifest_entry)
        
        print(f"Saved matrix {idx}: {original_shape} -> {display_shape} "
              f"(flattened to {rows}x{cols}) as {detected_dtype} to: {filepath}")
    
    # Create manifest file
    if create_manifest and saved_files:
        manifest_path = os.path.join(output_dir, f"{prefix}_manifest.txt")
        with open(manifest_path, 'w') as f:
            f.write("# Matrix Save Manifest\n")
            f.write(f"# Total matrices: {len(saved_files)}\n")
            f.write("# Format: index|filename|original_shape|flattened_shape|type|dtype|device|saved_dtype|precision|delimiter\n")
            for entry in manifest_data:
                f.write(f"{entry['index']}|{entry['filename']}|{entry['original_shape']}|"
                       f"{entry['flattened_shape']}|{entry['original_type']}|{entry['original_dtype']}|"
                       f"{entry['original_device']}|{entry['saved_dtype']}|{entry['precision']}|"
                       f"'{entry['delimiter']}'\n")
        print(f"Created manifest file: {manifest_path}")



if __name__ == "__main__":

    os.environ["KERNEL_AUX_FILE_DIR"] = "./tmp"
    # os.environ["MLIR_ENABLE_DUMP"] = "1"
    # os.environ["MLIR_DUMP_PATH"] = "./tmp/dump_8_32.mlir"
    os.environ["TRITON_CPU_BACKEND"] = "1"

    torch.manual_seed(0)
    # Test parameters
    Z_test = 1
    H_test = 2
    N_CTX_test = 128 # Sequence length
    D_HEAD_test = 64
    sm_scale_test = 1.0 / math.sqrt(D_HEAD_test)

    q_torch = torch.randn((Z_test, H_test, N_CTX_test, D_HEAD_test), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    k_torch = torch.randn((Z_test, H_test, N_CTX_test, D_HEAD_test), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    v_torch = torch.randn((Z_test, H_test, N_CTX_test, D_HEAD_test), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    
    benchmark_triton(q_torch, k_torch, v_torch, sm_scale_test, D_HEAD_test)
    # _, _ = flash_attention_fwd(
    #     q_torch.clone(), k_torch.clone(), v_torch.clone(), sm_scale_test, 
    #     BLOCK_M=8, BLOCK_N=32 # BLOCK_M=8, BLOCK_N=16
    # )

    # print(f"Testing FlashAttention with Z={Z_test}, H={H_test}, N_CTX_Q={N_CTX_test}, D_HEAD={D_HEAD_test}")

    # flag = 0
    # Test non-causal
    # print("\n--- Non-Causal Attention Test ---")
    # try:
    #     ref_out, is_match = run_and_verify(q_torch=q_torch, k_torch=k_torch, v_torch=v_torch, 
    #                                        sm_scale_test=sm_scale_test, is_causal=False)
    #     if is_match:
    #         save_matrices_to_txt(q_torch, k_torch, v_torch, ref_out, precision=9, output_dir="benchmark/auto-tuner/flash_attention_fwd/run/test_data")
    #         # flag += 1
    # except Exception as e:
    #     print(f"ERROR during non-causal test: {e}")
    #     import traceback
    #     traceback.print_exc()

    # benchmark_triton(q_torch, k_torch, v_torch, sm_scale_test, DMODEL=D_HEAD_test)

    # # Test causal
    # print("\n--- Causal Attention Test ---")
    # try:
    #     ref_out, is_match = run_and_verify(q_torch=q_torch, k_torch=k_torch, v_torch=v_torch, 
    #                                        sm_scale_test=sm_scale_test, is_causal=True)
    #     if is_match:
    #         save_matrices_to_txt(q_torch, k_torch, v_torch, ref_out, prefix="causal_matrix")
    #         flag += 1
    # except Exception as e:
    #     print(f"ERROR during causal test: {e}")
    #     import traceback
    #     traceback.print_exc()

    # if flag == 2:
    #     benchmark_triton(q_torch, k_torch, v_torch, sm_scale_test)

