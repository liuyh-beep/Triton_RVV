import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import os
import numpy as np

# --- Configuration ---
DEVICE = 'cpu'
DTYPE = torch.float32
DTYPE_ACC = tl.float32

# Set Triton to run on the CPU backend
triton.runtime.driver.set_active_to_cpu()

def fused_moe_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # These will be updated by pre_hook:
    sorted_token_ids_ptr,
    expert_ids_ptr,
    # Matrix dimensions
    N, K, num_tokens_post_padded,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    # compute_type: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can be any shape representing batches and K is the feature dimension of each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is the number of experts, K is the input feature dimension, and N is the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the total number of tokens post padding, topk is the number of times each token is repeated,
        and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens, repeated topk times and arranged by the expert index they are assigned to.
    - expert_ids: A tensor containing the indices of the expert for each block. It determines which expert matrix from B should be used for each block in A.
    This kernel performs the multiplication of a token by its corresponding expert matrix as determined by `expert_ids`. The sorting of `sorted_token_ids`
    by expert index and padding ensures divisibility by BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) # [BLOCK_SIZE_M, ]
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = (offs_token < num_valid_tokens) & (offs_token >= 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    '''
    b: [E, N, K]
    [
     [0 + 0 * K], [0 + 1 * K], ..., [0 + (BLOCK_SIZE_N -1) * K],
     [1 +  0 * K], [1 + 1 * K], ..., [1 + (BLOCK_SIZE_N -1) * K],
     ...
     [BLOCK_SIZE_K-1 + 0 * K], [BLOCK_SIZE_K-1 + 1 * K], ..., [BLOCK_SIZE_K-1 + (BLOCK_SIZE_N -1) * K]
    ]
    '''

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int):

    num_tokens, top_k = topk_ids.shape
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    expert_counts.index_add_(0, topk_ids.flatten(), torch.ones_like(topk_ids.flatten(), dtype=torch.int32))
    padded_expert_counts = torch.ceil(expert_counts.float() / block_size).long() * block_size
    total_padded_tokens = torch.sum(padded_expert_counts).item()
    
    sorted_token_ids = torch.full((total_padded_tokens,), -1, dtype=torch.int32, device=topk_ids.device)
    expert_ids_for_blocks = []
    
    token_expert_pairs = []
    for i in range(num_tokens):
        for k in range(top_k):
            expert_id = topk_ids[i, k].item()
            flat_token_id = i * top_k + k
            token_expert_pairs.append((flat_token_id, expert_id))

    expert_tokens = [[] for _ in range(num_experts)]
    for flat_token_id, expert_id in token_expert_pairs:
        expert_tokens[expert_id].append(flat_token_id)

    current_pos = 0
    for expert_id, tokens in enumerate(expert_tokens):
        num_blocks = padded_expert_counts[expert_id].item() // block_size
        expert_ids_for_blocks.extend([expert_id] * num_blocks)
        if len(tokens) > 0:
            sorted_token_ids[current_pos : current_pos + len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        current_pos += padded_expert_counts[expert_id].item()
        
    # num_tokens_post_padded = torch.tensor([total_padded_tokens], dtype=torch.int32, device=topk_ids.device)

    return sorted_token_ids, torch.tensor(expert_ids_for_blocks, dtype=torch.int32, device=topk_ids.device), total_padded_tokens

def get_moe_fused_autotune_config(num_threads=1):
    configs = []
    block_sizes_m = [4] #[4]
    block_sizes_n = [4, 8]
    block_sizes_k = [8]
    
    for bm in block_sizes_m:
        for bn in block_sizes_n:
            for bk in block_sizes_k:
                configs.append(
                    triton.Config(
                        {
                            'BLOCK_SIZE_M': bm, 
                            'BLOCK_SIZE_N': bn, 
                            'BLOCK_SIZE_K': bk, 
                            'GROUP_SIZE_M': 8, 
                            'top_k': 2
                        }, num_threads=num_threads
                    )
                )
    return configs

def benchmark_triton(a, b, c,
                     sorted_token_ids, expert_ids_ptr,
                     num_tokens_post_padded, num_valid_tokens):

    M, K_a = a.shape
    E, N, K_b = b.shape
    assert K_a == K_b

    fn_jit = triton.jit(fused_moe_kernel)
    fn_jit_tuned = triton.runtime.Autotuner(
        fn=fn_jit,
        arg_names=fn_jit.arg_names,
        # pre_hook=moe_align_block_size,
        # post_hook=verify,
        configs=get_moe_fused_autotune_config(),
        key=[],
        reset_to_zero=None,
        restore_value=None
    )

    grid = lambda META: (
        triton.cdiv(num_tokens_post_padded, META['BLOCK_SIZE_M'])
          * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    fn_jit_tuned[grid](
        a, b, c, 
        sorted_token_ids, 
        expert_ids_ptr,
        N, K_a, num_tokens_post_padded,
        num_valid_tokens,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(2), b.stride(1),
        c.stride(1), c.stride(2)
    )

def pytorch_fused_moe(
    a, b, sorted_token_ids, expert_ids,
    num_tokens_post_padded, num_valid_tokens, top_k,
    block_size_m
):
    """
    PyTorch implementation matching the Triton fused MoE kernel.
    
    Args:
        a: Input tensor of shape (M, K) representing tokens
        b: Expert weight tensor of shape (E, N, K) where E is number of experts
        sorted_token_ids: Sorted token indices for each position
        expert_ids: Expert index for each block
        num_tokens_post_padded: Total number of tokens after padding
        num_valid_tokens: Number of valid tokens
        top_k: Number of experts per token
        block_size_m: Block size for M dimension
    
    Returns:
        c: Output tensor of shape (M, top_k, N)
    """
    device = a.device
    dtype = a.dtype
    
    M, K = a.shape
    E, N, K_b = b.shape
    assert K == K_b
    
    # Get actual number of tokens to process
    num_tokens_actual = num_tokens_post_padded.item() if torch.is_tensor(num_tokens_post_padded) else num_tokens_post_padded
    
    # Initialize output - this matches the Triton kernel's expectation
    # The output should be indexed by the sorted_token_ids
    max_token_id = sorted_token_ids.max().item() if len(sorted_token_ids) > 0 else 0
    c = torch.zeros((max_token_id + 1, N), device=device, dtype=dtype)
    
    # Process in blocks like the Triton kernel
    num_blocks = (num_tokens_actual + block_size_m - 1) // block_size_m
    
    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size_m
        end_idx = min(start_idx + block_size_m, num_tokens_actual)
        
        if start_idx >= num_tokens_actual:
            break
            
        # Get expert for this block
        if block_idx < len(expert_ids):
            expert_id = expert_ids[block_idx].item()
        else:
            continue
            
        if expert_id < 0 or expert_id >= E:
            continue
            
        # Process tokens in this block
        for i in range(start_idx, end_idx):
            if i >= len(sorted_token_ids):
                break
                
            token_id = sorted_token_ids[i].item()
            
            # Check token validity (matching Triton's token_mask)
            if token_id < 0 or token_id >= num_valid_tokens:
                continue
                
            # Get original token index (accounting for top_k repetition)
            original_token_idx = token_id // top_k
            
            if original_token_idx >= M:
                continue
                
            # Get token features
            token_features = a[original_token_idx]  # Shape: (K,)
            
            # Get expert weights (note: b has shape (E, N, K))
            expert_weights = b[expert_id]  # Shape: (N, K)
            
            # Compute matrix multiplication: token @ expert.T
            output = torch.matmul(token_features, expert_weights.T)  # Shape: (N,)
            
            # Store result
            c[token_id] = output
    
    # Reshape output to match expected format (M, top_k, N)
    final_output = torch.zeros((M, top_k, N), device=device, dtype=dtype)
    
    for i in range(len(sorted_token_ids)):
        if i >= num_tokens_actual:
            break
            
        token_id = sorted_token_ids[i].item()
        if token_id < 0 or token_id >= num_valid_tokens:
            continue
            
        original_token_idx = token_id // top_k
        top_k_idx = token_id % top_k
        
        if original_token_idx < M and top_k_idx < top_k and token_id < len(c):
            final_output[original_token_idx, top_k_idx] = c[token_id]
    
    return final_output

def run_and_verify_triton_kernel(a, b, triton_c, sorted_token_ids, expert_ids,
                                num_tokens_post_padded, num_valid_tokens, top_k, config):
    """
    Runs the Triton kernel with a fixed configuration and compares its result
    against the PyTorch reference output.
    """

    _, K_in = a.shape
    _, N_out, K_in_b = b.shape
    assert K_in == K_in_b

    grid = (
        triton.cdiv(num_tokens_post_padded, config['BLOCK_SIZE_M'])
          * triton.cdiv(N_out, config['BLOCK_SIZE_N']),
    )

    kernel = triton.jit(fused_moe_kernel)

    kernel[grid](
        a, b, triton_c,
        sorted_token_ids, 
        expert_ids, 
        N_out, K_in, num_tokens_post_padded, # EM = num_tokens_post_padded
        num_valid_tokens,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(2), b.stride(1),
        triton_c.stride(1), triton_c.stride(2),
        BLOCK_SIZE_M=config['BLOCK_SIZE_M'],
        BLOCK_SIZE_N=config['BLOCK_SIZE_N'],
        BLOCK_SIZE_K=config['BLOCK_SIZE_K'],
        GROUP_SIZE_M=config['GROUP_SIZE_M'],
        top_k=top_k,
    )

    ref_c = pytorch_fused_moe(
        a, b, sorted_token_ids, expert_ids,
        num_tokens_post_padded, num_valid_tokens, top_k,
        block_size_m=config['BLOCK_SIZE_M'],
    )

    
    is_match = False

    if triton_c.shape == ref_c.shape:
        print("Comparing Triton output with PyTorch reference output...")
        are_close = torch.allclose(ref_c, triton_c, atol=1e-5, rtol=1e-3)
        
        if are_close:
            # print("✅ SUCCESS: Triton kernel output matches PyTorch reference output.")
            is_match = True
        else:
            # print("❌ FAILURE: Triton kernel output does not match PyTorch reference output.")
            max_diff = torch.max(torch.abs(ref_c - triton_c)).item()
            print(f"   Max absolute difference: {max_diff}")

    return triton_c, is_match

def save_matrices_to_txt(*matrices, 
                        output_dir: str = ".", 
                        precision: int = 9,
                        dtype: str = "float",
                        prefix: str = "matrix",
                        delimiter: str = " ",
                        create_manifest: bool = False,
                        start_idx: int = 1):
    """
    Save multiple torch tensors or numpy arrays to txt files with intelligent flattening.
    
    Args:
        *matrices: Variable number of torch tensors or numpy arrays
        output_dir: Directory to save the txt files (default: current directory)
        precision: Number of decimal places for floating point numbers (default: 6)
        dtype: Data type for formatting ("float", "int", "scientific", "auto") (default: "float")
        prefix: Prefix for output filenames (default: "matrix")
        delimiter: Delimiter between values in the same row (default: " ")
        create_manifest: Whether to create a manifest file with metadata (default: False)
        start_idx: Starting index for filename numbering (default: 1)
    
    Returns:
        List of saved file paths
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    manifest_data = []
    
    for i, matrix in enumerate(matrices):
        idx = start_idx + i  # Use custom starting index
        
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
        
        # Generate filename with custom index
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
    
    return saved_files

if __name__ == "__main__":
    torch.manual_seed(0)
    
    M = 1000
    E = 8
    K = 128
    N = 512
    topK = 2
    
    block_sizes_m = [4, 8, 16, 32, 64]

    a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)  # 1000 tokens
    b = torch.randn((E, N, K), device=DEVICE, dtype=DTYPE)  # 8 experts, each with K inputs and N outputs

    gating_output = torch.randn(M, E, device=DEVICE)  # Gating scores for each token across experts
    _, topk_ids = torch.topk(gating_output, topK, dim=-1)
    
    # Save input matrices a and b once (they don't change across block sizes)
    print("Saving input matrices a and b...")
    save_matrices_to_txt(a, b, 
                        output_dir="/home/yuhao/T_RVV/benchmark/auto-tuner/fused_moe/run/test_data", 
                        precision=9,)
    
    # Track results for each block size
    successful_configs = []
    failed_configs = []
    
    print(f"\nTesting {len(block_sizes_m)} different block sizes: {block_sizes_m}")
    print("=" * 60)
    
    for block_size_m in block_sizes_m:
        print(f"\n--- Testing BLOCK_SIZE_M = {block_size_m} ---")
        
        # Create configuration for this block size
        config = {
            'BLOCK_SIZE_M': block_size_m, 
            'BLOCK_SIZE_N': 64, 
            'BLOCK_SIZE_K': 32, 
            'GROUP_SIZE_M': 8
        }
        
        try:
            # Call moe_align_block_size for this block size
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids, block_size_m, E
            )
            
            # print(f"moe_align_block_size completed:")
            # print(f"  - sorted_token_ids shape: {sorted_token_ids.shape}")
            # print(f"  - expert_ids shape: {expert_ids.shape}")
            # print(f"  - num_tokens_post_padded: {num_tokens_post_padded}")
            
            # Create output tensor for this configuration
            c = torch.zeros((M, topK, N), device=DEVICE, dtype=DTYPE)
            
            # Run and verify the Triton kernel
            triton_out, is_match = run_and_verify_triton_kernel(
                a, b, c, sorted_token_ids, expert_ids,
                num_tokens_post_padded, topk_ids.numel(), topK, config
            )
            
            if is_match:
                print(f"✅ SUCCESS for BLOCK_SIZE_M = {block_size_m}")
                successful_configs.append(block_size_m)
                
                # Save the successful configuration data
                output_dir = f"/home/yuhao/T_RVV/benchmark/auto-tuner/fused_moe/run/test_data"
                print(f"Saving results for BLOCK_SIZE_M = {block_size_m} to {output_dir}...")
                
                save_matrices_to_txt(
                    triton_out,           # Output from Triton kernel
                    start_idx=3,
                    output_dir=output_dir,
                    prefix=f"matrix_BLOCK_SIZE_M_{block_size_m}",
                )
                
                save_matrices_to_txt(
                    sorted_token_ids,     # Processed token IDs
                    expert_ids,           # Expert IDs for blocks
                    start_idx=4,
                    dtype="int",
                    output_dir=output_dir,
                    prefix=f"matrix_BLOCK_SIZE_M_{block_size_m}",
                )

            else:
                print(f"❌ FAILURE for BLOCK_SIZE_M = {block_size_m}")
                failed_configs.append(block_size_m)
                
        except Exception as e:
            print(f"❌ ERROR for BLOCK_SIZE_M = {block_size_m}: {str(e)}")
            failed_configs.append(block_size_m)
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if successful_configs:
        print(f"✅ SUCCESSFUL configurations (BLOCK_SIZE_M): {successful_configs}")
        print(f"   Data saved for {len(successful_configs)} configurations")
    else:
        print("❌ No configurations were successful")
    
    if failed_configs:
        print(f"❌ FAILED configurations (BLOCK_SIZE_M): {failed_configs}")
    
