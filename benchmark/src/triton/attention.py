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


def atten_transpose_kernel(
    K_ptr, K_out_ptr,
    Z, H, N_CTX, D_HEAD,
    stride_kz, stride_kh, stride_km, stride_kk,
    stride_out_z, stride_out_h, stride_out_d, stride_out_n,
    BLOCK_SIZE_M: tl.constexpr, # Tile size for Q sequence length dimension
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, # Tile size for K sequence length dimension
    C_D_HEAD: tl.constexpr, # Head dimension (must be == D_HEAD)
    COL_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    # Calculate which batch, head, and N_CTX block we're processing
    num_n_blocks = tl.cdiv(N_CTX, BLOCK_SIZE_M)
    
    # Calculate batch and head indices
    batch_head_idx = pid // num_n_blocks
    n_block_idx = pid % num_n_blocks
    
    batch_idx = batch_head_idx // H
    head_idx = batch_head_idx % H
    
    k_block_ptr = tl.make_block_ptr(
        base=K_ptr,
        shape=(Z, H, N_CTX, D_HEAD),
        strides=(stride_kz, stride_kh, stride_km, stride_kk),
        offsets=(batch_idx, head_idx, n_block_idx * BLOCK_SIZE_M, 0),  # Fixed offsets
        block_shape=(1, 1, BLOCK_SIZE_M, C_D_HEAD),
        order=(3, 2, 1, 0)
    )

    k_block_out_ptr = tl.make_block_ptr(
        base=K_out_ptr,
        shape=(Z, H, D_HEAD, N_CTX),
        strides=(stride_out_z, stride_out_h, stride_out_d, stride_out_n),
        offsets=(batch_idx, head_idx, 0, n_block_idx * BLOCK_SIZE_M),  # Fixed offsets
        block_shape=(1, 1, C_D_HEAD, BLOCK_SIZE_M),
        order=(3, 2, 1, 0)
    )

    # Load and transpose
    k_block = tl.load(k_block_ptr)
    k_block_T = tl.trans(k_block, (0, 1, 3, 2))
    tl.store(k_block_out_ptr, k_block_T)


def atten_matmul_0_kernel(
    a_ptr, # [M, K]
    b_ptr, # [K, N]
    c_ptr, # [M, N]
    M, K, N, # grid = M / BLK_M * N / BLK_N
    Z, H, 
    stride_am, stride_ak,  # K, 1
    stride_bk, stride_bn,  # N, 1
    stride_cm, stride_cn,  # N, 1
    # IsScale,
    # sm_scale,
    BLOCK_SIZE_M: tl.constexpr, # Tile size for Q sequence length dimension
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, # Tile size for K sequence length dimension
    C_D_HEAD: tl.constexpr, # Head dimension (must be == D_HEAD)
    COL_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    zh_offset = tl.program_id(axis=1)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    block_offset_m = pid_m * BLOCK_SIZE_M
    block_offset_n = pid_n * BLOCK_SIZE_N
    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(Z * H * M, K), strides=(stride_am, stride_ak),
                                offsets=(zh_offset * M + block_offset_m, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                order=(1, 0))
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(Z * H * K, N), strides=(stride_bk, stride_bn),
                                offsets=(zh_offset * K, block_offset_n), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                order=(1, 0))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        a = tl.load(a_tile_ptr)
        b = tl.load(b_tile_ptr)

        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

        a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_SIZE_K])
        b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_SIZE_K, 0])


    # Convert the accumulator to the output matrix C's type if needed.
    # if IsScale:
    #     c = accumulator * sm_scale
    # else:
    #     c = accumulator
    c = accumulator

    c_tile_ptr = tl.make_block_ptr(base=c_ptr, shape=(Z * H * M, N), strides=(stride_cm, stride_cn),
                                offsets=(zh_offset * M + block_offset_m, block_offset_n),
                                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_tile_ptr, c)


def atten_matmul_1_kernel(
    a_ptr, # [M, K]
    b_ptr, # [K, N]
    c_ptr, # [M, N]
    M, N, K, # grid = M / BLK_M * K / BLK_K
    Z, H, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # IsScale,
    # sm_scale,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    C_D_HEAD: tl.constexpr, # Head dimension (must be == D_HEAD)
    COL_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    zh_offset = tl.program_id(axis=1)

    num_pid_n = tl.cdiv(K, BLOCK_SIZE_K)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    block_offset_m = pid_m * BLOCK_SIZE_M
    block_offset_n = pid_n * BLOCK_SIZE_K
    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(Z * H * M, N), strides=(stride_am, stride_ak),
                                offsets=(zh_offset * M + block_offset_m, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                                order=(1, 0))
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(Z * H * N, K), strides=(stride_bk, stride_bn),
                                offsets=(zh_offset * N, block_offset_n), block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
                                order=(1, 0))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_N)):

        a = tl.load(a_tile_ptr)
        b = tl.load(b_tile_ptr)

        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

        a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_SIZE_N])
        b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_SIZE_N, 0])


    # Convert the accumulator to the output matrix C's type if needed.
    # if IsScale:
    #     c = accumulator * sm_scale
    # else:
    #     c = accumulator
    c = accumulator

    c_tile_ptr = tl.make_block_ptr(base=c_ptr, shape=(Z * H * M, K), strides=(stride_cm, stride_cn),
                                offsets=(zh_offset * M + block_offset_m, block_offset_n),
                                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
    tl.store(c_tile_ptr, c)


def softmax_kernel(
    input_ptr, output_ptr, # Z, H, N_CTX, N_CTX
    Z, H, N_CTX, 
    input_stride_m, input_stride_k,
    output_stride_m, output_stride_k,
    BLOCK_SIZE_M: tl.constexpr, # Tile size for Q sequence length dimension
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, # Tile size for K sequence length dimension
    C_D_HEAD: tl.constexpr, # Head dimension (must be == D_HEAD)
    COL_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    off_zh_idx = tl.program_id(axis=1)

    row_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(Z * H * N_CTX, N_CTX),
        strides=(input_stride_m, input_stride_k),
        offsets=(off_zh_idx * N_CTX + row_idx, 0),  # Fixed offsets
        block_shape=(1, COL_SIZE),
        order=(1, 0)
    )

    row = tl.load(row_ptr)
    row = row * tl.rsqrt(C_D_HEAD * 1.0)

    row_minus_max = row - tl.max(row, axis=1)
    # tl.device_print("row_minus_max", row_minus_max)

    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)
    softmax_output = numerator / denominator

    output_row_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(Z * H * N_CTX, N_CTX),
        strides=(output_stride_m, output_stride_k),
        offsets=(off_zh_idx * N_CTX + row_idx, 0),  # Fixed offsets
        block_shape=(1, COL_SIZE),
        order=(1, 0)
    )

    tl.store(output_row_ptr, softmax_output)


def get_attention_kernel_autotune_config(N_CTX, D_HEAD, num_threads=1):
    configs = []
    
    BLK_M = [4, 8, 16, 32, 64]
    BLK_K = [8, 16, 32, 64]
    BLK_N = [8, 16, 32, 64]

    # BLK_M = [4, 8]
    # BLK_K = [8]
    # BLK_N = [16]

    for block_m in BLK_M:
        for block_k in BLK_K:
            for block_n in BLK_N:
                configs.append(
                    triton.Config({
                                'BLOCK_SIZE_M': block_m,
                                'BLOCK_SIZE_K': block_k,
                                'BLOCK_SIZE_N': block_n,
                                'C_D_HEAD': D_HEAD,
                                'COL_SIZE': N_CTX,
                            }, num_threads=num_threads)
                    )
    
    return configs

def benchmark_triton(
    Q, K, V,
):
    Z, H, N_CTX, D_HEAD = Q.shape

    transpose_jit = triton.jit(atten_transpose_kernel)
    transpose_tuned = triton.runtime.Autotuner(
        transpose_jit,
        transpose_jit.arg_names,
        reset_to_zero=None, 
        restore_value=None,
        configs=get_attention_kernel_autotune_config(N_CTX, D_HEAD),
        key=[],
    )

    K_t = torch.empty((Z, H, D_HEAD, N_CTX), dtype=K.dtype, device=K.device)
    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_SIZE_M']) * Z * H, )
    transpose_tuned[grid](
        K, K_t,
        Z, H, N_CTX, D_HEAD,
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        K_t.stride(0), K_t.stride(1), K_t.stride(2), K_t.stride(3),
    )

    matmul_jit = triton.jit(atten_matmul_0_kernel)
    matmul_tuned = triton.runtime.Autotuner(
        matmul_jit,
        matmul_jit.arg_names,
        reset_to_zero=None, 
        restore_value=None,
        configs=get_attention_kernel_autotune_config(N_CTX, D_HEAD),
        key=[],
    )

    S = torch.zeros((Z, H, N_CTX, N_CTX), dtype=DTYPE_TORCH_INPUT, device=Q.device)
    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_SIZE_M']) * triton.cdiv(N_CTX, META['BLOCK_SIZE_N']), Z * H)
    matmul_tuned[grid](
        Q, K_t, S,
        N_CTX, D_HEAD, N_CTX,
        Z, H, 
        Q.stride(2), Q.stride(3),
        K_t.stride(2), K_t.stride(3),
        S.stride(2), S.stride(3),
    )

    grid = (N_CTX, Z * H)
    softmax_jit = triton.jit(softmax_kernel)
    softmax_tuned = triton.runtime.Autotuner(
        softmax_jit,
        softmax_jit.arg_names,
        reset_to_zero=None, 
        restore_value=None,
        configs=get_attention_kernel_autotune_config(N_CTX, D_HEAD),
        key=[],
    )

    P = torch.empty_like(S, dtype=S.dtype, device=S.device) # [N_CTX, N_CTX]
    softmax_tuned[grid](
        S, P,
        Z, H, N_CTX,
        S.stride(2), S.stride(3),
        P.stride(2), P.stride(3),
    )

    Output = torch.empty_like(Q, dtype=Q.dtype, device=Q.device)
    matmul_jit = triton.jit(atten_matmul_1_kernel)
    matmul_tuned = triton.runtime.Autotuner(
        matmul_jit,
        matmul_jit.arg_names,
        reset_to_zero=None, 
        restore_value=None,
        configs=get_attention_kernel_autotune_config(N_CTX, D_HEAD),
        key=[],
    )
    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_SIZE_M']) * triton.cdiv(D_HEAD, META['BLOCK_SIZE_K']), Z * H)
    matmul_tuned[grid](
        P, V, Output,
        N_CTX, N_CTX, D_HEAD,
        Z, H,
        P.stride(2), P.stride(3),
        V.stride(2), V.stride(3),
        Output.stride(2), Output.stride(3),
    )

def attention(
    Q, K, V,
    BLOCK_SIZE_M: int, # Tile size for Q sequence length dimension
    BLOCK_SIZE_N: int, # Tile size for K sequence length dimension
    BLOCK_SIZE_K: int,
):
    assert Q.dim() == K.dim() == V.dim() == 4
    assert Q.is_contiguous(), "Matrix Q must be contiguous"
    assert K.is_contiguous(), "Matrix K must be contiguous"
    assert V.is_contiguous(), "Matrix V must be contiguous"

    Z, H, N_CTX, D_HEAD = Q.shape
    
    K_t = torch.empty((Z, H, D_HEAD, N_CTX), dtype=K.dtype, device=K.device)
    grid = (triton.cdiv(N_CTX, BLOCK_SIZE_M) * Z * H, )
    transpose_jit = triton.jit(atten_transpose_kernel)
    transpose_jit[grid](
        K, K_t,
        Z, H, N_CTX, D_HEAD,
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        K_t.stride(0), K_t.stride(1), K_t.stride(2), K_t.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        C_D_HEAD=D_HEAD,
        COL_SIZE=N_CTX,
    )

    S = torch.zeros((Z, H, N_CTX, N_CTX), dtype=DTYPE_TORCH_INPUT, device=Q.device)
    grid = (triton.cdiv(N_CTX, BLOCK_SIZE_M) * triton.cdiv(N_CTX, BLOCK_SIZE_N), Z * H)
    matmul_jit = triton.jit(atten_matmul_0_kernel)
    matmul_jit[grid]( # [N_CTX, D_HEAD], [D_HEAD, N_CTX]
        Q, K_t, S,
        N_CTX, D_HEAD, N_CTX,
        Z, H, 
        Q.stride(2), Q.stride(3),
        K_t.stride(2), K_t.stride(3),
        S.stride(2), S.stride(3),
        # IsScale=True,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        C_D_HEAD=D_HEAD,
        COL_SIZE=N_CTX,
    )

    P = torch.empty_like(S, dtype=S.dtype, device=S.device) # [N_CTX, N_CTX]
    grid = (N_CTX, Z * H)
    softmax_jit = triton.jit(softmax_kernel)
    softmax_jit[grid](
        S, P,
        Z, H, N_CTX,
        S.stride(2), S.stride(3),
        P.stride(2), P.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_N,
        BLOCK_SIZE_N=BLOCK_SIZE_K,
        C_D_HEAD=D_HEAD,
        COL_SIZE=N_CTX,
    )

    Output = torch.empty_like(Q, dtype=Q.dtype, device=Q.device)
    matmul_jit = triton.jit(atten_matmul_1_kernel)
    grid = (triton.cdiv(N_CTX, BLOCK_SIZE_M) * triton.cdiv(D_HEAD, BLOCK_SIZE_K), Z * H)
    matmul_jit[grid]( #[N_CTX, N_CTX], [N_CTX, D_HEAD]
        P, V, Output,
        N_CTX, N_CTX, D_HEAD,
        Z, H,
        P.stride(2), P.stride(3),
        V.stride(2), V.stride(3),
        Output.stride(2), Output.stride(3),
        # IsScale=False,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        C_D_HEAD=D_HEAD,
        COL_SIZE=N_CTX
    )

    return Output

def save_matrices_to_txt(*matrices, 
                        output_dir: str = ".", 
                        precision: int = 9,
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

def torch_attention_reference(Q, K, V):
    """
    PyTorch reference implementation of scaled dot-product attention
    
    Args:
        Q: Query tensor [Z, H, N_CTX, D_HEAD]
        K: Key tensor [Z, H, N_CTX, D_HEAD] 
        V: Value tensor [Z, H, N_CTX, D_HEAD]
        sm_scale: Scale factor (usually 1/sqrt(D_HEAD))
    
    Returns:
        Output tensor [Z, H, N_CTX, D_HEAD]
    """

    _, _, _, D_HEAD = Q.shape
    # Scale factor (standard practice)
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    # Step 1: Compute attention scores Q @ K^T
    # Q: [Z, H, N_CTX, D_HEAD]
    # K^T: [Z, H, D_HEAD, N_CTX]
    # Result: [Z, H, N_CTX, N_CTX]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Step 2: Scale scores
    scores = scores * sm_scale
    
    # Step 3: Apply softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Step 4: Apply attention weights to values
    # attention_weights: [Z, H, N_CTX, N_CTX]
    # V: [Z, H, N_CTX, D_HEAD]
    # Result: [Z, H, N_CTX, D_HEAD]
    output = torch.matmul(attention_weights, V)
    
    return output

def test_attention_correctness():
    """Test attention implementation against PyTorch reference"""
    
    torch.manual_seed(42)
    
    test_cases = [
        # (Z, H, N_CTX, D_HEAD, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
        (4, 32, 1024, 64, 4, 8, 16),
        # (1, 1, 32, 32, 16, 16, 16),      # Small case 
        # (1, 2, 64, 32, 16, 16, 16),      # Multi-head
        # (2, 4, 32, 64, 8, 8, 16),        # Multi-batch, multi-head
        # (1, 1, 128, 64, 32, 32, 32),     # Larger case
    ]
    
    for Z, H, N_CTX, D_HEAD, BLOCK_M, BLOCK_N, BLOCK_K in test_cases:
        print(f"Testing attention with shape Q,K,V: ({Z}, {H}, {N_CTX}, {D_HEAD})")
        print(f"Block sizes: M={BLOCK_M}, N={BLOCK_N}, K={BLOCK_K}")
        
        # Create random input tensors
        Q = torch.randn((Z, H, N_CTX, D_HEAD), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
        K = torch.randn((Z, H, N_CTX, D_HEAD), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
        V = torch.randn((Z, H, N_CTX, D_HEAD), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
        
        # Ensure contiguity
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
                
        import time
        # Compute reference output using PyTorch
        start_time = time.time()
        torch_output = torch_attention_reference(Q, K, V)
        torch_time = (time.time() - start_time)
        print(f"PyTorch attention computation time: {torch_time:.6f} seconds")
        
        # Time Triton implementation
        start_time = time.time()
        # Compute Triton output
        triton_output = attention(Q, K, V, BLOCK_M, BLOCK_N, BLOCK_K)
        triton_time = (time.time() - start_time)
        print(f"Triton attention computation time: {triton_time:.6f} seconds")


        # Check if outputs match
        rtol = 1e-3
        atol = 1e-3
        if torch.allclose(triton_output, torch_output, atol=atol, rtol=rtol):
            print(f"✅ Triton and PyTorch attention match")
        else:
            max_diff = torch.max(torch.abs(triton_output - torch_output))
            mean_diff = torch.mean(torch.abs(triton_output - torch_output))
            print(f"❌ Triton and PyTorch attention differ")
            print(f"   Maximum difference: {max_diff}")
            print(f"   Mean difference: {mean_diff}")
            
            # Debug specific batch/head combinations
            for z in range(min(Z, 2)):
                for h in range(min(H, 2)):
                    diff = torch.max(torch.abs(triton_output[z, h] - torch_output[z, h]))
                    print(f"   Batch {z}, Head {h} max diff: {diff}")
        
        print("-" * 70)


if __name__ == "__main__":
    
    os.environ["TRITON_CPU_BACKEND"] = "1"
    os.environ["MLIR_ENABLE_DUMP"] = "atten_matmul_0_kernel"
    os.environ["MLIR_DUMP_PATH"] = "/home/yuhao/T_RVV/tmp/attention_test/matmul_0.mlir"
    os.environ["TRITON_ENABLE_LLVM_DEBUG"] = "1"
    os.environ["TRITON_LLVM_DEBUG_ONLY"] = "triton-cpu-dot-conversion"
    # os.environ["KERNEL_LAUNCHER_INCLUDE_DIR"] = "/home/yuhao/T_RVV/tmp/attention_include"
    # os.environ["KERNEL_AUX_FILE_DIR"] = "/home/yuhao/T_RVV/tmp/attention_src"

    # test_attention_correctness()

    Z, H, N_CTX, D_HEAD = 4, 32, 1024, 64

    Q = torch.randn((Z, H, N_CTX, D_HEAD), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    K = torch.randn((Z, H, N_CTX, D_HEAD), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    V = torch.randn((Z, H, N_CTX, D_HEAD), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    
    # Ensure contiguity
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # attention(Q, K, V, BLOCK_SIZE_M=8, BLOCK_SIZE_N=8, BLOCK_SIZE_K=16)

    # ref_out = torch_attention_reference(Q, K, V)
    # save_matrices_to_txt(Q, K, V, ref_out,
    #                      output_dir="/home/yuhao/T_RVV/benchmark/auto-tuner/attention/run/test_data")

    # benchmark_triton(Q, K, V)