'''
Currently masked load is not supported yet
'''

import torch
import triton
import triton.language as tl
import os

# KERNEL_LAUNCHER_INCLUDE_DIR=os.getenv("KERNEL_LAUNCHER_INCLUDE_DIR", "/home/yuhao/T_RVV/benchmark/src/launcher/include")
# KERNEL_AUX_FILE_DIR=os.getenv("KERNEL_AUX_FILE_DIR", "/home/yuhao/T_RVV/benchmark/src/launcher/src/matmul")

# KERNEL_LAUNCHER_INCLUDE_DIR="/home/yuhao/T_RVV/benchmark/src/launcher/include" KERNEL_AUX_FILE_DIR="/home/yuhao/T_RVV/benchmark/src/launcher/src/matmul"

DTYPE = getattr(torch, (os.getenv("DTYPE", "float32")))
# Choose block size depending on dtype. We have more register
# capacity for bfloat16/float16 compared to float32.
# BLOCK_SIZE_M = 8 if DTYPE == torch.float32 else 32
# BLOCK_SIZE_K = 8 if DTYPE == torch.float32 else 32
# BLOCK_SIZE_N = 8

def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,  
        USE_BLOCK_POINTERS: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    if USE_BLOCK_POINTERS:
        block_offset_m = pid_m * BLOCK_SIZE_M
        block_offset_n = pid_n * BLOCK_SIZE_N
        a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                       offsets=(block_offset_m, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                       order=(1, 0))
        b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                       offsets=(0, block_offset_n), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                       order=(1, 0))
    else:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_tile_ptr = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_tile_ptr = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to matrix C's type after the loop, if C has lower precision type (for example, float16 and bfloat16).
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.

        a = tl.load(a_tile_ptr)
        b = tl.load(b_tile_ptr)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        # Advance the ptrs to the next K block.
        if USE_BLOCK_POINTERS:
            a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_SIZE_K])
            b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_SIZE_K, 0])
        else:
            a_tile_ptr += BLOCK_SIZE_K * stride_ak
            b_tile_ptr += BLOCK_SIZE_K * stride_bk

    # Convert the accumulator to the output matrix C's type if needed.
    c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C.
    if USE_BLOCK_POINTERS:
        c_tile_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                       offsets=(block_offset_m, block_offset_n),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
        tl.store(c_tile_ptr, c)
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_tile_ptr = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_tile_ptr, c)


# Triton Benchmark
def get_matmul_kernel_autotune_config(num_threads=1):
    configs = []

    block_sizes_M = [4, 8, 16, 32, 64, 128]
    block_sizes_N = [8, 16, 32, 64, 128]
    block_sizes_K = [8, 16, 32, 64, 128]

    # block_sizes_M = [4, 8]
    # block_sizes_N = [8]
    # block_sizes_K = [8]


    for block_m in block_sizes_M:
        for block_n in block_sizes_N:
            for block_k in block_sizes_K:
                configs.append(
                    triton.Config({
                                'BLOCK_SIZE_M': block_m,
                                'BLOCK_SIZE_N': block_n,
                                'BLOCK_SIZE_K': block_k,
                                'USE_BLOCK_POINTERS': True
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


def matmul_triton(a: torch.Tensor, b: torch.Tensor, M: int, N: int, K: int, 
                  BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, BLOCK_SIZE_K: int):

    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape

    c = torch.zeros((M, N), device='cpu', dtype=DTYPE)

    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        USE_BLOCK_POINTERS=True,  #
    )
    return c

def run_and_verify_triton_kernel(a, b, config):
    """
    Runs the Triton kernel with a fixed configuration and compares its result
    against the PyTorch reference output.
    """

    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    triton_c = torch.zeros((M, N), device='cpu', dtype=DTYPE)

    grid = (triton.cdiv(M, config['BLOCK_SIZE_M']) * triton.cdiv(N, config['BLOCK_SIZE_N']), )

    kernel = triton.jit(matmul_kernel)
    kernel[grid](
        a, b, triton_c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        triton_c.stride(0), triton_c.stride(1),
        BLOCK_SIZE_M=config['BLOCK_SIZE_M'],
        BLOCK_SIZE_N=config['BLOCK_SIZE_N'],
        BLOCK_SIZE_K=config['BLOCK_SIZE_K'],
        USE_BLOCK_POINTERS=config['USE_BLOCK_POINTERS'],
    )

    ref_c = torch.matmul(a, b)

    is_match = False

    if triton_c.shape == ref_c.shape:
        print("Comparing Triton output with PyTorch reference output...")
        are_close = torch.allclose(ref_c, triton_c, atol=1e-5, rtol=1e-3)
        
        if are_close:
            print("✅ SUCCESS: Triton kernel output matches PyTorch reference output.")
            is_match = True
        else:
            print("❌ FAILURE: Triton kernel output does not match PyTorch reference output.")

            print(f"Triton out:\n{triton_c[:2,:5]}")
            print(f"PyTorch out:\n{ref_c[:2,:5]}")

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
    os.environ["TRITON_CPU_BACKEND"] = "1"
    os.environ["MLIR_ENABLE_DUMP"] = "1"
    os.environ["MLIR_DUMP_PATH"] = "./tmp/matmul_dump_32_32_32.mlir"

    # Unit Test
    M, N, K = 1024, 1024, 1024

    torch.manual_seed(0)
    triton.runtime.driver.set_active_to_cpu()

    a = torch.randn((M, N), device='cpu', dtype=DTYPE)
    b = torch.randn((N, K), device='cpu', dtype=DTYPE)

    config = {
            'BLOCK_SIZE_M': 32, 
            'BLOCK_SIZE_N': 32, 
            'BLOCK_SIZE_K': 32, 
            'USE_BLOCK_POINTERS': True
    }

    # ref_c, is_match = run_and_verify_triton_kernel(a, b, config)

    # output_dir = f"/home/yuhao/T_RVV/benchmark/auto-tuner/matmul/run/test_data"   
    # print(f"Saving input and ref_c data into {output_dir}")
    # save_matrices_to_txt(
    #         a, b,
    #         ref_c,           # Output from Triton kernel
    #         output_dir=output_dir,
    # )
    
    # benchmark_triton((M, N, K), a, b)

    # if is_match:
    #     output_dir = f"/home/yuhao/T_RVV/benchmark/auto-tuner/matmul/run/test_data"   
    #     print(f"Saving input and ref_c data into {output_dir}")
    #     save_matrices_to_txt(
    #         a, b,
    #         ref_c,           # Output from Triton kernel
    #         output_dir=output_dir,
    #     )

        # benchmark_triton((M, N, K), a, b)