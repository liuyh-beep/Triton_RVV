#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class BuildConfig:
    """Configuration for building a kernel."""

    # Path constants
    script_dir: Path = field(
        default_factory=lambda: Path(os.path.dirname(os.path.abspath(__file__)))
    )

    # Fields derived from script_dir - need to use default_factory
    build_dir: Path = field(init=False)
    src_dir: Path = field(init=False)

    # Fields with fixed values
    riscv_gnu_toolchain_dir: Path = field(default=Path("/llvm_rvv/rvv/riscv"))
    llvm_build_dir: Path = field(default=Path("/llvm_rvv/llvm/llvm-project/build"))
    pyc: str = field(default="python3")

    # New fields for auto-tuner structure
    auto_tuner_dir: Path = field(init=False)
    kernel_auto_tuner_dir: Path = field(init=False)
    kernel_bin_dir: Path = field(init=False)
    kernel_dump_dir: Path = field(init=False)
    kernel_perf_dir: Path = field(init=False)
    kernel_perf_data_dir: Path = field(init=False)
    kernel_config_file: Path = field(init=False)

    is_last_variant: bool = field(default=False)

    # Compiler tools - will be initialized in post_init
    clangpp_base: str = field(init=False)
    objdump: str = field(init=False)
    ar: str = field(init=False)
    as_tool: str = field(init=False)

    # Gem5 Simulator Path
    copy2Gem5: bool = field(default=False)
    gem5_des_path: Path = field(default=Path("/home/yuhao/gem5/tt_rvv/triton/bin"))

    # Default configuration values
    debug_mode: bool = field(default=False)
    debug_suffix: str = field(default="")
    vectorize: bool = field(default=False)
    vector_suffix: str = field(default="")
    kernel_type: str = field(default="")
    kernel_ir: str = field(default="")
    kernel_launcher: str = field(default="")
    c_kernel: str = field(default="")
    triton_kernel: str = field(default="")
    blk_values: str = field(default="")
    static_link: bool = field(default=True)
    static_suffix: str = field(default="_static")
    # mode: int = field(default=0)  # 0: none, 1: accuracy, 2: keep_test
    # mode_suffix: str = field(default="")
    opt_level: str = field(default="O2")
    opt_suffix: str = field(default="_O2")
    # Do not need riscv-v-vector-bits-min now, 
    # it would be the same value as whatever zvlb extension is passed to -march
    # For instance, -march=rv64gcv_zvl256b, the vector_bits_min is 256

    kernel_source: str = field(default="")
    config_file: str = field(default="")
    lib_name: str = field(default="")
    json_key: str = field(default="")

    # Directories - will be set based on kernel_type
    lib_dir: Path = field(default=Path())
    obj_dir: Path = field(default=Path())
    bin_dir: Path = field(default=Path())
    kernel_launcher_include_dir: Path = field(default=Path())
    kernel_launcher_src_dir: Path = field(default=Path())

    clean_first: bool = field(default=False)

    # Computed values
    clangpp: str = field(default="")
    kernel_enable: str = field(default="")

    transfer_to_remote: bool = field(default=False)
    remote_user: str = field(default="yuhao")
    remote_host: str = field(default="10.32.44.164")
    remote_dir: str = field(default="/home/yuhao/triton_riscv_test/triton/tuning_test")
    ssh_ctrl_socket: str = field(default="/tmp/ssh_ctrl_socket")

    def __post_init__(self):
        # Initialize derived paths
        self.build_dir = self.script_dir.parent / "benchmark" / "build"
        self.src_dir = self.script_dir.parent / "benchmark" / "src"

        # Initialize auto-tuner paths
        self.auto_tuner_dir = self.script_dir.parent / "benchmark" / "auto-tuner"

        # Initialize compiler tools
        self.clangpp_base = (
            f"{self.llvm_build_dir}/bin/clang++ --target=riscv64-unknown-linux-gnu "
            f"--sysroot={self.riscv_gnu_toolchain_dir}/sysroot "
            f"-fuse-ld=lld -fveclib=SLEEF -lm -L /home/kevin/sleef/build-riscv64/lib "
            f"--gcc-toolchain={self.riscv_gnu_toolchain_dir}"
        )
        self.objdump = (
            f"{self.riscv_gnu_toolchain_dir}/bin/riscv64-unknown-linux-gnu-objdump"
        )
        self.ar = f"{self.llvm_build_dir}/bin/llvm-ar"
        self.as_tool = f"{self.llvm_build_dir}/bin/llvm-as"

    def setup_auto_tuner_dirs(self, kernel_name: str) -> None:
        """Setup auto-tuner directory structure for a specific kernel."""
        self.kernel_auto_tuner_dir = self.auto_tuner_dir / kernel_name
        self.kernel_dump_dir = self.kernel_auto_tuner_dir / "dump"
        self.kernel_bin_dir = self.kernel_auto_tuner_dir / "run" / "bin"
        self.kernel_test_dir = self.kernel_auto_tuner_dir / "run" / "test_data"
        self.kernel_config_file = self.kernel_auto_tuner_dir / "run" / "configs.json"
        self.kernel_perf_dir = self.kernel_auto_tuner_dir / "run" / "perf"
        self.kernel_perf_data_dir = self.kernel_perf_dir / "perf_data"

        # Create all directories
        create_dir_if_not_exists(self.kernel_auto_tuner_dir)
        create_dir_if_not_exists(self.kernel_bin_dir)
        create_dir_if_not_exists(self.kernel_dump_dir)
        create_dir_if_not_exists(self.kernel_perf_dir) 
        create_dir_if_not_exists(self.kernel_test_dir)
        create_dir_if_not_exists(self.kernel_perf_data_dir)

        # Initialize or load configs.json
        self.init_kernel_config_file()

    def clean_auto_tuner_dirs(self, kernel_name: str) -> None:
        """Clean auto-tuner directories for a specific kernel."""
        # Setup the kernel-specific paths first
        kernel_auto_tuner_dir = self.auto_tuner_dir / kernel_name
        kernel_dump_dir = kernel_auto_tuner_dir / "dump"
        kernel_bin_dir = kernel_auto_tuner_dir / "run" / "bin"
        
        def remove_dir_contents(directory: Path) -> None:
            if directory.exists():
                print(f"Cleaning directory: {directory}")
                try:
                    for item in directory.glob("*"):
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            from shutil import rmtree
                            rmtree(item)
                    print(f"Successfully cleaned {directory}")
                except Exception as e:
                    print(f"Error while cleaning {directory}: {e}")
            else:
                print(f"Directory does not exist: {directory}")

        # Clean the auto-tuner specific directories
        remove_dir_contents(kernel_bin_dir)
        remove_dir_contents(kernel_dump_dir)

    def get_relative_path(self, full_path: Path) -> str:
        """Convert absolute path to relative path from auto-tuner directory."""
        try:
            return str(Path(full_path).relative_to(self.auto_tuner_dir))
        except ValueError as e:
            print(f"Warning: Could not create relative path for {full_path}: {e}")
            return str(full_path)

    def init_kernel_config_file(self) -> None:
        """Initialize or load the kernel-specific configs.json file."""
        initial_config = {
            "variants": [],
            "perf_files": {
                "stat": self.get_relative_path(self.kernel_perf_dir / "perf_stats.csv")
            }
        }
        with open(self.kernel_config_file, "w") as f:
            json.dump(initial_config, f, indent=4)


    def clean(self) -> None:
        """Clean build artifacts in bin_dir and obj_dir."""
        def remove_dir_contents(directory: Path) -> None:
            if directory.exists():
                print(f"Cleaning directory: {directory}")
                try:
                    for item in directory.glob("*"):
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            from shutil import rmtree
                            rmtree(item)
                    print(f"Successfully cleaned {directory}")
                except Exception as e:
                    print(f"Error while cleaning {directory}: {e}")
            else:
                print(f"Directory does not exist: {directory}")

        # Check if auto-tuner directories are initialized
        if hasattr(self, 'kernel_dump_dir') and self.kernel_dump_dir is not None:
            remove_dir_contents(self.kernel_dump_dir)
        if hasattr(self, 'kernel_bin_dir') and self.kernel_bin_dir is not None:
            remove_dir_contents(self.kernel_bin_dir)

        # Always clean these directories
        remove_dir_contents(self.bin_dir)
        remove_dir_contents(self.lib_dir)
        remove_dir_contents(self.obj_dir)



def run_command(cmd: str, check: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and return its exit code, stdout and stderr."""
    print(f"Running: {cmd}")
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    exit_code = process.returncode

    # if stdout:
    #     print(f"Output: {stdout}")

    if exit_code != 0 and check:
        print(f"Error: Command failed with code {exit_code}")
        print(f"stderr: {stderr}")
        sys.exit(exit_code)

    return exit_code, stdout, stderr


def create_dir_if_not_exists(directory: Union[str, Path]) -> None:
    """Create a directory if it doesn't exist."""
    directory = Path(directory)
    if not directory.is_dir():
        print(f"Creating directory: {directory}")
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: Failed to create directory {directory}: {e}")
            sys.exit(1)


def find_source_key_in_json(config_file: Path, source_path: str) -> Optional[str]:
    """Find the source file key in the JSON configuration file."""
    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file {config_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read config file {config_file}: {e}")
        sys.exit(1)

    print(f"Looking for source in JSON: {source_path}")

    # Try with full path
    if source_path in config_data:
        print(f"Found using full path: {source_path}")
        return source_path

    # Try with basename
    basename = os.path.basename(source_path)
    if basename in config_data:
        print(f"Found using basename: {basename}")
        return basename

    # Try with relative path from src_dir
    src_dir_str = str(BuildConfig().src_dir)
    if src_dir_str in source_path:
        relative_path = source_path.replace(src_dir_str + "/", "")
        if relative_path in config_data:
            print(f"Found using relative path: {relative_path}")
            return relative_path

    print(f"No configuration found for {source_path} in {config_file}")
    return None


def load_config_from_json(config: BuildConfig) -> None:
    """Load configuration from JSON file."""
    try:
        with open(config.config_file, "r") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file {config.config_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read config file {config.config_file}: {e}")
        sys.exit(1)

    print(f"Found configuration under key: {config.json_key}")

    # Get the specific configuration for this source file
    source_config = config_data[config.json_key]

    # Extract optional parameters
    if source_config.get("debug", False):
        config.debug_mode = True
        config.debug_suffix = "_g"
    
    if source_config.get("vectorize", False): 
        config.vectorize = True
        config.VLEN = source_config.get("VLEN", 256)
        # change march=rv64gcv_zvl256b
        config.vector_suffix = f"VL{config.VLEN}"

    if source_config.get("dynamic", False):
        config.static_link = False
        config.static_suffix = ""

    if "opt" in source_config:
        opt_level = source_config["opt"]
        if opt_level not in ["O2", "O3"]:
            print("Error: optimization level must be O2 or O3")
            sys.exit(1)
        config.opt_level = opt_level
        config.opt_suffix = f"_{opt_level}"

    if "mode" in source_config:
        mode_val = source_config["mode"]
        if mode_val == "accuracy":
            config.mode = 0
            config.mode_suffix = ""
        elif mode_val == "single_block":
            config.mode = 1
            config.mode_suffix = "_single_block"
        elif mode_val == "single_iteration":
            config.mode = 2
            config.mode_suffix = "_single_iteration"


    if "copy2Gem5" in source_config:
        config.copy2Gem5 = source_config["copy2Gem5"]

    if "remote" in source_config:
        remote_config = source_config["remote"]
        config.transfer_to_remote = remote_config.get("enable", False)
        if config.transfer_to_remote:
            config.remote_user = remote_config.get("user", config.remote_user)
            config.remote_host = remote_config.get("host", config.remote_host)
            config.remote_dir = remote_config.get("dir", config.remote_dir)
            config.ssh_ctrl_socket = remote_config.get("socket", config.ssh_ctrl_socket)

    # Print configuration summary
    print(f"Configuration loaded from {config.config_file} for {config.json_key}")
    # print(f"- Block values: {blk[0]} {blk[1]} {blk[2]}")
    print(f"- Debug mode: {'enabled' if config.debug_mode else 'disabled'}")
    print(
        f"- Vectorization: {f'enabled (VLEN: {config.VLEN} bits)' if config.vectorize else 'disabled'}"
    )
    print(f"- Linking: {'static' if config.static_link else 'dynamic'}")
    print(f"- Optimization: {config.opt_level}")
    print(
        f"- Mode: {'single_block' if config.mode == 1 else 'Normal' if config.mode == 0 else 'single_iteration'}"
    )


def setup_directories(config: BuildConfig) -> None:
    """Set up compilation directories."""
    if config.kernel_type == "triton":
        config.lib_dir = config.build_dir / "lib" / "triton"
        config.obj_dir = config.build_dir / "obj" / "triton"
        config.bin_dir = config.build_dir / "bin" / "triton"
    elif config.kernel_type == "c":
        config.lib_dir = config.build_dir / "lib" / "c"
        config.obj_dir = config.build_dir / "obj" / "c"
        config.bin_dir = config.build_dir / "bin" / "c"
    else:
        print(f"Error: Invalid kernel type {config.kernel_type}")
        sys.exit(1)

    config.kernel_launcher_include_dir = config.src_dir / "launcher" / "include"
    config.kernel_launcher_src_dir = config.src_dir / "launcher" / "src"

    create_dir_if_not_exists(config.kernel_launcher_include_dir)
    create_dir_if_not_exists(config.kernel_launcher_src_dir)
    create_dir_if_not_exists(config.obj_dir)
    create_dir_if_not_exists(config.bin_dir)
    create_dir_if_not_exists(config.lib_dir)


def configure_compiler_flags(config: BuildConfig) -> None:
    """Configure compiler flags."""
    # Debug flags
    debug_flag = "-g -fno-omit-frame-pointer " if config.debug_mode else ""

    #-march=rv64gcv_zvl256b_zicbop (zicbop is for prefetch)
    # Architecture flags
    march = f"rv64gcv_zvl{str(config.VLEN)}b_zicbop  -mllvm -force-tail-folding-style=data-with-evl -mllvm -prefer-predicate-over-epilogue=predicate-dont-vectorize" if config.vectorize else "rv64gc_zicbop"

    # Build complete CLANGPP command
    config.clangpp = (
        f"{config.clangpp_base} -march={march} -mabi=lp64d"
    )

    if config.static_link:
        config.clangpp += " -static"

    config.clangpp += f" -{config.opt_level} {debug_flag}"

    # # Mode flags for accuracy checking and test data
    if config.mode == 1:
        config.clangpp += " -DSINGLE_BLOCK"
    elif config.mode == 0:
        config.clangpp += " -DCHECK_ACCURACY"
    elif config.mode == 2:
        config.clangpp += " -DSINGLE_ITERATION"
    #print(f"Compiler command: {config.clangpp}")


def build_c_kernel(config: BuildConfig) -> None:
    """Build C kernel."""
    config.kernel_enable = "C_KERNEL_ENABLE"

    if config.clean_first:
        kernel_name = os.path.basename(config.c_kernel).replace(".cpp", "")
        config.clean_auto_tuner_dirs(kernel_name)

    # Build support library
    run_command(
        f"{config.clangpp} -fPIC -I {config.build_dir}/../../env_build/include "
        f"-c {config.src_dir}/support/*.cpp -o {config.obj_dir}/support.o"
    )

    run_command(
        f"{config.ar} rcs {config.lib_dir}/libsupport.a {config.obj_dir}/support.o"
    )

    kernel_name = os.path.basename(config.c_kernel).replace(".cpp", "")
    out_obj_dir = config.obj_dir / kernel_name
    create_dir_if_not_exists(out_obj_dir)

    # Generate assembly output for kernel
    # TODO: For the openmp(multi-thread), there is a bug for libarcher.so on laptop
    #       $ ../triton/v256_matmul_16_16_16_g_static_O3.elf
    #       libarcher.so: cannot open shared object file: No such file or directory

    run_command(
        f"{config.clangpp} -fPIC -I {config.build_dir}/../../env_build/include -S {config.c_kernel} " #-fopenmp
        f"-o {out_obj_dir}/{config.vector_suffix}_{kernel_name}{config.blk_values}_kernel_src.s"
    )
    print(
        f"The ASM code of kernel part is at {out_obj_dir}/{config.vector_suffix}_{kernel_name}{config.blk_values}_kernel_src.s"
    )

    # Compile kernel
    run_command(
        f"{config.clangpp} -fPIC -I {config.build_dir}/../../env_build/include -c {config.c_kernel} " #-fopenmp
        f"-o {out_obj_dir}/{kernel_name}.o"
    )

    # Create library
    run_command(
        f"{config.ar} rcs {config.lib_dir}/libc{kernel_name}.a {out_obj_dir}/{kernel_name}.o"
    )

    config.lib_name = f"c{kernel_name}"


def find_block_size_variants(kernel_aux_file_dir: Path) -> list:
    """Find all block size variant directories."""
    parent_dir = kernel_aux_file_dir.parent
    base_name = kernel_aux_file_dir.name
    variants = []
    
    if parent_dir.exists() and parent_dir.is_dir():
        for item in parent_dir.iterdir():
            if item.is_dir() and item.name.startswith(base_name + "_"):
                # print(item.name)
                variants.append(item)
    
    return variants


def build_triton_kernel(config: BuildConfig) -> None:
    """Build Triton kernel."""
    config.kernel_enable = "TRITON_KERNEL_ENABLE"
    kernel_name = os.path.basename(config.triton_kernel).replace(".py", "")

    # Clean if requested - add this section
    if config.clean_first:
        # Determine the actual kernel directory name based on mode
        if config.mode == 1:
            actual_kernel_name = kernel_name + "_single_block"
        elif config.mode == 2:
            actual_kernel_name = kernel_name + "_single_iteration"
        else:
            actual_kernel_name = kernel_name
        
        config.clean_auto_tuner_dirs(actual_kernel_name)

    # Build support library
    run_command(
        f"{config.clangpp} -fPIC -I {config.build_dir}/../../env_build/include "
        f"-c {config.src_dir}/support/*.cpp -o {config.obj_dir}/support.o"
    )

    run_command(
        f"{config.ar} rcs {config.lib_dir}/libsupport.a {config.obj_dir}/support.o"
    )

    kernel_name = os.path.basename(config.triton_kernel).replace(".py", "")
    base_kernel_aux_file_dir = config.kernel_launcher_src_dir / kernel_name / kernel_name
    #print("base_kernel_aux_file_dir:", base_kernel_aux_file_dir.name)
    
    # Run Python to generate kernel
    env = os.environ.copy()
    env["KERNEL_LAUNCHER_INCLUDE_DIR"] = str(config.kernel_launcher_include_dir)
    env["KERNEL_AUX_FILE_DIR"] = str(base_kernel_aux_file_dir)
    env["TRITON_CPU_BACKEND"] = "1"
    env["USE_BLOCK_POINTERS"] = "1"
    env["TRITON_ALWAYS_COMPILE"] = "1"
    
    # The following lines are used by debug flags
    # mlir_dump_dir = config.src_dir / "launcher" / "mlir_dump"
    # create_dir_if_not_exists(mlir_dump_dir)
    # mlir_dump_file = mlir_dump_dir / "dumped.mlir"  
    # reproducer_dir = config.src_dir / "launcher" / "reproducer"
    # create_dir_if_not_exists(reproducer_dir)
    # reproducer_file = reproducer_dir / "reproducer.mlir"

    # # Debug flags
    # env["MLIR_ENABLE_DUMP"] = "1"
    # env["MLIR_DUMP_PATH"] = str(mlir_dump_file)
    # env["TRITON_REPRODUCER_PATH"] = str(reproducer_file)

    print("Run tuning...")
    try:
        subprocess.run(
            [config.pyc, config.triton_kernel],
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Python to generate kernel: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")

    # Find all block size variants
    '''
    base_kernel_aux_file_dir = config.kernel_launcher_src_dir / kernel_name: 'launcher/src/matmul/matmul'
    base_kernel_aux_file_dir has the block size and other parameters in its name.
    '''
    variant_dirs = find_block_size_variants(base_kernel_aux_file_dir)
    if not variant_dirs:
        print(f"Warning: No block size variants found for {kernel_name}")
        variant_dirs = [base_kernel_aux_file_dir]  # fallback to base directory
    
    total_variants = len(variant_dirs)

    for idx, kernel_aux_file_dir in enumerate(variant_dirs, 1):
        # Set the flag for the last variant
        config.is_last_variant = (idx == total_variants)

        # Extract block values and group size from directory name
        if kernel_aux_file_dir != base_kernel_aux_file_dir:
            blk_json = kernel_aux_file_dir / "blk_constants.json"
            if not os.path.exists(blk_json):
                print(f"Warning: blk_constants.json not found in {kernel_aux_file_dir}")
                continue
            try:
                with open(blk_json, "r") as f:
                    blk_constants = json.load(f)

                block_fields = {}
                block_values = []
                
                for key, value in blk_constants.items():
                    if key.startswith("BLOCK"):
                        block_fields[key] = value
                        block_values.append(str(value))
                if block_fields:
                    config.block_fields = block_fields
                    combined_value = "_".join(block_values)
                    config.blk_values= f"_{combined_value}"
                    print(f"\nProcessing variant with block values: {combined_value}")
                else:
                    print(f"Warning: No block size constants found in {blk_json}")
                    config.blk_values = ""
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in {blk_json}: {e}")
                config.blk_values = ""
        else:
            config.blk_values = ""  # default case
            
        out_obj_dir = config.obj_dir / kernel_name
        create_dir_if_not_exists(out_obj_dir)

        # Setup auto-tuner directories for this kernel if not already set up
        if not hasattr(config, 'kernel_auto_tuner_dir'):
            config.setup_auto_tuner_dirs(kernel_name + "_single_block" if config.mode == 1 
                                         else kernel_name if config.mode == 0 
                                         else kernel_name + "_single_iteration")

        # Find all LLVM IR files in the kernel_aux_file_dir
        llir_files = list(kernel_aux_file_dir.glob("*.llir"))
        if not llir_files:
            print(f"Warning: No .llir files found in {kernel_aux_file_dir}")
            continue
        
        print(f"Found {len(llir_files)} LLVM IR files to process")
        
        # Process each LLVM IR file - convert to bitcode
        bitcode_files = []
        for llir_file in llir_files:
            llir_basename = llir_file.stem  # filename without extension
            bc_file = out_obj_dir / f"{llir_basename}.bc"
            
            print(f"Converting {llir_file.name} to bitcode")
            
            # Convert LLVM IR to bitcode
            run_command(
                f"{config.as_tool} -o {bc_file} {llir_file}"
            )
            
            bitcode_files.append(str(bc_file))

        # Link all bitcode files into one combined bitcode file
        combined_bc_file = out_obj_dir / f"{kernel_name}{config.blk_values}_combined.bc"
        bitcode_files_str = " ".join(bitcode_files)
        
        print(f"Linking {len(bitcode_files)} bitcode files into combined bitcode")
        run_command(
            f"llvm-link -o {combined_bc_file} {bitcode_files_str}"
        )
        
        # Generate assembly from combined bitcode
        kernel_src_asm = config.kernel_dump_dir / f"{config.vector_suffix}_{kernel_name}{config.blk_values}_kernel_src.s"
        run_command(
            f"{config.clangpp} -fPIC -S {combined_bc_file} -o {kernel_src_asm}"
        )
        print(f"Generated kernel assembly at: {kernel_src_asm}")
        
        # Compile combined bitcode to single object file
        combined_obj_file = out_obj_dir / f"{kernel_name}{config.blk_values}_kernel.o"
        run_command(
            f"{config.clangpp} -fPIC -c {combined_bc_file} -o {combined_obj_file}"
        )
        
        kernel_object_files = [str(combined_obj_file)]

        # Find and compile all launcher.cpp files
        launcher_files = list(kernel_aux_file_dir.glob("*launcher.cpp"))
        if not launcher_files:
            print(f"Warning: No launcher.cpp files found in {kernel_aux_file_dir}")
            continue

        print(f"Found {len(launcher_files)} launcher files to process")
        
        launcher_object_files = []
        for launcher_file in launcher_files:
            launcher_basename = launcher_file.stem  # filename without extension
            launcher_obj = out_obj_dir / f"{launcher_basename}.o"
            
            print(f"Processing {launcher_file.name}")
            
            run_command(
                f"{config.clangpp} -I {config.build_dir}/../../env_build/include "
                f"-fPIC -I {config.kernel_launcher_include_dir} -c {launcher_file} "
                f"-o {launcher_obj}"
            )
            
            launcher_object_files.append(str(launcher_obj))

        # Create static library combining all kernel objects and launcher objects
        lib_file = config.lib_dir / f"libtriton{kernel_name}{config.blk_values}.a"
        all_objects = launcher_object_files + kernel_object_files
        objects_str = " ".join(all_objects)
        
        run_command(
            f"{config.ar} rcs {lib_file} {objects_str}"
        )
        
        print(f"Created static library: {lib_file}")
        print(f"Library contains: {len(kernel_object_files)} kernel objects + {len(launcher_object_files)} launcher objects")

        config.lib_name = f"triton{kernel_name}{config.blk_values}"
        
        # Build final executable for this variant
        build_final_executable(config)

def setup_remote_connection(config: BuildConfig) -> None:
    """Set up SSH master connection for remote transfers."""
    if not config.transfer_to_remote:
        return

    print("\nSetting up SSH master connection...")
    remote = f"{config.remote_user}@{config.remote_host}"
    
    try:
        # Check if connection already exists
        exit_code, _, _ = run_command(
            f"ssh -O check -S {config.ssh_ctrl_socket} {remote}",
            check=False
        )
        
        if exit_code != 0:
            # Create new master connection
            run_command(f"ssh -M -S {config.ssh_ctrl_socket} -fnN {remote}")
            print("SSH master connection established")
    except Exception as e:
        print(f"Error setting up SSH connection: {e}")
        sys.exit(1)

def cleanup_remote_connection(config: BuildConfig) -> None:
    """Clean up SSH master connection."""
    if not config.transfer_to_remote:
        return

    print("\nCleaning up SSH master connection...")
    remote = f"{config.remote_user}@{config.remote_host}"
    
    try:
        run_command(f"ssh -S {config.ssh_ctrl_socket} -O exit {remote}")
        print("SSH master connection closed")
    except Exception as e:
        print(f"Error cleaning up SSH connection: {e}")

def transfer_to_remote(config: BuildConfig, kernel_name: str) -> None:
    """Transfer entire kernel directory to remote server."""
    if not config.transfer_to_remote:
        return

    print(f"\nTransferring {kernel_name} directory to remote server...")
    
    remote = f"{config.remote_user}@{config.remote_host}"
    remote_auto_tuner_dir = f"{config.remote_dir}/auto-tuner/{kernel_name}"
    
    try:
        # Create remote auto-tuner directory if it doesn't exist
        run_command(
            f"ssh -S {config.ssh_ctrl_socket} {remote} 'mkdir -p {remote_auto_tuner_dir}'"
        )
        
        # Transfer the entire kernel directory
        # config.kernel_auto_tuner_dir = self.auto_tuner_dir / kernel_name
        run_command(
            f"scp -r -o ControlPath={config.ssh_ctrl_socket} {config.kernel_auto_tuner_dir}/run {remote}:{remote_auto_tuner_dir}"
        )
        
        print(f"Successfully transferred {kernel_name} directory to {remote}:{remote_auto_tuner_dir}/")
        
    except Exception as e:
        print(f"Error during directory transfer: {e}")

def build_final_executable(config: BuildConfig) -> None:
    """Build final executable."""
    kernel_name = os.path.basename(config.kernel_source).split(".")[0]
    main = config.src_dir / "main" / f"{kernel_name}_kernel.cpp"

    if not main.is_file():
        print(f"Error: main file not found: {main}")
        sys.exit(1)

    # Construct output filename
    elf_name = (
        f"{config.vector_suffix}_{kernel_name}{config.blk_values}"
        f"{config.debug_suffix}{config.static_suffix}{config.opt_suffix}.elf"
    )
    out_file = config.kernel_bin_dir / elf_name

    # Build executable with CHECK_ACCURACY enabled
    run_command(
        # clang -### -c --target=riscv64-unknown-linux-gnu -fveclib=SLEEF -march=rv64gcv %s
        # /llvm_rvv/llvm/llvm-project/clang/test/Driver/fveclib.c
        f"{config.clangpp} {main} -I {config.build_dir}/../../env_build/include "
        f"-I {config.kernel_launcher_include_dir} -L {config.lib_dir} -L {config.build_dir}/../../env_build/sysroot/lib "
        f"-l{config.lib_name} -lsupport -latomic -lsleef -std=c++17 "
        f"-D{config.kernel_enable} -fPIC -o {out_file}"
    )

    print(f"The ELF with accuracy check is at {out_file}")

    # Generate and save assembly dump
    asm_file = config.kernel_dump_dir / f"{elf_name}.s"
    run_command(
        f'{config.objdump} -d -S --source-comment="@src " {out_file} > {asm_file}'
    )
    print(f"The complete ASM code is at {asm_file}")

    try:
        if config.block_fields:
            variant_info = {
                # "block_size": config.blk_values.strip("_").split("_") if config.blk_values else [],
                **config.block_fields,
                "elf_path": config.get_relative_path(str(out_file)),
                "asm_path": config.get_relative_path(str(asm_file)),
                "perf_data_path": config.get_relative_path(str(config.kernel_perf_data_dir / f"{elf_name}_perf.data")),
                "perf_report_path": config.get_relative_path(str(config.kernel_perf_data_dir / f"{elf_name}_perf_report.txt"))
            }
            
            with open(config.kernel_config_file, "r") as f:
                config_data = json.load(f)
            config_data["variants"].append(variant_info)

            with open(config.kernel_config_file, "w") as f:
                json.dump(config_data, f, indent=4)
                            
            print(f"Updated {config.kernel_config_file} with relative paths")
        else:
            print("Warning: No block size constants found, not updating config file")
    except Exception as e:
            print(f"Error updating kernel config file: {e}")

    if config.is_last_variant:
        if config.transfer_to_remote:
            # Set up SSH connection if needed
            setup_remote_connection(config)
            transfer_to_remote(config, kernel_name)

    if config.copy2Gem5:
        print("\nCopying ELF to gem5 directory...")
        try:
            run_command(f"cp {out_file} {config.gem5_des_path}/")
            print(f"Successfully copied {out_file} to {config.gem5_des_path}/")
        except Exception as e:
            print(f"Error copying file to gem5: {e}")


def parse_args() -> Tuple[str, str, str, bool]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build kernel from source")

    kernel_group = parser.add_mutually_exclusive_group(required=True)
    kernel_group.add_argument("-t", "--triton", help="Path to the Triton Python file")
    kernel_group.add_argument("-c", "--c-kernel", help="Path to the C++ kernel file")

    cur_path = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(cur_path, "..", "benchmark", "config.json")

    parser.add_argument(
        "-j", "--config", default=default_config_path, help="Path to the JSON configuration file"
    )

    parser.add_argument(
        "--clean", action="store_true", help="Clean build artifacts before building"
    )
    args = parser.parse_args()

    kernel_type = "triton" if args.triton else "c"
    kernel_source = args.triton if args.triton else args.c_kernel
    config_file = args.config
    clean_first = args.clean

    return kernel_type, kernel_source, config_file, clean_first


def main():
    """Main function."""
    #print("Starting script execution")

    kernel_type, kernel_source, config_file, clean_first = parse_args()

    # Initialize configuration
    config = BuildConfig()
    config.kernel_type = kernel_type
    config.kernel_source = kernel_source
    config.config_file = config_file
    config.clean_first = clean_first

    if kernel_type == "triton":
        config.triton_kernel = kernel_source
    else:
        config.c_kernel = kernel_source

    # Validate inputs
    if not os.path.isfile(kernel_source):
        print(f"Error: Source file not found: {kernel_source}")
        sys.exit(1)

    if not os.path.isfile(config_file):
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)

    # Check if the kernel source has a configuration in the JSON file
    json_key = find_source_key_in_json(Path(config_file), kernel_source)
    if not json_key:
        print(
            f"Error: Configuration for {os.path.basename(kernel_source)} not found in {config_file}"
        )
        print(
            f"Tried looking for: {kernel_source}, {os.path.basename(kernel_source)}, and relative paths"
        )
        sys.exit(1)

    config.json_key = json_key

    # Load configuration
    load_config_from_json(config)

    # Setup directories
    setup_directories(config)

    # Configure compiler flags
    configure_compiler_flags(config)

    # Build kernel based on type
    if config.kernel_type == "triton":
        build_triton_kernel(config)
    elif config.kernel_type == "c":
        build_c_kernel(config)

    # Clean up SSH connection if needed
    cleanup_remote_connection(config)

    print("Script execution completed successfully")


if __name__ == "__main__":
    main()
