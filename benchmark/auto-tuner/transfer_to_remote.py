import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass, field


'''
Usage: python3 benchmark/auto-tuner/transfer_to_remote.py <kernel_name>
'''

@dataclass
class BuildConfig:
    """Configuration for building a kernel."""

    # Path constants
    script_dir: Path = field(
        default_factory=lambda: Path(os.path.dirname(os.path.abspath(__file__)))
    )

    # New fields for auto-tuner structure
    auto_tuner_dir: Path = field(init=False)
    kernel_auto_tuner_dir: Path = field(init=False)

    # Gem5 Simulator Path
    copy2Gem5: bool = field(default=False)
    gem5_des_path: Path = field(default=Path("/home/yuhao/gem5/tt_rvv/triton/bin"))

    remote_user: str = field(default="yuhao")
    remote_host: str = field(default="10.32.44.164")
    remote_dir: str = field(default="/home/yuhao/triton_riscv_test/triton/auto-tuner")
    ssh_ctrl_socket: str = field(default="/tmp/ssh_ctrl_socket")

    def __post_init__(self):
        # Initialize the auto_tuner_dir
        self.auto_tuner_dir = self.script_dir


def run_command(cmd: str, check: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and return its exit code, stdout and stderr."""
    #print(f"Running: {cmd}")
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    exit_code = process.returncode


    if exit_code != 0 and check:
        print(f"Error: Command failed with code {exit_code}")
        print(f"stderr: {stderr}")
        sys.exit(exit_code)

    return exit_code, stdout, stderr

def setup_remote_connection(config: BuildConfig) -> None:
    """Set up SSH master connection for remote transfers."""
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


def transfer_to_remote(config: BuildConfig, kernel_name: str) -> None:
    """Transfer kernel_name/run directory to remote server."""

    print(f"\nTransferring {kernel_name}/run directory to remote server...")
    
    remote = f"{config.remote_user}@{config.remote_host}"
    # Update kernel_auto_tuner_dir based on kernel_name
    config.kernel_auto_tuner_dir = config.auto_tuner_dir / kernel_name
    remote_kernel_dir = f"{config.remote_dir}/{kernel_name}"
    
    try:
        # Create remote kernel directory if it doesn't exist
        run_command(
            f"ssh -S {config.ssh_ctrl_socket} {remote} 'mkdir -p {remote_kernel_dir}'"
        )
        
        # Transfer just the run directory
        local_run_dir = config.kernel_auto_tuner_dir / "run"
        if not local_run_dir.exists():
            print(f"Error: Run directory not found at {local_run_dir}")
            return
            
        run_command(
            f"scp -r -o ControlPath={config.ssh_ctrl_socket} {local_run_dir} {remote}:{remote_kernel_dir}/"
        )
        
        print(f"Successfully transferred {kernel_name}/run directory to {remote}:{remote_kernel_dir}/")
        
    except Exception as e:
        print(f"Error during directory transfer: {e}")


def cleanup_remote_connection(config: BuildConfig) -> None:
    """Clean up SSH master connection."""
    print("\nCleaning up SSH master connection...")
    remote = f"{config.remote_user}@{config.remote_host}"
    
    try:
        run_command(f"ssh -S {config.ssh_ctrl_socket} -O exit {remote}")
        print("SSH master connection closed")
    except Exception as e:
        print(f"Error cleaning up SSH connection: {e}")


def main():
    """Main function to handle command line arguments and execute transfers."""
    parser = argparse.ArgumentParser(description="Transfer kernel directory to remote server")
    parser.add_argument("kernel_name", help="Name of the kernel to transfer")
    
    args = parser.parse_args()
    
    # Create and configure the build config
    config = BuildConfig()
    
    try:
        # Set up SSH connection
        setup_remote_connection(config)
        
        # Transfer the kernel directory
        transfer_to_remote(config, args.kernel_name)
    finally:
        # Always clean up the SSH connection
        cleanup_remote_connection(config)


if __name__ == "__main__":
    main()