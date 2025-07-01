import subprocess
import sys
import os
import argparse
from typing import Tuple

'''Usage: python3 auto-tuner/report_to_host.py <kernel_name>'''

class Config:
    """Configuration class to hold all settings."""
    def __init__(self):
        self.remote_host = "10.26.1.50"
        self.remote_user = "yuhao"
        self.ssh_ctrl_socket = "/tmp/ssh-control-socket"
        self.kernel_auto_tuner_dir = os.path.dirname(os.path.abspath(__file__))
        self.remote_dir = "/home/yuhao/T_RVV/benchmark/auto-tuner"
        self.ssh_password = None


def run_command(cmd: str, check: bool = True, input_text: str = None) -> Tuple[int, str, str]:
    """Execute a command and return the output"""
    print(f"Running command: {cmd}")
    try:
        kwargs = {
            'shell': True,
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'text': False  # Changed from True to False
        }

        if input_text is not None:
            if isinstance(input_text, str):
                input_text = input_text.encode('utf-8')
            kwargs['input'] = input_text
            print("(With password input)")

        result = subprocess.run(cmd, **kwargs)

        # Safely decode stdout and stderr
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')

        if check and result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"Error: {stderr}")
            if check:
                raise Exception(f"Command failed: {cmd}")

        return result.returncode, stdout, stderr

    except Exception as e:
        print(f"Exception while executing command: {e}")
        if check:
            raise
        return 1, "", str(e)


def setup_remote_connection(config: Config) -> None:
    """Set up SSH master connection for remote transfers."""

    print("\nSetting up SSH master connection...")
    remote = f"{config.remote_user}@{config.remote_host}"

    try:
        # Check if connection already exists
        exit_code, _, _ = run_command(
            f"ssh -p 5000 -O check -S {config.ssh_ctrl_socket} {remote}",
            check=False
        )

        if exit_code != 0:
            # Create new master connection
            run_command(f"ssh -p 5000 -M -S {config.ssh_ctrl_socket} -fnN {remote}")
            print("SSH master connection established")
    except Exception as e:
        print(f"Error setting up SSH connection: {e}")
        sys.exit(1)


def cleanup_remote_connection(config: Config) -> None:
    """Clean up SSH master connection."""

    print("\nCleaning up SSH master connection...")
    remote = f"{config.remote_user}@{config.remote_host}"

    try:
        run_command(f"ssh -p 5000 -S {config.ssh_ctrl_socket} -O exit {remote}")
        print("SSH master connection closed")
    except Exception as e:
        print(f"Error cleaning up SSH connection: {e}")


def transfer_to_remote(config: Config, kernel_name: str) -> None:
    """Transfer entire kernel directory to remote server."""

    print(f"\nTransferring {kernel_name} directory to remote server...")

    remote = f"{config.remote_user}@{config.remote_host}"
    remote_auto_tuner_dir = f"{config.remote_dir}"

    try:
        # Create remote auto-tuner directory if it doesn't exist
        run_command(
            f"ssh -p 5000 -S {config.ssh_ctrl_socket} {remote} 'mkdir -p {remote_auto_tuner_dir}/{kernel_name}/run/perf'"
        )

        # Transfer the entire kernel directory
        run_command(
            f"scp -P 5000 -r -o ControlPath={config.ssh_ctrl_socket} {config.kernel_auto_tuner_dir}/{kernel_name}/run/perf/perf_stats.csv {remote}:{remote_auto_tuner_dir}/{kernel_name}/run/perf"
        )

        print(f"Successfully transferred {kernel_name} directory to {remote}:{remote_auto_tuner_dir}/{kernel_name}/run/perf")

    except Exception as e:
        print(f"Error during directory transfer: {e}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Transfer kernel directory to remote server')
    parser.add_argument('kernel_name', type=str, help='Name of the kernel to transfer')
    parser.add_argument('--remote-host', type=str, help='Remote host address')
    parser.add_argument('--remote-user', type=str, help='Remote username')
    parser.add_argument('--local-dir', type=str, help='Local directory containing kernel files')
    parser.add_argument('--remote-dir', type=str, help='Remote directory to transfer files to')
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Create and configure the Config object
    config = Config()
    
    # Override config values with command line arguments if provided
    if args.remote_host:
        config.remote_host = args.remote_host
    if args.remote_user:
        config.remote_user = args.remote_user
    if args.local_dir:
        config.kernel_auto_tuner_dir = args.local_dir
    if args.remote_dir:
        config.remote_dir = args.remote_dir

    # Get kernel name from command line arguments
    kernel_name = args.kernel_name
    print(f"Working with kernel: {kernel_name}")

    try:
        # Set up remote connection
        setup_remote_connection(config)

        # Transfer the kernel directory to remote server
        transfer_to_remote(config, kernel_name)

        # Clean up remote connection
        cleanup_remote_connection(config)
        
        print(f"Transfer completed successfully for kernel: {kernel_name}")
    except Exception as e:
        print(f"Error during transfer process: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
