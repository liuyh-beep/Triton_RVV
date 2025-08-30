import subprocess
import re
import csv
import sys
import os
import json
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any

'''perf script for running perf stat and record/report for cache events, integrated into a unified CSV output'''

""" This script is used to run `perf stat` and `perf record` for L2 cache event """

"""Usage: python3 run_perf.py {kernel_name}/configs.json"""


# Configuration parameters
REMOTE_HOST = "10.26.1.50"  # Remote hostname
REMOTE_USER = "yuhao"  # Remote username
SSH_PORT = 5000  # SSH port
SSH_CTRL_SOCKET = "/tmp/ssh-control-socket"  # Path to SSH control socket
KERNEL_AUTO_TUNER_DIR = "."  # Local directory containing kernel files
REMOTE_AUTO_TUNER_DIR = "/home/yuhao/T_RVV/benchmark/auto-tuner"  # Remote directory to transfer files to
TRANSFER_TO_REMOTE = False  # Whether to transfer files to remote server


class Config:
    """Configuration class to hold all settings."""
    def __init__(self):
        self.remote_host = REMOTE_HOST
        self.remote_user = REMOTE_USER
        self.ssh_port = SSH_PORT
        self.ssh_ctrl_socket = SSH_CTRL_SOCKET
        self.kernel_auto_tuner_dir = KERNEL_AUTO_TUNER_DIR
        self.remote_dir = REMOTE_AUTO_TUNER_DIR
        self.transfer_to_remote = TRANSFER_TO_REMOTE
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
    if not config.transfer_to_remote:
        return

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
    if not config.transfer_to_remote:
        return

    print("\nCleaning up SSH master connection...")
    remote = f"{config.remote_user}@{config.remote_host}"

    try:
        run_command(f"ssh -p 5000 -S {config.ssh_ctrl_socket} -O exit {remote}")
        print("SSH master connection closed")
    except Exception as e:
        print(f"Error cleaning up SSH connection: {e}")


def transfer_to_remote(config: Config, kernel_name: str) -> None:
    """Transfer entire kernel directory to remote server."""
    if not config.transfer_to_remote:
        return

    print(f"\nTransferring {kernel_name} directory to remote server...")

    remote = f"{config.remote_user}@{config.remote_host}"
    remote_auto_tuner_dir = f"{config.remote_dir}"

    try:
        # Create remote auto-tuner directory if it doesn't exist
        run_command(
            f"ssh -p 5000 -S {config.ssh_ctrl_socket} {remote} 'mkdir -p {remote_auto_tuner_dir}'"
        )

        # Transfer the entire kernel directory
        run_command(
            f"scp -P 5000 -r -o ControlPath={config.ssh_ctrl_socket} {config.kernel_auto_tuner_dir}/{kernel_name}/perf {remote}:{remote_auto_tuner_dir}/{kernel_name}"
        )

        print(f"Successfully transferred {kernel_name} directory to {remote}:{remote_auto_tuner_dir}/")

    except Exception as e:
        print(f"Error during directory transfer: {e}")


def check_elf_output(elf_path: str) -> Tuple[bool, str, str, str]:
    """
    Run the ELF file and check if its output is correct.
    Returns a tuple of (is_correct, stdout, stderr, kernel_time)
    """
    print(f"Checking output of {elf_path}...")
    
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    try:
        # Run the ELF file without perf
        result = subprocess.run(
            elf_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        # Extract kernel time if available
        kernel_time = "N/A"
        kernel_time_match = re.search(r'[Rr]unning\s+[Tt]riton\s+[Kk]ernel\s+[Tt]ime:\s*([\d.]+)\s*s', stderr)
        if kernel_time_match:
            kernel_time = kernel_time_match.group(1)
            
        # Check if output contains "out OK"
        not_correct = "NOT OK" in stdout
        
        if not_correct:
            print(f"ELF output is INCORRECT: {elf_path}")
            if "NOT OK" in stdout:
                print("Output verification failed")
            else:
                print("Unable to determine output correctness")
                
        return not_correct, stdout, stderr, kernel_time
        
    except Exception as e:
        print(f"Error running ELF: {e}")
        return False, "", str(e), "N/A"


def extract_parameters(variant: Dict[str, Any]) -> Dict[str, int]:
    """Extract all BLOCK_SIZE parameters from the variant dictionary"""
    params = {}
    
    # Extract all BLOCK_SIZE parameters
    for key, value in variant.items():
        if key.startswith('BLOCK_') and isinstance(value, int):
            params[key] = value
    
    # Check if block_size list is available as a fallback
    if not params and 'block_size' in variant and isinstance(variant['block_size'], list):
        block_size = variant['block_size']
        if len(block_size) >= 1:
            params['BLOCK_SIZE_M'] = int(block_size[0])
        if len(block_size) >= 2:
            params['BLOCK_SIZE_N'] = int(block_size[1])
        if len(block_size) >= 3:
            params['BLOCK_SIZE_K'] = int(block_size[2])
    
    return params


def run_perf_stat(elf_path: str) -> Tuple[str, str, str]:
    """run perf stat with all events and get stat data"""
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    # Combined all event groups into one command
    all_events = ('u_mode_cycle:u,eu_vfpu_full:u,vidu_total_cycle:u,vector_micro_op:u,'
                 'vector_load_inst:u,vector_store_inst:u,vector_inst:u,'
                 'l1d_load_miss:u,l1d_load_access:u,l1d_access:u,l1d_miss:u')
                 # 'l2_access,l2_miss,l2_load_access,l2_store_access'

    # Modify perf command to capture both program output and perf stats
    perf_cmd = ['perf', 'stat', '-e', all_events,
                '--output', 'perf.txt',  # Save perf stats to a file
                elf_path]

    # Run the command and capture both stdout and stderr
    result = subprocess.run(perf_cmd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True,
                          env=env)
    # Read perf stats from file
    with open('perf.txt', 'r') as f:
        perf_output = f.read()

    # Clean up temporary file
    os.remove('perf.txt')

    return result.stdout, perf_output, result.stderr  # Return all outputs


def parse_perf_output(stdout: str, perf_output: str, stderr: str) -> Dict[str, Any]:
    """parse all perf stat output metrics"""
    metrics = {}

    # Look for kernel time in program output (stderr)
    kernel_time_match = re.search(r'[Rr]unning\s+[Tt]riton\s+[Kk]ernel\s+[Tt]ime:\s*([\d.]+)\s*s', stderr)
    if kernel_time_match:
        metrics['kernel_time'] = float(kernel_time_match.group(1))

    # Define all counter patterns in one dictionary
    counter_patterns = {
        # Micro operation metrics
        'u_mode_cycle': r'([\d,]+)\s+u_mode_cycle:u',
        'eu_vfpu_full': r'([\d,]+)\s+eu_vfpu_full:u',
        'vidu_total_cycle': r'([\d,]+)\s+vidu_total_cycle:u',
        'vector_micro_op': r'([\d,]+)\s+vector_micro_op:u',
        
        # Instruction metrics
        'vector_load_inst': r'([\d,]+)\s+vector_load_inst:u',
        'vector_store_inst': r'([\d,]+)\s+vector_store_inst:u',
        'vector_inst': r'([\d,]+)\s+vector_inst:u',
        
        # Cache metrics
        'l1d_load_miss': r'([\d,]+)\s+l1d_load_miss:u',
        'l1d_load_access': r'([\d,]+)\s+l1d_load_access:u',
        'l1d_access': r'([\d,]+)\s+l1d_access:u',
        'l1d_miss': r'([\d,]+)\s+l1d_miss:u'
    }

    # Parse all perf counters
    for metric, pattern in counter_patterns.items():
        match = re.search(pattern, perf_output)
        if match:
            value = match.group(1).replace(',', '')
            metrics[metric] = int(value)
        else:
            metrics[metric] = 0  # Set to 0 if not found
            print(f"metric {metric} is zero.")

    # Parse total time from perf output
    time_match = re.search(r'([\d.]+) seconds time elapsed', perf_output)
    if time_match:
        metrics['total_time'] = float(time_match.group(1))

    return metrics


def perf_stat(elf_path: str, not_valid: bool = True) -> Dict[str, Any]:
    '''
    run perf stat for all metrics and parse its output
    '''
    # If the ELF output is not valid, return metrics with error information
    if not_valid:
        return {
            'error': 'INCORRECT_OUTPUT',
            'kernel_time': 0.0,
            'total_time': 0.0
        }
    
    metrics = {}

    print(f"Running {elf_path} with perf events...")

    stdout, perf_output, stderr = run_perf_stat(elf_path)
    metrics = parse_perf_output(stdout, perf_output, stderr)

    return metrics


def save_to_csv(metrics: Dict[str, Any], params: Dict[str, int], csv_file: str, elf_name: str, 
                not_valid: bool = True, l2_metrics: Dict[str, Any] = None) -> None:
    """
    Save all metrics to a single CSV file, including L2 cache metrics from perf record/report
    
    Args:
        metrics: Dictionary containing the metrics to save
        params: Dictionary containing block size parameters
        csv_file: Path to the CSV file
        elf_name: Name of the ELF file
        not_valid: Whether the ELF output is invalid
        l2_metrics: L2 cache metrics from perf record/report
    """
    file_exists = os.path.exists(csv_file)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write headers only if the file is new
        if not file_exists:
            # Combined headers for all metric types
            headers = ['Timestamp', 'ELF Name']
            
            # Add all block size parameters dynamically
            # Collect all possible block size parameters from the current run
            all_block_params = list(params.keys())
            headers.extend(all_block_params)
            
            headers.extend(['Valid Output', 'Kernel Time (s)', 'Total Time (s)'])
                        
            # Add micro metrics
            headers.extend(['u_mode_cycle', 'eu_vfpu_full', 'vidu_total_cycle', 'vector_micro_op'])
            
            # Add instruction metrics
            headers.extend(['vector_load_inst', 'vector_store_inst', 'vector_inst'])
            
            # Add cache metrics
            headers.extend(['l1d_load_miss', 'l1d_load_access', 'l1d_access', 'l1d_miss',
                            'l2_access(perf record)', 'l2_miss(perf record)', 
                            'l2_load_access(perf record)', 'l2_store_access(perf record)'])
            
            writer.writerow(headers)

        # Build the row
        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            elf_name
        ]
        
        # Add all block size parameters in the correct order
        all_block_params = list(params.keys())
        # all_block_params.sort()  # Sort for consistent ordering
        
        for param in all_block_params:
            row.append(params[param])
        
        # Add validity, kernel time, and total time
        row.extend([
            "NO" if not_valid else "YES",
            format_value(metrics.get('kernel_time', '')) if 'kernel_time' in metrics else "N/A",
            format_value(metrics.get('total_time', '')) if 'total_time' in metrics else "N/A"
        ])

        # If output is invalid, add empty values for all performance metrics
        if not_valid:
            # Add N/A for all performance metrics (11 + 4 L2 metrics)
            row.extend(["N/A"] * 15)
        else:
            # Add micro metrics
            row.extend([
                format_value(metrics.get('u_mode_cycle', 0), False),
                format_value(metrics.get('eu_vfpu_full', 0), False),
                format_value(metrics.get('vidu_total_cycle', 0), False),
                format_value(metrics.get('vector_micro_op', 0), False)
            ])
            
            # Add instruction metrics
            row.extend([
                format_value(metrics.get('vector_load_inst', 0), False),
                format_value(metrics.get('vector_store_inst', 0), False),
                format_value(metrics.get('vector_inst', 0), False)
            ])
            
            # Add L1 cache metrics
            row.extend([
                format_value(metrics.get('l1d_load_miss', 0), False),
                format_value(metrics.get('l1d_load_access', 0), False),
                format_value(metrics.get('l1d_access', 0), False),
                format_value(metrics.get('l1d_miss', 0), False)
            ])
            
            # Add L2 cache metrics from perf record/report if available
            if l2_metrics and 'events' in l2_metrics:
                l2_events = l2_metrics['events']
                row.extend([
                    format_value(l2_events.get('l2_access:u', {}).get('period', 0), False),
                    format_value(l2_events.get('l2_miss:u', {}).get('period', 0), False),
                    format_value(l2_events.get('l2_load_access:u', {}).get('period', 0), False),
                    format_value(l2_events.get('l2_store_access:u', {}).get('period', 0), False)
                ])
            else:
                # Add zeros if L2 metrics are not available
                row.extend(["0", "0", "0", "0"])

        writer.writerow(row)


def format_value(value, is_float=True):
    """Format value for CSV output"""
    if value == '':
        return ''
    if is_float:
        try:
            return f"{float(value):.6f}"
        except (ValueError, TypeError):
            return "0.000000"
    try:
        return f"{int(value)}"
    except (ValueError, TypeError):
        return "0"


################# perf record and report ###########################

def run_perf_record_report(perf_data_path: str, perf_report_path: str, elf_path: str, kernel_name: str, cache_event: str) -> Dict[str, Any]:
    """
    Run perf record and perf report for a specific cache event on an ELF file,
    then extract and return the data.

    Args:
        perf_data_path: Path to save the perf.data file
        perf_report_path: Path to save the perf report TXT file
        elf_path: Path to the ELF file to profile
        kernel_name: Name of the kernel to filter in perf report
        cache_event: Cache event to profile (e.g., "l1d_load_miss")

    Returns:
        Dictionary containing the extracted metrics
    """
    # Get the directory containing the ELF file
    elf_dir = os.path.dirname(elf_path)
    elf_name = os.path.basename(elf_path)

    # Create output directory for perf data if it doesn't exist
    perf_data_dir = os.path.dirname(perf_data_path)
    os.makedirs(perf_data_dir, exist_ok=True)

    print(f"Running perf record for {elf_name} with event {cache_event}...")

    # Run perf record
    record_cmd = f"perf record -o {perf_data_path} -e {cache_event} -F 4001 {elf_path} "
    returncode, _, stderr = run_command(record_cmd, check=False)

    if returncode != 0:
        print(f"Warning: perf record command failed: {stderr}")
        return {"error": f"perf record failed: {stderr}"}

    kernel_name = kernel_name.removesuffix("_uncomplete")
    report_cmd = f"perf report -i {perf_data_path} -S {kernel_name}_kernel --show-total-period --stdio > {perf_report_path}"
    returncode, _, stderr = run_command(report_cmd, check=False)

    if returncode != 0:
        print(f"Warning: perf report command failed: {stderr}")
        return {"error": f"perf report failed: {stderr}"}

    # Extract data from the report file
    try:
        with open(perf_report_path, 'r') as f:
            report_content = f.read()
            #print(f"Report file size: {len(report_content)} bytes")

        # Extract metrics from the report content
        metrics = extract_perf_metrics(report_content, cache_event)
        return metrics

    except Exception as e:
        print(f"Error extracting data from perf report: {e}")
        return {"error": f"Failed to extract data: {str(e)}"}

def extract_perf_metrics(report_content: str, cache_events_str: str) -> Dict[str, Any]:
    """
    Extract relevant metrics from the perf report output for multiple cache events.

    Args:
        report_content: Content of the perf report file
        cache_events_str: Comma-separated list of cache events that were profiled

    Returns:
        Dictionary containing the extracted metrics for all events
    """
    # Parse the cache events string into a list
    cache_events = cache_events_str.split(',')
    #print(f"Looking for events: {cache_events}")
    
    # Print some of the report content for debugging
    #print(f"Report content sample (first 200 chars): {report_content[:200]}...")

    # Initialize metrics dictionary with each event
    metrics = {
        'events': {}
    }

    # Find all samples sections and their event counts in the report
    # This pattern matches lines like "# Samples: 23K of event 'l2_access:u'"
    samples_sections = re.findall(
        r"#\s*Samples:\s+([\d.]+)([kKMG]?)\s+of\s+event\s+'([^']+)'", 
        report_content
    )
    
    #print(f"Found {len(samples_sections)} sample sections in the report")
    
    for count_str, unit, event_name in samples_sections:
        #print(f"Found event: {event_name} with {count_str}{unit} samples")
        
        # Convert count to a number if needed
        # (Not used in the current implementation but might be useful later)
        
        # Check if this is one of our target events
        base_event = event_name.split(':')[0] if ':' in event_name else event_name
        if base_event not in cache_events and event_name not in cache_events:
            print(f"  Skipping non-target event: {event_name}")
            continue
            
        # Find the section of the report for this event
        event_section_start = report_content.find(f"# Samples: {count_str}{unit} of event '{event_name}'")
        if event_section_start == -1:
            print(f"  Could not locate section for event {event_name}")
            continue
            
        # Find the end of this event's section (next event section or end of file)
        next_section = report_content.find("# Samples:", event_section_start + 10)
        if next_section != -1:
            event_section = report_content[event_section_start:next_section]
        else:
            event_section = report_content[event_section_start:]
            
        # Find the period and overhead in this section
        # This pattern matches lines like "97.81%     158127910  VL256_matmul_4_  VL256_matmul_4_8_8_g_static_O2.elf"
        period_match = re.search(r"(\d+\.\d+)%\s+(\d+)", event_section)
        
        if period_match:
            overhead_percent = float(period_match.group(1))
            period = int(period_match.group(2))
            #print(f"  Found for {event_name}: {overhead_percent}% overhead, {period} period")
            
            metrics['events'][event_name] = {
                'overhead_percent': overhead_percent,
                'period': period
            }
        else:
            print(f"  No period information found for {event_name}")
            metrics['events'][event_name] = {
                'overhead_percent': 0.0,
                'period': 0
            }
    
    # Make sure all target events are in the metrics, even if not found in the report
    for event in cache_events:
        if event not in metrics['events']:
            print(f"Event {event} not found in report, adding with zero values")
            metrics['events'][event] = {
                'overhead_percent': 0.0,
                'period': 0
            }
    
    return metrics

############## perf record and report end ####################

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = Config()

    # Load configuration from JSON file
    try:
        with open(config_file, 'r') as f:
            json_config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)

    # Update config with values from JSON
    perf_files = json_config.get('perf_files', {})
    perf_stat_csv = perf_files.get('stat')

    # Delete existing CSV file at the beginning of script execution
    if os.path.exists(perf_stat_csv):
        print(f"Deleting existing CSV file at startup: {perf_stat_csv}")
        os.remove(perf_stat_csv)

    # Get kernel name from config file path
    config_dir = os.path.dirname(os.path.abspath(config_file))
    kernel_name = os.path.basename(os.path.dirname(config_dir))
    if not kernel_name or kernel_name == ".":
        # Fallback if we can't determine kernel name from path
        kernel_name = "unknown_kernel"

    print(f"Working with kernel: {kernel_name}")

    try:
        for variant in json_config.get('variants', []):
            elf_path = variant.get('elf_path')
            
            if not elf_path:
                print(f"Warning: Missing elf_path in variant: {variant}")
                continue

            elf_name = os.path.basename(elf_path)
            
            # Extract all parameters from the variant configuration
            params = extract_parameters(variant)
            
            if not params:
                print(f"Warning: No block size parameters found in variant: {variant}")
                # Create a minimal set of parameters to avoid errors
                params = {'BLOCK_SIZE': 0}
                
            # First check if the ELF output is correct
            not_valid, _, _, kernel_time = check_elf_output(elf_path)

            # Create simple metrics if output is invalid
            if not_valid:
                # Extract kernel time if available
                kernel_time_value = 0.0
                try:
                    kernel_time_value = float(kernel_time) if kernel_time != "N/A" else 0.0
                except ValueError:
                    pass
                    
                error_metrics = {
                    'error': 'INCORRECT_OUTPUT',
                    'kernel_time': kernel_time_value,
                    'total_time': 0.0
                }
                
                # Save error metrics to CSV without L2 metrics
                save_to_csv(error_metrics, params, perf_stat_csv, elf_name, not_valid)
                print(f"Warning: Recorded invalid output for {elf_name} in {perf_stat_csv}")
            else:
                # Run perf stat for all metrics
                metrics = perf_stat(elf_path, not_valid)
                
                # Run perf record and report for L2 cache events
                perf_data_path = variant.get('perf_data_path')
                perf_report_path = variant.get('perf_report_path')
                
                l2_metrics = None
                if perf_data_path and perf_report_path:
                    # Define L2 cache events to profile
                    cache_events = "l2_access:u,l2_miss:u,l2_load_access:u,l2_store_access:u"
                    l2_metrics = run_perf_record_report(perf_data_path, perf_report_path, elf_path, kernel_name, cache_events)
                    if 'error' in l2_metrics:
                        print(f"Warning: L2 cache profiling failed: {l2_metrics['error']}")
                        l2_metrics = None
                
                # Save all metrics to CSV including L2 metrics from perf record/report
                save_to_csv(metrics, params, perf_stat_csv, elf_name, not_valid, l2_metrics)
                print(f"All perf metrics for {elf_name} saved to {perf_stat_csv}")

        # Transfer files after all processing is done
        if config.transfer_to_remote:
            print(f"CSV files generated: {perf_stat_csv}")

            # Setup SSH connection and transfer files
            setup_remote_connection(config)
            transfer_to_remote(config, kernel_name)

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

