#!/usr/bin/env python3

'''
python3 plot_making.py --kernel-dir matmul
'''

#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from pathlib import Path
import re
from itertools import combinations

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze BLOCK_SIZE parameter impact on performance')
    parser.add_argument('--csv', required=True, help='Path to CSV file with performance data')
    parser.add_argument('--events', default='l1d_load_miss,l1d_load_access', 
                       help='Comma-separated list of performance events to plot')
    parser.add_argument('--output-dir', default='./plots', help='Directory to save plots')
    parser.add_argument('--figsize', nargs=2, type=int, default=[15, 8], 
                       help='Figure size (width height)')
    return parser.parse_args()

def load_and_validate_data(csv_path):
    """Load CSV data and validate it has the required columns"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from {csv_path}")
        
        # Check for required columns
        required_cols = ['Kernel Time (s)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

def find_block_size_parameters(df):
    """Find all columns that start with 'BLOCK_SIZE'"""
    block_size_cols = [col for col in df.columns if col.startswith('BLOCK_SIZE')]
    print(f"Found BLOCK_SIZE parameters: {block_size_cols}")
    return block_size_cols

def validate_events(df, events):
    """Validate that requested events exist in the dataframe"""
    available_events = [col for col in df.columns if col not in ['Timestamp', 'ELF Name', 'Valid Output']]
    missing_events = [event for event in events if event not in available_events]
    
    if missing_events:
        print(f"Warning: Events not found in data: {missing_events}")
        print(f"Available events: {available_events}")
        events = [event for event in events if event in available_events]
    
    return events

def prepare_data_for_clustering(df, target_param, other_params, events):
    """Prepare data for clustered visualization"""
    print(f"Preparing data for {target_param}...")
    
    # Check if target parameter exists
    if target_param not in df.columns:
        print(f"Error: {target_param} not found in data!")
        return pd.DataFrame()
    
    # Start with all data
    df_work = df.copy()
    
    # Filter by Valid Output if column exists and has True values
    if 'Valid Output' in df.columns:
        valid_mask = df_work['Valid Output'] == "YES"
        if valid_mask.any():
            df_work = df_work[valid_mask]
            print(f"Filtered by Valid Output: {len(df_work)} rows remaining")
    
    # Check required columns exist
    required_cols = [target_param] + other_params + ['Kernel Time (s)']
    available_cols = [col for col in required_cols if col in df_work.columns]
    missing_cols = [col for col in required_cols if col not in df_work.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
    
    # Add available event columns
    available_events = [event for event in events if event in df_work.columns]
    all_required = available_cols + available_events
    
    # Remove rows with missing values only in available columns
    df_clean = df_work.dropna(subset=all_required)
    print(f"After removing missing values: {len(df_clean)} rows")
    
    if len(df_clean) == 0:
        print("No valid data after filtering!")
        return pd.DataFrame()
    
    # Create cluster labels (combinations of other parameters)
    if other_params:
        available_other_params = [p for p in other_params if p in df_clean.columns]
        if available_other_params:
            # Create more readable cluster labels
            df_clean['cluster_label'] = df_clean[available_other_params].apply(
                lambda x: 'x'.join([str(val) for val in x]), axis=1
            )
        else:
            df_clean['cluster_label'] = 'baseline'
    else:
        df_clean['cluster_label'] = 'baseline'
    
    return df_clean

def plot_combined_line_bar_chart(df, target_param, other_params, events, output_dir, figsize):
    """Create a chart with lines for events and bars for Kernel Time on dual y-axes"""
    
    # Prepare data
    df_plot = prepare_data_for_clustering(df, target_param, other_params, events)
    
    if len(df_plot) == 0:
        print(f"No valid data for parameter {target_param}")
        return
    
    # Get unique values for clustering
    target_values = sorted(df_plot[target_param].unique())
    cluster_labels = sorted(df_plot['cluster_label'].unique())
    
    # Aggregate data (mean values for each combination)
    all_metrics = events + ['Kernel Time (s)']
    grouped_data = df_plot.groupby(['cluster_label', target_param])[all_metrics].mean().reset_index()
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Only create second y-axis if we have events to plot
    if events:
        ax2 = ax1.twinx()
    else:
        ax2 = None
    
    # Create title
    other_param_names = [p.split("_")[-1] for p in other_params] if other_params else ['baseline']
    fig.suptitle(f'Impact of {target_param} on Performance\n(Clustered by {" × ".join(other_param_names)})', 
                 fontsize=16, fontweight='bold')
    
    # Prepare positions for clustered elements
    x_pos = np.arange(len(cluster_labels))
    bar_width = 0.8 / len(target_values)
    
    # Color palettes
    line_colors = plt.cm.tab10(np.linspace(0, 1, len(events))) if events else []
    bar_colors = plt.cm.Set3(np.linspace(0, 1, len(target_values)))
    
    # Plot events as lines on left y-axis
    if events and ax2 is not None:
        for event_idx, event in enumerate(events):
            if event in grouped_data.columns:
                # Prepare data for line plot
                line_data = []
                x_positions = []
                
                for cluster_idx, cluster in enumerate(cluster_labels):
                    cluster_data = grouped_data[grouped_data['cluster_label'] == cluster]
                    
                    for target_val in target_values:
                        target_data = cluster_data[cluster_data[target_param] == target_val]
                        if not target_data.empty:
                            value = target_data[event].iloc[0]
                            if not pd.isna(value):
                                target_idx = target_values.index(target_val)
                                offset = (target_idx - len(target_values)/2 + 0.5) * bar_width
                                x_positions.append(cluster_idx + offset)
                                line_data.append(value)
                
                # Plot line with markers
                if line_data:
                    ax2.plot(x_positions, line_data, 'o-', 
                            label=f'{event}', 
                            color=line_colors[event_idx], 
                            linewidth=2, 
                            markersize=6,
                            alpha=0.8)
    
    # Plot Kernel Time as bars on right y-axis (or main axis if no events)
    kernel_ax = ax1 if not events else ax1  # Always use ax1 for bars for better visibility
    
    kernel_data = grouped_data.pivot(index='cluster_label', columns=target_param, values='Kernel Time (s)')
    
    for i, target_val in enumerate(target_values):
        if target_val in kernel_data.columns:
            values = [kernel_data.loc[cluster, target_val] if cluster in kernel_data.index and not pd.isna(kernel_data.loc[cluster, target_val]) else 0 
                     for cluster in cluster_labels]
            
            offset = (i - len(target_values)/2 + 0.5) * bar_width
            
            bars = kernel_ax.bar(x_pos + offset, values, bar_width, 
                               label=f'Kernel Time ({target_param.split("_")[-1]}={target_val})', 
                               color=bar_colors[i], 
                               alpha=0.7,
                               edgecolor='black',
                               linewidth=0.5)
    
    # Customize axes
    ax1.set_xlabel('Parameter Combinations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Kernel Time (s)', fontsize=12, color='darkblue', fontweight='bold')
    
    if events and ax2 is not None:
        ax2.set_ylabel('Performance Events', fontsize=12, color='darkred', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='darkred')
        
        # Set different scales if needed
        if len(events) > 1:
            # Normalize event scales if they're very different
            event_ranges = []
            for event in events:
                if event in grouped_data.columns:
                    event_data = grouped_data[event].dropna()
                    if not event_data.empty:
                        event_ranges.append(event_data.max() - event_data.min())
            
            if event_ranges and max(event_ranges) > 0:
                # Adjust y-axis to show all events clearly
                ax2.set_ylim(bottom=0)
    
    ax1.tick_params(axis='y', labelcolor='darkblue')
    
    # Set x-axis
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cluster_labels, rotation=45, ha='right', fontsize=10)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    
    if events and ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Combine legends
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
    else:
        all_lines = lines1
        all_labels = labels1
    
    # Position legend outside the plot area
    ax1.legend(all_lines, all_labels, 
              loc='center left', 
              bbox_to_anchor=(1.15, 0.5),
              fontsize=10)
    
    # Add grid
    ax1.grid(True, alpha=0.3, axis='y')
    if events and ax2 is not None:
        ax2.grid(True, alpha=0.2, axis='y', linestyle='--')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Save plot
    target_name = target_param.split('_')[-1]  # Get just the parameter name (M, N, K)
    output_path = Path(output_dir) / f'{target_name}_impact_combined.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_path}")
    
    plt.show()

def create_parameter_combinations(df, block_size_cols):
    """Create all possible combinations of block size parameters for analysis"""
    combinations_data = []
    
    for target_param in block_size_cols:
        other_params = [col for col in block_size_cols if col != target_param]
        combinations_data.append({
            'target_param': target_param,
            'other_params': other_params
        })
    
    return combinations_data

def generate_summary_statistics(df, block_size_cols, events):
    """Generate summary statistics for all parameters"""
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    all_metrics = events + ['Kernel Time (s)']
    
    for param in block_size_cols:
        print(f"\n{param} Statistics:")
        print("-" * 30)
        
        if param in df.columns:
            param_stats = df.groupby(param)[all_metrics].agg(['mean', 'std', 'min', 'max'])
            print(param_stats.round(4))

def main():
    args = parse_arguments()
    
    # Parse events
    events = [event.strip() for event in args.events.split(',')]
    
    # Load data
    df = load_and_validate_data(args.csv)
    
    # Find BLOCK_SIZE parameters
    block_size_cols = find_block_size_parameters(df)
    
    if not block_size_cols:
        print("No BLOCK_SIZE parameters found in the data!")
        return
    
    # Validate events
    events = validate_events(df, events)
    print(f"Will plot events: {events}")
    
    # Create parameter combinations for analysis
    combinations = create_parameter_combinations(df, block_size_cols)
    
    # Generate plots for each parameter
    for combo in combinations:
        target_param = combo['target_param']
        other_params = combo['other_params']
        
        print(f"\nGenerating combined line-bar chart for {target_param}")
        print(f"Clusters represent combinations of: {[p.split('_')[-1] for p in other_params]}")
        
        # Create combined chart with lines for events and bars for Kernel Time
        plot_combined_line_bar_chart(df, target_param, other_params, events, 
                                    args.output_dir, args.figsize)
    
    # Generate summary statistics
    generate_summary_statistics(df, block_size_cols, events)
    
    print(f"\nAll plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()