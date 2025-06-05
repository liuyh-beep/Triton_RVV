import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import argparse
import glob
from typing import Dict, Any
'''
python3 plot_making.py --kernel-dir matmul
'''
def load_csv_data(csv_file: str) -> pd.DataFrame:
    """Load data from CSV file into a pandas DataFrame"""
    try:
        df = pd.read_csv(csv_file)
        # Convert "Valid Output" column to boolean for easier filtering
        if 'Valid Output' in df.columns:
            df['Valid Output'] = df['Valid Output'] == 'YES'
        # Filter out invalid outputs
        if 'Valid Output' in df.columns:
            valid_df = df[df['Valid Output']]
            if len(valid_df) == 0:
                print(f"Warning: No valid outputs found in {csv_file}")
                return df
            return valid_df
        return df
    except Exception as e:
        print(f"Error loading CSV file {csv_file}: {e}")
        return pd.DataFrame()

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare and clean data for visualization"""
    if df.empty:
        return df
    
    # Create a combined BlockSize column for better labeling
    df['BlockConfig'] = df.apply(
        lambda row: f"{int(row['BlockSize M'])}x{int(row['BlockSize N'])}x{int(row['BlockSize K'])}", 
        axis=1
    )
    
    # Calculate total block size for sorting
    df['TotalSize'] = df['BlockSize M'] * df['BlockSize N'] * df['BlockSize K']
    
    # Sort first by total size, then by individual dimensions
    df = df.sort_values(by=['TotalSize', 'BlockSize M', 'BlockSize N', 'BlockSize K'])
    
    # Remove any N/A or string values from numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def generate_plots(df: pd.DataFrame, output_dir: str, plot_type: str) -> None:
    """Generate plots based on the data and plot type"""
    if df.empty:
        print(f"No {plot_type} data to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {plot_type} visualizations in {output_dir}...")
    
    # Define the metrics and plots for each plot type
    plot_configs = {
        'micro': {
            'key_metrics': ['u_mode_cycle', 'eu_vfpu_full', 'vidu_total_cycle', 'vector_micro_op'],
            'plots': [
                {'type': 'bar', 'x': 'BlockConfig', 'y': 'u_mode_cycle', 
                 'title': 'User Mode Cycles by Block Configuration',
                 'filename': 'user_mode_cycles.png'},
                {'type': 'stacked_bar', 
                 'data': {'y1': 'vidu_total_cycle', 'y2': 'u_mode_cycle', 'stack_label': 'other_cycles'},
                 'title': 'Cycle Distribution by Block Configuration',
                 'filename': 'cycle_distribution.png',
                 'colors': ['#ff9999','#66b3ff'],
                 'legend': ['Vector Instruction Unit Cycles', 'Other Cycles']},
                {'type': 'scatter', 'x': 'vector_micro_op', 'y': 'Kernel Time (s)', 
                 'size': 'u_mode_cycle', 'hue': 'BlockConfig',
                 'title': 'Kernel Time vs. Vector Micro Operations',
                 'filename': 'kernel_time_vs_micro_ops.png'},
                {'type': 'heatmap', 'metrics': ['u_mode_cycle', 'eu_vfpu_full', 'vidu_total_cycle', 'vector_micro_op'],
                 'title': 'Normalized Micro-Architecture Metrics by Block Configuration',
                 'filename': 'micro_metrics_heatmap.png'}
            ]
        },
        'instructions': {
            'key_metrics': ['vector_inst', 'vector_load_inst', 'vector_store_inst'],
            'plots': [
                {'type': 'bar', 'x': 'BlockConfig', 'y': 'vector_inst', 
                 'title': 'Total Vector Instructions by Block Configuration',
                 'filename': 'total_vector_instructions.png'},
                {'type': 'stacked_bar', 
                 'data': {'y1': 'vector_load_inst', 'y2': 'vector_store_inst', 
                          'y3': 'vector_inst', 'compute_label': 'other_vector_inst'},
                 'title': 'Vector Instruction Distribution by Block Configuration',
                 'filename': 'instruction_distribution.png',
                 'colors': ['#ff9999','#66b3ff','#99ff99'],
                 'legend': ['Load Instructions', 'Store Instructions', 'Compute Instructions']},
                {'type': 'scatter', 'x': 'vector_inst', 'y': 'Kernel Time (s)', 
                 'size': 'load_store_ratio', 'hue': 'BlockConfig',
                 'title': 'Kernel Time vs. Total Vector Instructions',
                 'filename': 'kernel_time_vs_instructions.png',
                 'prep_func': lambda df: df.assign(load_store_ratio=df['vector_load_inst'] / (df['vector_store_inst'] + 1))}
                # {'type': 'pie', 
                #  'values': ['vector_load_inst', 'vector_store_inst', 'compute_inst'],
                #  'labels': ['Load', 'Store', 'Compute'],
                #  'title': 'Instruction Mix for {block_config}',
                #  'filename': 'instruction_mix_{block_config}.png',
                #  'prep_func': lambda df: df.assign(compute_inst=df['vector_inst'] - df['vector_load_inst'] - df['vector_store_inst'])}
            ]
        },
        'cache': {
            'key_metrics': ['l1d_load_miss', 'l1d_load_access', 'l1d_access', 'l1d_miss', 
                          'l2_access', 'l2_miss', 'l2_load_access', 'l2_store_access'],
            'plots': [
                # {'type': 'bar_with_hue', 
                #  'data': {'id_vars': 'BlockConfig', 
                #           'value_vars': ['l1d_miss_rate', 'l2_miss_rate'],
                #           'var_name': 'Cache Level', 'value_name': 'Miss Rate (%)'},
                #  'title': 'Cache Miss Rates by Block Configuration',
                #  'filename': 'cache_miss_rates.png',
                #  'legend_labels': ['L1 Data Cache', 'L2 Cache'],
                #  'prep_func': lambda df: df.assign(
                #      l1d_miss_rate=df['l1d_miss'] / df['l1d_access'] * 100,
                #      l2_miss_rate=df['l2_miss'] / df['l2_access'] * 100)},
                {'type': 'stacked_bar', 
                 'data': {'y1': 'l1d_load_access', 'store_access': 'l1d_store_access'},
                 'title': 'L1 Data Cache Access Distribution by Block Configuration',
                 'filename': 'l1_access_distribution.png',
                 'colors': ['#ff9999','#66b3ff'],
                 'legend': ['Load Accesses', 'Store Accesses'],
                 'prep_func': lambda df: df.assign(
                     l1d_store_access=df['l1d_access'] - df['l1d_load_access'])},
                {'type': 'scatter', 'x': 'l1d_miss_rate', 'y': 'Kernel Time (s)', 
                 'size': 'l2_miss_rate', 'hue': 'BlockConfig',
                 'title': 'Kernel Time vs. L1 Data Cache Miss Rate',
                 'filename': 'kernel_time_vs_miss_rate.png',
                 'prep_func': lambda df: df.assign(
                     l1d_miss_rate=df['l1d_miss'] / df['l1d_access'] * 100,
                     l2_miss_rate=df['l2_miss'] / df['l2_access'] * 100)},
                {'type': 'line', 
                 'metrics': ['l1d_access', 'l1d_miss', 'l2_access', 'l2_miss'],
                 'title': 'Memory Hierarchy Access Pattern by Block Configuration',
                 'filename': 'memory_hierarchy_pattern.png',
                 'log_scale': True},
                {'type': 'heatmap', 
                 'metrics': ['l1d_load_miss', 'l1d_load_access', 'l1d_miss', 'l1d_access', 
                           'l2_miss', 'l2_access', 'l2_load_access', 'l2_store_access'],
                 'title': 'Normalized Cache Metrics by Block Configuration',
                 'filename': 'cache_metrics_heatmap.png'}
            ]
        }
    }
    
    # Check if we have a plot configuration for this type
    if plot_type not in plot_configs:
        print(f"No plot configuration for type: {plot_type}")
        return
    
    # Check if we have the required metrics
    config = plot_configs[plot_type]
    for metric in config['key_metrics']:
        if metric not in df.columns:
            print(f"Missing required metric for {plot_type}: {metric}")
            return
    
    # Create the plots
    for plot_config in config['plots']:
        try:
            create_plot(df, output_dir, plot_config)
        except Exception as e:
            print(f"Error creating plot {plot_config.get('filename', 'unknown')}: {e}")

def create_plot(df: pd.DataFrame, output_dir: str, plot_config: Dict[str, Any]) -> None:
    """Create a specific plot based on the configuration"""
    plot_type = plot_config.get('type', '')
    
    # Prepare data if needed
    prep_func = plot_config.get('prep_func')
    if prep_func:
        df = prep_func(df)
    
    # Create the plot based on its type
    if plot_type == 'bar':
        plt.figure(figsize=(12, 7))
        sns.barplot(x=plot_config['x'], y=plot_config['y'], data=df)
        plt.title(plot_config['title'])
        plt.xlabel('Block Configuration (MxNxK)')
        plt.ylabel(plot_config['y'].replace('_', ' ').title())
        plt.xticks(rotation=45, fontsize=8)  # Reduced font size
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_config['filename']))
        plt.close()
    
    elif plot_type == 'bar_with_hue':
        plt.figure(figsize=(12, 7))
        # Prepare data for bar with hue
        data_config = plot_config['data']
        melted_df = df.melt(id_vars=data_config['id_vars'], 
                           value_vars=data_config['value_vars'],
                           var_name=data_config['var_name'], 
                           value_name=data_config['value_name'])
        
        sns.barplot(x=data_config['id_vars'], y=data_config['value_name'], 
                    hue=data_config['var_name'], data=melted_df)
        plt.title(plot_config['title'])
        plt.xlabel('Block Configuration (MxNxK)')
        plt.ylabel(data_config['value_name'])
        plt.xticks(rotation=45, fontsize=8)  # Reduced font size
        if 'legend_labels' in plot_config:
            plt.legend(title=data_config['var_name'], labels=plot_config['legend_labels'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_config['filename']))
        plt.close()
    
    elif plot_type == 'stacked_bar':
        plt.figure(figsize=(12, 7))
        data_config = plot_config['data']
        
        # Create DataFrame for stacked bar
        df_stack = df[['BlockConfig']].copy()
        
        # Add first data series
        if 'y1' in data_config:
            df_stack[data_config['y1']] = df[data_config['y1']]
        
        # Add second data series
        if 'y2' in data_config and 'stack_label' in data_config:
            # For simple case with two variables where one is derived from the other
            df_stack[data_config['stack_label']] = df[data_config['y2']] - df[data_config['y1']]
        elif 'y2' in data_config:
            # For explicitly provided second variable
            df_stack[data_config['y2']] = df[data_config['y2']]
        
        # Add third data series if present
        if 'y3' in data_config and 'compute_label' in data_config:
            # For cases like compute instructions derived from total - (load+store)
            df_stack[data_config['compute_label']] = df[data_config['y3']] - df[data_config['y1']] - df[data_config['y2']]
        
        # For special cases like store accesses
        if 'store_access' in data_config:
            df_stack[data_config['store_access']] = df[data_config['store_access']]
        
        # Set index and determine columns to plot
        df_stack = df_stack.set_index('BlockConfig')
        plot_columns = [col for col in df_stack.columns if col != 'BlockConfig']
        
        df_stack[plot_columns].plot(kind='bar', stacked=True, 
                                    color=plot_config.get('colors', None))
        plt.title(plot_config['title'])
        plt.xlabel('Block Configuration (MxNxK)')
        plt.ylabel('Count')
        if 'legend' in plot_config:
            plt.legend(plot_config['legend'])
        plt.xticks(rotation=45, fontsize=5)  # Reduced font size
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_config['filename']))
        plt.close()
    
    elif plot_type == 'scatter':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=plot_config['x'], y=plot_config['y'], 
                        size=plot_config['size'], hue=plot_config['hue'], data=df)
        plt.title(plot_config['title'])
        plt.xlabel(plot_config['x'].replace('_', ' ').title())
        plt.ylabel(plot_config['y'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_config['filename']))
        plt.close()
    
    elif plot_type == 'line':
        plt.figure(figsize=(12, 7))
        # Prepare data for line plot
        metrics = plot_config['metrics']
        line_df = df[['BlockConfig'] + metrics].copy()
        line_df = line_df.set_index('BlockConfig')
        line_df.plot(kind='line', marker='o')
        plt.title(plot_config['title'])
        plt.xlabel('Block Configuration (MxNxK)')
        plt.ylabel('Count')
        if plot_config.get('log_scale', False):
            plt.yscale('log')
            plt.ylabel('Count (log scale)')
        plt.grid(True, which="both", ls="--")
        plt.xticks(rotation=45, fontsize=8)  # Reduced font size
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_config['filename']))
        plt.close()
    
    elif plot_type == 'heatmap':
        # Prepare data for heatmap
        metrics = plot_config['metrics']
        if all(metric in df.columns for metric in metrics):
            # Normalize the data for better visualization
            norm_df = df[metrics].copy()
            for col in norm_df.columns:
                if norm_df[col].max() > 0:
                    norm_df[col] = norm_df[col] / norm_df[col].max()
            
            # Create pivot table with BlockConfig as index
            pivot_df = pd.DataFrame()
            for i, row in df.iterrows():
                for metric in metrics:
                    pivot_df.loc[row['BlockConfig'], metric] = norm_df.loc[i, metric]
            
            plt.figure(figsize=(14, 10))
            sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f')
            plt.title(plot_config['title'])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, plot_config['filename']))
            plt.close()
    
    # elif plot_type == 'pie':
    #     # Create a directory for pie charts
    #     pie_dir = os.path.join(output_dir, 'instruction_mix')
    #     os.makedirs(pie_dir, exist_ok=True)
        
    #     for i, row in df.iterrows():
    #         block_config = row['BlockConfig']
    #         values = [row[val] for val in plot_config['values']]
    #         labels = plot_config['labels']
            
    #         # Skip if all values are zero
    #         if sum(values) == 0:
    #             continue
            
    #         plt.figure(figsize=(8, 8))
    #         plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
    #                 colors=['#ff9999','#66b3ff','#99ff99'])
    #         plt.axis('equal')
    #         plt.title(plot_config['title'].format(block_config=block_config))
    #         plt.tight_layout()
    #         filename = plot_config['filename'].format(block_config=block_config)
    #         plt.savefig(os.path.join(pie_dir, filename))
    #         plt.close()

def create_dual_axis_charts(micro_df: pd.DataFrame, inst_df: pd.DataFrame, 
                           cache_df: pd.DataFrame, output_dir: str) -> None:
    """Create dual axis charts showing kernel time vs. various metrics"""
    print("Generating dual-axis relationship charts...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to plot against kernel time
    relationships = []
    
    # Create TotalSize column and sort both dataframes consistently
    if not inst_df.empty and 'Kernel Time (s)' in inst_df.columns:
        # Ensure proper sorting of block configurations
        inst_df['TotalSize'] = inst_df['BlockSize M'] * inst_df['BlockSize N'] * inst_df['BlockSize K']
        inst_df = inst_df.sort_values(by=['TotalSize', 'BlockSize M', 'BlockSize N', 'BlockSize K'])
        
        for metric in ['vector_inst', 'vector_load_inst', 'vector_store_inst']:
            if metric in inst_df.columns:
                relationships.append({
                    'df': inst_df,
                    'metric': metric,
                    'title': f'Kernel Time vs. {metric.replace("_", " ").title()}',
                    'filename': f'kernel_time_vs_{metric}.png',
                    'color': '#ff9999' if 'load' in metric else '#66b3ff' if 'store' in metric else '#99ff99'
                })
    
    if not cache_df.empty and 'Kernel Time (s)' in cache_df.columns:
        # Ensure proper sorting of block configurations
        cache_df['TotalSize'] = cache_df['BlockSize M'] * cache_df['BlockSize N'] * cache_df['BlockSize K']
        cache_df = cache_df.sort_values(by=['TotalSize', 'BlockSize M', 'BlockSize N', 'BlockSize K'])
        
        for metric in ['l1d_access', 'l1d_miss', 'l2_access', 'l2_miss']:
            if metric in cache_df.columns:
                relationships.append({
                    'df': cache_df,
                    'metric': metric,
                    'title': f'Kernel Time vs. {metric.replace("_", " ").title()}',
                    'filename': f'kernel_time_vs_{metric}.png',
                    'color': '#ff9999' if 'l1' in metric else '#66b3ff'
                })
    
    # Create combined chart with all metrics
    if relationships:
        # Create a consistently sorted combined chart
        # We'll use the first available dataframe as the base for our block configurations
        base_df = None
        if not inst_df.empty and 'BlockConfig' in inst_df.columns:
            base_df = inst_df.copy()
        elif not cache_df.empty and 'BlockConfig' in cache_df.columns:
            base_df = cache_df.copy()
        
        if base_df is not None:
            # Sort by total block size and individual dimensions
            base_df['TotalSize'] = base_df['BlockSize M'] * base_df['BlockSize N'] * base_df['BlockSize K']
            base_df = base_df.sort_values(by=['TotalSize', 'BlockSize M', 'BlockSize N', 'BlockSize K'])
            
            plt.figure(figsize=(14, 8))
            
            # Create the bar chart for kernel time
            ax1 = plt.gca()
            bars = ax1.bar(base_df['BlockConfig'], base_df['Kernel Time (s)'], 
                          color='skyblue', alpha=0.7, label='Kernel Time (s)')
            ax1.set_ylabel('Kernel Time (s)', fontsize=12)
            ax1.set_xlabel('Block Configuration (MxNxK)', fontsize=12)
            ax1.tick_params(axis='x', rotation=45, labelsize=8)  # Reduced font size
            
            # Find best and worst kernel time
            best_idx = base_df['Kernel Time (s)'].idxmin()
            worst_idx = base_df['Kernel Time (s)'].idxmax()
            best_config = base_df.loc[best_idx, 'BlockConfig']
            worst_config = base_df.loc[worst_idx, 'BlockConfig']
            best_time = base_df.loc[best_idx, 'Kernel Time (s)']
            worst_time = base_df.loc[worst_idx, 'Kernel Time (s)']
            
            # Get the x positions for these configs
            x_positions = np.arange(len(base_df))
            best_x_pos = np.where(base_df['BlockConfig'] == best_config)[0][0]
            worst_x_pos = np.where(base_df['BlockConfig'] == worst_config)[0][0]
            
            # Annotate best and worst
            ax1.annotate(f'Best: {best_time:.4f}s', 
                         xy=(best_x_pos, best_time),
                         xytext=(best_x_pos, best_time - 0.02),
                         textcoords='data',
                         arrowprops=dict(facecolor='green', shrink=0.05),
                         horizontalalignment='center',
                         verticalalignment='top',
                         fontweight='bold',
                         color='green')
                         
            ax1.annotate(f'Worst: {worst_time:.4f}s', 
                         xy=(worst_x_pos, worst_time),
                         xytext=(worst_x_pos, worst_time + 0.02),
                         textcoords='data',
                         arrowprops=dict(facecolor='red', shrink=0.05),
                         horizontalalignment='center',
                         verticalalignment='bottom',
                         fontweight='bold',
                         color='red')
            
            # Create secondary y-axis for metrics
            ax2 = ax1.twinx()
            
            # Plot each metric as a line using the same block config ordering
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
            for i, rel in enumerate(relationships):
                source_df = rel['df']
                metric = rel['metric']
                
                # Ensure we map metrics to the same block configuration order
                metric_values = []
                for block_config in base_df['BlockConfig']:
                    # Find the matching row in the source dataframe
                    matching_row = source_df[source_df['BlockConfig'] == block_config]
                    if not matching_row.empty:
                        metric_values.append(matching_row[metric].values[0])
                    else:
                        # Use NaN for missing values
                        metric_values.append(np.nan)
                
                color_idx = i % len(colors)
                ax2.plot(base_df['BlockConfig'], metric_values, 
                        marker='o', linestyle='-', color=colors[color_idx], 
                        label=rel['metric'].replace('_', ' ').title())
            
            # Add legends
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            plt.title('Kernel Time vs. Performance Metrics', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'kernel_time_vs_all_metrics.png'))
            plt.close()
            
            # Now create individual charts for each metric
            for rel in relationships:
                source_df = rel['df']
                metric = rel['metric']
                
                # Create a new figure
                plt.figure(figsize=(12, 7))
                
                # Create the bar chart for kernel time using the same ordering
                ax1 = plt.gca()
                bars = ax1.bar(base_df['BlockConfig'], base_df['Kernel Time (s)'], 
                              color='skyblue', alpha=0.7, label='Kernel Time (s)')
                ax1.set_ylabel('Kernel Time (s)', fontsize=12)
                ax1.set_xlabel('Block Configuration (MxNxK)', fontsize=12)
                ax1.tick_params(axis='x', rotation=45, labelsize=8)  # Reduced font size
                
                # Find best and worst kernel time for this specific dataset
                best_idx = base_df['Kernel Time (s)'].idxmin()
                worst_idx = base_df['Kernel Time (s)'].idxmax()
                best_config = base_df.loc[best_idx, 'BlockConfig']
                worst_config = base_df.loc[worst_idx, 'BlockConfig']
                best_time = base_df.loc[best_idx, 'Kernel Time (s)']
                worst_time = base_df.loc[worst_idx, 'Kernel Time (s)']
                
                # Map metric values to the same block configuration order
                metric_values = []
                for block_config in base_df['BlockConfig']:
                    # Find the matching row in the source dataframe
                    matching_row = source_df[source_df['BlockConfig'] == block_config]
                    if not matching_row.empty:
                        metric_values.append(matching_row[metric].values[0])
                    else:
                        # Use NaN for missing values
                        metric_values.append(np.nan)
                
                # Find best and worst metric values
                metric_array = np.array(metric_values)
                valid_indices = ~np.isnan(metric_array)
                
                if np.any(valid_indices):
                    metric_best_value = np.nanmin(metric_array)
                    metric_worst_value = np.nanmax(metric_array)
                    
                    metric_best_idx = np.nanargmin(metric_array)
                    metric_worst_idx = np.nanargmax(metric_array)
                    
                    metric_best_config = base_df['BlockConfig'].iloc[metric_best_idx]
                    metric_worst_config = base_df['BlockConfig'].iloc[metric_worst_idx]
                
                # Get the x positions for these configs
                best_x_pos = np.where(base_df['BlockConfig'] == best_config)[0][0]
                worst_x_pos = np.where(base_df['BlockConfig'] == worst_config)[0][0]
                
                # Annotate best and worst kernel times
                ax1.annotate(f'Best Time: {best_time:.4f}s', 
                             xy=(best_x_pos, best_time),
                             xytext=(best_x_pos, best_time - 0.02),
                             textcoords='data',
                             arrowprops=dict(facecolor='green', shrink=0.05),
                             horizontalalignment='center',
                             verticalalignment='top',
                             fontweight='bold',
                             color='green')
                             
                ax1.annotate(f'Worst Time: {worst_time:.4f}s', 
                             xy=(worst_x_pos, worst_time),
                             xytext=(worst_x_pos, worst_time + 0.02),
                             textcoords='data',
                             arrowprops=dict(facecolor='red', shrink=0.05),
                             horizontalalignment='center',
                             verticalalignment='bottom',
                             fontweight='bold',
                             color='red')
                
                # Create secondary y-axis for the metric
                ax2 = ax1.twinx()
                metric_name = metric.replace('_', ' ').title()
                line = ax2.plot(base_df['BlockConfig'], metric_values, 
                              marker='o', linestyle='-', color=rel['color'], 
                              label=metric_name)
                ax2.set_ylabel(metric_name, fontsize=12)
                
                # Annotate best and worst metric values if we have valid data
                if np.any(valid_indices):
                    # Only if they're different from the time best/worst
                    if metric_best_config != best_config:
                        metric_best_x_pos = np.where(base_df['BlockConfig'] == metric_best_config)[0][0]
                        ax2.annotate(f'Min {metric_name}: {metric_best_value:,.0f}', 
                                     xy=(metric_best_x_pos, metric_best_value),
                                     xytext=(metric_best_x_pos, metric_best_value * 0.9 if metric_best_value > 0 else metric_best_value * 1.1),
                                     textcoords='data',
                                     arrowprops=dict(facecolor=rel['color'], alpha=0.6, shrink=0.05),
                                     horizontalalignment='center',
                                     verticalalignment='top',
                                     fontweight='bold',
                                     color=rel['color'])
                                     
                    if metric_worst_config != worst_config:
                        metric_worst_x_pos = np.where(base_df['BlockConfig'] == metric_worst_config)[0][0]
                        ax2.annotate(f'Max {metric_name}: {metric_worst_value:,.0f}', 
                                     xy=(metric_worst_x_pos, metric_worst_value),
                                     xytext=(metric_worst_x_pos, metric_worst_value * 1.1 if metric_worst_value > 0 else metric_worst_value * 0.9),
                                     textcoords='data',
                                     arrowprops=dict(facecolor=rel['color'], alpha=0.6, shrink=0.05),
                                     horizontalalignment='center',
                                     verticalalignment='bottom',
                                     fontweight='bold',
                                     color=rel['color'])
                
                # Add legends
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                
                plt.title(rel['title'], fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, rel['filename']))
                plt.close()

def create_summary_report(micro_df: pd.DataFrame, inst_df: pd.DataFrame, 
                         cache_df: pd.DataFrame, output_dir: str) -> None:
    """Create a summary report with the best configurations"""
    print("Generating summary report...")
    os.makedirs(output_dir, exist_ok=True)
    
    summary = []
    
    # Get best configurations based on different metrics
    if not micro_df.empty and 'Kernel Time (s)' in micro_df.columns:
        best_time = micro_df.loc[micro_df['Kernel Time (s)'].idxmin()]
        summary.append({
            'Metric': 'Fastest Kernel Time', 
            'Value': f"{best_time['Kernel Time (s)']:.6f} s",
            'Configuration': best_time['BlockConfig'],
            'Details': f"u_mode_cycle: {int(best_time['u_mode_cycle']):,}"
        })
    
    if not micro_df.empty and 'u_mode_cycle' in micro_df.columns:
        best_cycles = micro_df.loc[micro_df['u_mode_cycle'].idxmin()]
        summary.append({
            'Metric': 'Fewest User Mode Cycles', 
            'Value': f"{int(best_cycles['u_mode_cycle']):,}",
            'Configuration': best_cycles['BlockConfig'],
            'Details': f"Kernel Time: {best_cycles['Kernel Time (s)']:.6f} s"
        })
    
    if not inst_df.empty and 'vector_inst' in inst_df.columns:
        best_inst = inst_df.loc[inst_df['vector_inst'].idxmin()]
        summary.append({
            'Metric': 'Fewest Vector Instructions', 
            'Value': f"{int(best_inst['vector_inst']):,}",
            'Configuration': best_inst['BlockConfig'],
            'Details': f"Load: {int(best_inst['vector_load_inst']):,}, Store: {int(best_inst['vector_store_inst']):,}"
        })
    
    if not cache_df.empty and 'l1d_miss' in cache_df.columns and 'l1d_access' in cache_df.columns:
        cache_df['l1d_miss_rate'] = cache_df['l1d_miss'] / cache_df['l1d_access'] * 100
        best_l1_miss = cache_df.loc[cache_df['l1d_miss_rate'].idxmin()]
        summary.append({
            'Metric': 'Lowest L1D Cache Miss Rate', 
            'Value': f"{best_l1_miss['l1d_miss_rate']:.2f}%",
            'Configuration': best_l1_miss['BlockConfig'],
            'Details': f"Misses: {int(best_l1_miss['l1d_miss']):,}, Accesses: {int(best_l1_miss['l1d_access']):,}"
        })
    
    if not cache_df.empty and 'l2_miss' in cache_df.columns and 'l2_access' in cache_df.columns:
        cache_df['l2_miss_rate'] = cache_df['l2_miss'] / cache_df['l2_access'] * 100
        best_l2_miss = cache_df.loc[cache_df['l2_miss_rate'].idxmin()]
        summary.append({
            'Metric': 'Lowest L2 Cache Miss Rate', 
            'Value': f"{best_l2_miss['l2_miss_rate']:.2f}%",
            'Configuration': best_l2_miss['BlockConfig'],
            'Details': f"Misses: {int(best_l2_miss['l2_miss']):,}, Accesses: {int(best_l2_miss['l2_access']):,}"
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary)
    
    # Save summary to CSV
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(output_dir, 'best_configurations.csv'), index=False)
    
    # Create an overall performance comparison chart
    if not micro_df.empty and 'Kernel Time (s)' in micro_df.columns:
        plt.figure(figsize=(12, 7))
        sns.barplot(x='BlockConfig', y='Kernel Time (s)', data=micro_df)
        plt.title('Kernel Execution Time by Block Configuration')
        plt.xlabel('Block Configuration (MxNxK)')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, fontsize=8)  # Reduced font size
        # Highlight the best configuration
        if not summary_df.empty and 'Fastest Kernel Time' in summary_df['Metric'].values:
            best_config = summary_df.loc[summary_df['Metric'] == 'Fastest Kernel Time', 'Configuration'].values[0]
            best_idx = micro_df.index[micro_df['BlockConfig'] == best_config].tolist()[0]
            best_value = micro_df.loc[best_idx, 'Kernel Time (s)']
            plt.annotate(f'Best: {best_value:.4f}s', 
                         xy=(best_idx, best_value),
                         xytext=(best_idx, best_value + 0.05),
                         arrowprops=dict(facecolor='red', shrink=0.05),
                         horizontalalignment='center',
                         color='red')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_performance.png'))
        plt.close()

def find_csv_files(kernel_dir: str) -> Dict[str, str]:
    """Find CSV files in the kernel directory"""
    csv_files = {}
    
    # Define patterns to look for
    patterns = {
        'micro': ['micro', 'perf_stats_micro'],
        'instructions': ['ins', 'instructions', 'perf_stats_ins'],
        'cache': ['cache', 'perf_stats_cache']
    }
    
    # Find all CSV files in the kernel directory
    all_csvs = glob.glob(f"{kernel_dir}/perf/*.csv", recursive=True)
    
    # Categorize each CSV file
    for csv_file in all_csvs:
        print(csv_file)
        filename = os.path.basename(csv_file).lower()
        for category, keywords in patterns.items():
            if any(keyword in filename for keyword in keywords):
                csv_files[category] = csv_file
                break
    
    return csv_files

def create_cache_miss_rate_chart(cache_df: pd.DataFrame, output_dir: str) -> None:
    """Create a graph showing kernel time vs L1/L2 miss rates"""
    if cache_df.empty or 'Kernel Time (s)' not in cache_df.columns:
        print("Cannot create cache miss rate chart: missing required data")
        return
        
    # Calculate miss rates if not already present
    if 'l1d_miss_rate' not in cache_df.columns and 'l1d_miss' in cache_df.columns and 'l1d_access' in cache_df.columns:
        cache_df['l1d_miss_rate'] = cache_df['l1d_miss'] / cache_df['l1d_access'] * 100
        
    if 'l2_miss_rate' not in cache_df.columns and 'l2_miss' in cache_df.columns and 'l2_access' in cache_df.columns:
        cache_df['l2_miss_rate'] = cache_df['l2_miss'] / cache_df['l2_access'] * 100
    
    # Check if we have the miss rates
    if 'l1d_miss_rate' not in cache_df.columns or 'l2_miss_rate' not in cache_df.columns:
        print("Cannot create cache miss rate chart: missing miss rate data")
        return
    
    # Sort by block config for consistent ordering
    df = cache_df.sort_values('BlockConfig')
    
    plt.figure(figsize=(14, 8))
    
    # Create the bar chart for kernel time
    ax1 = plt.gca()
    bars = ax1.bar(df['BlockConfig'], df['Kernel Time (s)'], 
                  color='skyblue', alpha=0.7, label='Kernel Time (s)')
    ax1.set_ylabel('Kernel Time (s)', fontsize=12)
    ax1.set_xlabel('Block Configuration (MxNxK)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)  # Reduced font size
    
    # Find best and worst kernel time
    best_idx = df['Kernel Time (s)'].idxmin()
    worst_idx = df['Kernel Time (s)'].idxmax()
    best_config = df.loc[best_idx, 'BlockConfig']
    worst_config = df.loc[worst_idx, 'BlockConfig']
    best_time = df.loc[best_idx, 'Kernel Time (s)']
    worst_time = df.loc[worst_idx, 'Kernel Time (s)']
    
    # Get the x positions for these configs
    x_positions = np.arange(len(df))
    best_x_pos = x_positions[df['BlockConfig'] == best_config][0]
    worst_x_pos = x_positions[df['BlockConfig'] == worst_config][0]
    
    # Annotate best and worst
    ax1.annotate(f'Best: {best_time:.4f}s', 
                 xy=(best_x_pos, best_time),
                 xytext=(best_x_pos, best_time - 0.02),
                 textcoords='data',
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 horizontalalignment='center',
                 verticalalignment='top',
                 fontweight='bold',
                 color='green')
                 
    ax1.annotate(f'Worst: {worst_time:.4f}s', 
                 xy=(worst_x_pos, worst_time),
                 xytext=(worst_x_pos, worst_time + 0.02),
                 textcoords='data',
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 fontweight='bold',
                 color='red')
    
    # Create secondary y-axis for miss rates
    ax2 = ax1.twinx()
    
    # Plot L1 and L2 miss rates
    l1_line = ax2.plot(df['BlockConfig'], df['l1d_miss_rate'], 
                      marker='o', linestyle='-', color='#ff9999', 
                      label='L1 Data Cache Miss Rate (%)')
    l2_line = ax2.plot(df['BlockConfig'], df['l2_miss_rate'], 
                      marker='s', linestyle='--', color='#66b3ff', 
                      label='L2 Cache Miss Rate (%)')
    
    ax2.set_ylabel('Miss Rate (%)', fontsize=12)
    
    # Find best and worst miss rates
    l1_best_idx = df['l1d_miss_rate'].idxmin()
    l1_worst_idx = df['l1d_miss_rate'].idxmax()
    l2_best_idx = df['l2_miss_rate'].idxmin()
    l2_worst_idx = df['l2_miss_rate'].idxmax()
    
    # L1 miss rate annotations
    l1_best_config = df.loc[l1_best_idx, 'BlockConfig']
    l1_best_rate = df.loc[l1_best_idx, 'l1d_miss_rate']
    if l1_best_config != best_config:  # Only if different from kernel time best
        l1_best_x_pos = x_positions[df['BlockConfig'] == l1_best_config][0]
        ax2.annotate(f'Min L1 Miss: {l1_best_rate:.2f}%', 
                     xy=(l1_best_x_pos, l1_best_rate),
                     xytext=(l1_best_x_pos, l1_best_rate * 0.8),
                     textcoords='data',
                     arrowprops=dict(facecolor='#ff9999', alpha=0.6, shrink=0.05),
                     horizontalalignment='center',
                     verticalalignment='top',
                     fontweight='bold',
                     color='#ff9999')
    
    # L2 miss rate annotations
    l2_best_config = df.loc[l2_best_idx, 'BlockConfig']
    l2_best_rate = df.loc[l2_best_idx, 'l2_miss_rate']
    if l2_best_config != best_config and l2_best_config != l1_best_config:  # Avoid overlap
        l2_best_x_pos = x_positions[df['BlockConfig'] == l2_best_config][0]
        ax2.annotate(f'Min L2 Miss: {l2_best_rate:.2f}%', 
                     xy=(l2_best_x_pos, l2_best_rate),
                     xytext=(l2_best_x_pos, l2_best_rate * 0.8),
                     textcoords='data',
                     arrowprops=dict(facecolor='#66b3ff', alpha=0.6, shrink=0.05),
                     horizontalalignment='center',
                     verticalalignment='top',
                     fontweight='bold',
                     color='#66b3ff')
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title('Kernel Time vs. Cache Miss Rates', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cache_miss_rates.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations from performance data CSV files')
    parser.add_argument('--kernel-dir', type=str, default='matmul', 
                        help='Directory containing kernel data (default: matmul)')
    parser.add_argument('--micro', type=str, help='Path to micro-architecture metrics CSV')
    parser.add_argument('--instructions', type=str, help='Path to instruction metrics CSV')
    parser.add_argument('--cache', type=str, help='Path to cache metrics CSV')
    parser.add_argument('--output-dir', type=str, help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Find CSV files if not explicitly provided
    csv_files = {}
    if not (args.micro and args.instructions and args.cache):
        csv_files = find_csv_files(args.kernel_dir)
        
    # Use provided CSV files or defaults
    micro_csv = args.micro or csv_files.get('micro')
    instructions_csv = args.instructions or csv_files.get('instructions')
    cache_csv = args.cache or csv_files.get('cache')
    
    # Set default output directory if not provided
    if not args.output_dir:
        kernel_name = os.path.basename(os.path.abspath(args.kernel_dir))
        args.output_dir = os.path.join(args.kernel_dir, 'visualizations')
    
    print(f"Processing kernel data from: {args.kernel_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    micro_dir = os.path.join(args.output_dir, 'micro')
    inst_dir = os.path.join(args.output_dir, 'instructions')
    cache_dir = os.path.join(args.output_dir, 'cache')
    summary_dir = os.path.join(args.output_dir, 'summary')
    relationship_dir = os.path.join(args.output_dir, 'relationships')
    
    # Load and prepare data
    micro_df = pd.DataFrame()
    inst_df = pd.DataFrame()
    cache_df = pd.DataFrame()
    
    if micro_csv:
        print(f"Loading micro-architecture data from {micro_csv}")
        micro_df = load_csv_data(micro_csv)
        micro_df = prepare_data(micro_df)
        if not micro_df.empty:
            generate_plots(micro_df, micro_dir, 'micro')
    else:
        print("No micro-architecture CSV file found")
    
    if instructions_csv:
        print(f"Loading instruction data from {instructions_csv}")
        inst_df = load_csv_data(instructions_csv)
        inst_df = prepare_data(inst_df)
        if not inst_df.empty:
            generate_plots(inst_df, inst_dir, 'instructions')
    else:
        print("No instructions CSV file found")
    
    if cache_csv:
        print(f"Loading cache data from {cache_csv}")
        cache_df = load_csv_data(cache_csv)
        cache_df = prepare_data(cache_df)
        if not cache_df.empty:
            generate_plots(cache_df, cache_dir, 'cache')
            # Add this line to create the new chart
            create_cache_miss_rate_chart(cache_df, cache_dir)
    else:
        print("No cache CSV file found")
    
    # Create dual axis charts
    create_dual_axis_charts(micro_df, inst_df, cache_df, relationship_dir)
    
    # Create summary report with best configurations
    create_summary_report(micro_df, inst_df, cache_df, summary_dir)
    
    print(f"Visualizations generated in {args.output_dir}")

if __name__ == "__main__":
    main()