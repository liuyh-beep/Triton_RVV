import pandas as pd
from scipy.stats import pearsonr
import os
import argparse

'''This script analyzes performance metric correlations for a given kernel.'''

"""Usage: python3 analysis.py <kernel_name>"""

def load_and_prepare_data(file_path):
    print(f"Reading from: {file_path}")
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    return df


def calculate_correlations(df, time_column='Kernel Time (s)'):
    exclude_cols = ['Timestamp', 'ELF Name', 'BlockSize M', 'BlockSize N', 'BlockSize K', 
                    'Valid Output', 'Total Time (s)']
    
    numeric_cols = []
    for col in df.columns:
        if col not in exclude_cols and col != time_column:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if not df[col].isna().all() and df[col].nunique() > 1:
                    numeric_cols.append(col)
            except:
                pass
    
    print(f"Numeric columns: {numeric_cols}")
    
    correlations = []
    for col in numeric_cols:
        try:
            valid_data = df[[col, time_column]].dropna()
            if len(valid_data) > 1:
                corr, p_value = pearsonr(valid_data[col], valid_data[time_column])
                correlations.append({
                    'Metric': col,
                    'Correlation': corr,
                    'P-Value': p_value
                })
        except Exception as e:
            print(f"Errors during calculating {col}: {e}")
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values(by='Correlation', key=abs, ascending=False)
        return corr_df
    else:
        return pd.DataFrame(columns=['Metric', 'Correlation', 'P-Value'])

def categorize_metrics(metrics):
    """Categorize metrics into cache, instruction, and micro operation metrics."""

    # add new metrics as you need
    cache_metrics = [col for col in metrics if any(x in col.lower() for x in 
                    ['l1d', 'l2', 'cache', 'miss', 'access', 'load_access', 'store_access', 'miss_rate'])]
    
    instruction_metrics = [col for col in metrics if any(x in col.lower() for x in 
                          ['inst', 'instruction', 'vector_load_inst', 'vector_store_inst', 'vector_inst'])]
    
    micro_op_metrics = [col for col in metrics if any(x in col.lower() for x in 
                       ['cycle', 'vfpu', 'vidu', 'micro_op', 'u_mode'])]
    
    instruction_metrics = [x for x in instruction_metrics if x not in cache_metrics]
    micro_op_metrics = [x for x in micro_op_metrics if x not in cache_metrics and x not in instruction_metrics]
    
    return cache_metrics, instruction_metrics, micro_op_metrics


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze performance metric correlations for a kernel')
    parser.add_argument('kernel_name', type=str, help='Name of the kernel to analyze')
    args = parser.parse_args()
    
    kernel_name = args.kernel_name
    
    # Create the data file path based on kernel name
    data_file = f'./{kernel_name}/perf/perf_stats.csv'
    
    try:
        # Check if the file exists
        if not os.path.exists(data_file):
            print(f"Error: File not found: {data_file}")
            print(f"Please make sure the file exists at ./{kernel_name}/perf/perf_stats.csv")
            return
            
        print(f"Analyzing kernel: {kernel_name}")
        print("Reading data...")
        df = load_and_prepare_data(data_file)
        
        print(f"Data loaded successfully with {len(df)} rows")
        
        valid_df = df[df['Valid Output'] == 'YES'].copy()
        if len(valid_df) < len(df):
            print(f"Filtered to {len(valid_df)} valid rows out of {len(df)} total rows")
        
        # Create output directory
        output_dir = f"./{kernel_name}/perf/analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        all_corr = calculate_correlations(valid_df)
        print("\nCorrelation between all metrics and running time:")
        print(all_corr)
        
        
        if not all_corr.empty:
            metrics = all_corr['Metric'].tolist()
            cache_metrics, instruction_metrics, micro_op_metrics = categorize_metrics(metrics)
            
            print(f"\nCache metrics: {cache_metrics}")
            print(f"Instruction metrics: {instruction_metrics}")
            print(f"Micro operation metrics: {micro_op_metrics}")
            
            cache_corr = all_corr[all_corr['Metric'].isin(cache_metrics)] if cache_metrics else pd.DataFrame()
            inst_corr = all_corr[all_corr['Metric'].isin(instruction_metrics)] if instruction_metrics else pd.DataFrame()
            uop_corr = all_corr[all_corr['Metric'].isin(micro_op_metrics)] if micro_op_metrics else pd.DataFrame()
            
            if not cache_corr.empty:
                print("\nCorrelation between cache metrics and running time:")
                print(cache_corr)
            
            if not inst_corr.empty:
                print("\nCorrelation between instruction metrics and running time:")
                print(inst_corr)
            
            if not uop_corr.empty:
                print("\nCorrelation between micro operation metrics and running time:")
                print(uop_corr)
        
        print("\nHigh correlation metrics (|Correlation| > 0.7):")
        high_corr = all_corr[abs(all_corr['Correlation']) > 0.7]
        
        if not high_corr.empty:
            print("\nAll high correlations:")
            print(high_corr[['Metric', 'Correlation']])
            
            # Save high correlations to CSV
            high_corr.to_csv(f"{output_dir}/high_correlations.csv", index=False)
            print(f"High correlation metrics have been saved to: {output_dir}/high_correlations.csv")
                        
        else:
            print("No metrics with high correlation (|Correlation| > 0.7) found")
            
        print(f"\nAnalysis complete! All results saved to {output_dir} directory")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()