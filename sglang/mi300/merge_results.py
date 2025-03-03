import pandas as pd
import glob
import sys
import re

def extract_concurrency(filename):
    """Extract concurrency value from filename."""
    match = re.search(r'_(\d+)\.csv$', filename)
    return int(match.group(1)) if match else 0

def read_csv_to_dict(file_path):
    """Read CSV file and convert to dictionary."""
    df = pd.read_csv(file_path)
    return dict(zip(df['Counter Name'], df['Value']))

def merge_results(file_pattern):
    """Merge results from multiple CSV files."""
    # Get all matching files and sort by concurrency
    files = glob.glob(file_pattern)
    files.sort(key=extract_concurrency)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    # Initialize results list
    results = []
    
    # Initialize metrics we want to track
    metrics = [
        'Expected Concurrency', 'Actual Concurrency', 'Successful requests',
        'Benchmark duration (s)', 'Total input tokens', 'Total generated tokens',
        'Request throughput (req/s)', 'Input token throughput (tok/s)',
        'Output token throughput (tok/s)', 'Mean TTFT (ms)', 'Median TTFT (ms)',
        'P99 TTFT (ms)', 'Mean TPOT (ms)', 'Median TPOT (ms)', 'P99 TPOT (ms)'
    ]
    
    # Process each file
    for file_path in files:
        data = read_csv_to_dict(file_path)
        row = [data['Model'], data['Backend']]  # Add model and backend first
        row.extend([data[metric] for metric in metrics])
        results.append(row)
    
    # Create DataFrame with model and backend as first columns
    columns = ['Model', 'Backend'] + metrics
    df = pd.DataFrame(results, columns=columns)
    
    # Convert Expected Concurrency to numeric type before sorting
    df['Expected Concurrency'] = pd.to_numeric(df['Expected Concurrency'])
    
    # Sort by Expected Concurrency first, then by Model name
    df = df.sort_values(['Expected Concurrency', 'Model'])
    
    # Save to CSV
    output_file = 'merged_benchmark_results.csv'
    df.to_csv(output_file, index=False)
    print(f"Results merged and saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python merge_results.py <file_pattern>")
        print("Example: python merge_results.py 'sglang_benchmark_result_*.csv'")
        sys.exit(1)
    
    merge_results(sys.argv[1]) 