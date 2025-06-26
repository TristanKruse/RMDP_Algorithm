#!/usr/bin/env python3
"""
Fix benchmark data consistency issues:
1. Filter rl_aca_phase1_final to same districts as other methods  
2. Update method labels (rl_aca = 4-phase, rl_aca_phase1_final = 1-phase)
"""

import pandas as pd
from pathlib import Path

def fix_benchmark_data():
    """Fix the benchmark results data consistency."""
    
    results_dir = Path("data/simulation_results")
    input_file = results_dir / "benchmark_results.csv"
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"Original data: {len(df)} records")
    
    # Get districts that are present in all methods except rl_aca_phase1_final
    other_methods = ["rl_aca", "fastest_aca", "aca_17"]
    
    # Find districts that exist for all the baseline methods
    districts_by_method = {}
    for method in other_methods:
        method_data = df[df["method"] == method]
        districts_by_method[method] = set(method_data["district"].unique())
        print(f"{method}: {len(districts_by_method[method])} districts")
    
    # Get intersection of districts (districts present in all baseline methods)
    common_districts = set.intersection(*districts_by_method.values())
    print(f"Common districts across baseline methods: {sorted(common_districts)}")
    
    # Filter data to only include common districts
    filtered_df = df[df["district"].isin(common_districts)].copy()
    print(f"After district filtering: {len(filtered_df)} records")
    
    # Check record counts by method after filtering
    print("\nRecord counts after filtering:")
    for method in filtered_df["method"].unique():
        count = len(filtered_df[filtered_df["method"] == method])
        print(f"  {method}: {count} records")
    
    # Save the fixed data
    output_file = results_dir / "benchmark_results.csv"
    filtered_df.to_csv(output_file, index=False)
    print(f"\nSaved fixed data to: {output_file}")
    
    return filtered_df

if __name__ == "__main__":
    fix_benchmark_data()