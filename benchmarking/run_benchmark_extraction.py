#!/usr/bin/env python3
"""
Quick script to run the Meituan benchmark extraction and verify results.

This script:
1. Runs the benchmark extraction
2. Loads and displays the results
3. Provides basic validation and statistics

Usage:
    python run_benchmark_extraction.py
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add the project root to Python path so we can import our extraction script
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our extraction function
try:
    from benchmarking.extract_meituan_benchmarks import extract_meituan_benchmarks
except ImportError:
    print("Error: Could not import extraction script.")
    print("Make sure 'extract_meituan_benchmarks.py' is in the data/ directory")
    sys.exit(1)


def validate_benchmark_data(df: pd.DataFrame) -> None:
    """Validate the extracted benchmark data for consistency."""
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)

    # Check for required columns
    required_cols = [
        "district",
        "day",
        "on_time_delivery_rate",
        "total_delay",
        "avg_delay_late_orders",
        "max_delay",
        "avg_distance_per_order",
        "total_orders",
        "orders_delivered",
        "late_orders_count",
        "undelivered_orders",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
    else:
        print("✅ All required columns present")

    # Check data coverage
    expected_districts = set(range(1, 23))  # 1-22
    expected_days = {20221017, 20221018, 20221019, 20221020, 20221021, 20221022, 20221023, 20221024}

    actual_districts = set(df["district"].unique())
    actual_days = set(df["day"].unique())

    print(f"\nData Coverage:")
    print(f"  Districts: {len(actual_districts)}/22 expected")
    print(f"  Missing districts: {expected_districts - actual_districts}")
    print(f"  Days: {len(actual_days)}/8 expected")
    print(f"  Missing days: {expected_days - actual_days}")
    print(f"  Total combinations: {len(df)} (max possible: 176)")

    # Check for data quality issues
    print(f"\nData Quality:")
    zero_orders = len(df[df["total_orders"] == 0])
    zero_delivered = len(df[df["orders_delivered"] == 0])
    negative_rates = len(df[df["on_time_delivery_rate"] < -100])

    print(f"  Zero total orders: {zero_orders}")
    print(f"  Zero delivered orders: {zero_delivered}")
    print(f"  Extreme negative on-time rates (<-100%): {negative_rates}")

    # Check logical consistency
    inconsistent = len(df[df["orders_delivered"] > df["total_orders"]])
    print(f"  Inconsistent delivery counts: {inconsistent}")

    if inconsistent == 0 and zero_orders < len(df) * 0.1:
        print("✅ Data quality looks good")
    else:
        print("⚠️  Some data quality issues detected")


def display_summary_statistics(df: pd.DataFrame) -> None:
    """Display summary statistics of the benchmark data."""
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    # Overall metrics
    print("Overall Performance:")
    print(
        f"  Average on-time rate: {df['on_time_delivery_rate'].mean():.2f}% (std: {df['on_time_delivery_rate'].std():.2f})"
    )
    print(f"  Average total delay: {df['total_delay'].mean():.1f} min (std: {df['total_delay'].std():.1f})")
    print(f"  Average orders per district-day: {df['total_orders'].mean():.1f}")
    print(f"  Average delivery success rate: {(df['orders_delivered']/df['total_orders']).mean()*100:.2f}%")
    print(f"  Average distance per order: {df['avg_distance_per_order'].mean():.2f} km")

    # Performance distribution
    print("\nPerformance Distribution:")
    print(
        f"  On-time rate - Min: {df['on_time_delivery_rate'].min():.1f}%, Max: {df['on_time_delivery_rate'].max():.1f}%"
    )
    print(f"  Total delay - Min: {df['total_delay'].min():.1f}min, Max: {df['total_delay'].max():.1f}min")

    # Best and worst performing district-days
    best_ontime = df.loc[df["on_time_delivery_rate"].idxmax()]
    worst_ontime = df.loc[df["on_time_delivery_rate"].idxmin()]

    print(f"\nBest performing (on-time rate):")
    print(
        f"  District {best_ontime['district']}, Day {best_ontime['day']}: {best_ontime['on_time_delivery_rate']:.1f}%"
    )

    print(f"Worst performing (on-time rate):")
    print(
        f"  District {worst_ontime['district']}, Day {worst_ontime['day']}: {worst_ontime['on_time_delivery_rate']:.1f}%"
    )


def compare_with_simulation_results(benchmark_df: pd.DataFrame) -> None:
    """Compare benchmark data with simulation results if available."""
    print("\n" + "=" * 50)
    print("COMPARISON WITH SIMULATION RESULTS")
    print("=" * 50)

    # Try to load the most recent simulation results
    results_dir = "data/simulation_results"
    if os.path.exists(results_dir):
        csv_files = [
            f for f in os.listdir(results_dir) if f.startswith("quick_benchmark_results_") and f.endswith(".csv")
        ]

        if csv_files:
            # Load the most recent results file
            latest_file = sorted(csv_files)[-1]
            sim_path = os.path.join(results_dir, latest_file)

            print(f"Loading simulation results from: {latest_file}")
            sim_df = pd.read_csv(sim_path)

            # Compare averages by method
            print("\nComparison (averages):")
            print(f"{'Metric':<25} {'Meituan Real':<15} {'fastest_aca':<15} {'aca_17':<15} {'rl_aca':<15}")
            print("-" * 85)

            # Group simulation results by method
            sim_grouped = sim_df.groupby("method").mean()

            # Key metrics to compare
            metrics = [
                ("on_time_delivery_rate", "On-time Rate (%)"),
                ("total_delay", "Total Delay (min)"),
                ("avg_distance_per_order", "Avg Distance (km)"),
                ("total_orders", "Total Orders"),
                ("orders_delivered", "Orders Delivered"),
            ]

            for metric, label in metrics:
                if metric in benchmark_df.columns:
                    real_val = benchmark_df[metric].mean()

                    # Get simulation values for each method
                    fastest_val = sim_grouped.loc["fastest_aca", metric] if "fastest_aca" in sim_grouped.index else 0
                    aca17_val = sim_grouped.loc["aca_17", metric] if "aca_17" in sim_grouped.index else 0
                    rl_val = sim_grouped.loc["rl_aca", metric] if "rl_aca" in sim_grouped.index else 0

                    print(f"{label:<25} {real_val:<15.1f} {fastest_val:<15.1f} {aca17_val:<15.1f} {rl_val:<15.1f}")

            print("\nNote: This is a basic comparison. More detailed analysis can be done in your benchmarking script.")

        else:
            print("No simulation results found for comparison")
    else:
        print("Simulation results directory not found")


def main():
    """Main function to run benchmark extraction and analysis."""
    print("=" * 60)
    print("MEITUAN BENCHMARK EXTRACTION & VALIDATION")
    print("=" * 60)

    # Set up paths - adjust these to match your directory structure
    data_dir = "data/meituan_data/processed/daily_orders"
    output_dir = "data/meituan_benchmark"

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\n❌ Error: Data directory not found: {data_dir}")
        print("\nPlease ensure your Meituan data is organized as:")
        print("data/meituan_data/processed/daily_orders/")
        print("├── 20221017/")
        print("│   ├── district_1_orders.csv")
        print("│   ├── district_2_orders.csv")
        print("│   └── ...")
        print("├── 20221018/")
        print("└── ...")
        return 1

    try:
        # Run the benchmark extraction
        print("\n🚀 Starting benchmark extraction...")
        output_file = extract_meituan_benchmarks(data_dir, output_dir)

        print(f"\n✅ Extraction completed successfully!")
        print(f"📁 Benchmark file created: {output_file}")

        # Load and analyze the results
        print("\n📊 Loading and analyzing results...")
        benchmark_df = pd.read_csv(output_file)

        # Display validation results
        validate_benchmark_data(benchmark_df)

        # Display summary statistics
        display_summary_statistics(benchmark_df)

        # Compare with simulation results if available
        compare_with_simulation_results(benchmark_df)

        print("\n" + "=" * 60)
        print("✅ BENCHMARK EXTRACTION & VALIDATION COMPLETED!")
        print("=" * 60)
        print(f"📁 Benchmark data saved to: {output_file}")
        print(f"📊 Extracted data for {len(benchmark_df)} district-day combinations")
        print("\nNext steps:")
        print("1. Review the validation results above")
        print("2. Integrate benchmark data into your algorithm_benchmarking.py script")
        print("3. Create visualizations comparing algorithms vs Meituan baseline")

        return 0

    except Exception as e:
        print(f"\n❌ Error during benchmark extraction: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check that your data directory structure matches the expected format")
        print("2. Verify that order CSV files contain the required columns")
        print("3. Check file permissions and disk space")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


def main():
    """Main function to run benchmark extraction and analysis."""
    print("=" * 60)
    print("MEITUAN BENCHMARK EXTRACTION")
    print("=" * 60)

    # Set up paths - adjust these to match your directory structure
    data_dir = "data/meituan_data/processed/daily_orders"
    output_dir = "data/meituan_benchmark"

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\n❌ Error: Data directory not found: {data_dir}")
        print("\nPlease ensure your Meituan data is organized as:")
        print("data/meituan_data/processed/daily_orders/")
        print("├── 20221017/")
        print("│   ├── district_1_orders.csv")
        print("│   ├── district_2_orders.csv")
        print("│   └── ...")
        print("├── 20221018/")
        print("└── ...")
        return 1

    try:
        # Run the benchmark extraction
        print("\n🚀 Starting benchmark extraction...")
        output_file = extract_meituan_benchmarks(data_dir, output_dir)

        print(f"\n✅ Extraction completed successfully!")
        print(f"📁 Benchmark file created: {output_file}")

        # Load and analyze the results
        print("\n📊 Loading and analyzing results...")
        benchmark_df = pd.read_csv(output_file)

        # Display validation results
        validate_benchmark_data(benchmark_df)

        # Display summary statistics
        display_summary_statistics(benchmark_df)

        # Compare with simulation results if available
        compare_with_simulation_results(benchmark_df)

        print("\n" + "=" * 60)
        print("✅ BENCHMARK EXTRACTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Benchmark data saved to: {output_file}")
        print(f"📊 Extracted data for {len(benchmark_df)} district-day combinations")
        print("\nNext steps:")
        print("1. Review the validation results above")
        print("2. Integrate benchmark data into your algorithm_benchmarking.py script")
        print("3. Create visualizations comparing algorithms vs Meituan baseline")

        return 0

    except Exception as e:
        print(f"\n❌ Error during benchmark extraction: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check that your data directory structure matches the expected format")
        print("2. Verify that order CSV files contain the required columns")
        print("3. Check file permissions and disk space")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
