#!/usr/bin/env python3
"""
Filter to remove districts where the fastest_aca method shows negative on-time rates.
Since fastest_aca should be the most reliable baseline, negative rates indicate simulation bugs.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_latest_benchmark_file():
    """Find the most recent benchmark results file."""
    results_dir = Path("data/simulation_results")
    
    # Look for timestamped files first, then fall back to fixed name
    benchmark_files = list(results_dir.glob("benchmark_results_*.csv"))
    if benchmark_files:
        # Return the most recent timestamped file
        latest_file = max(benchmark_files, key=lambda x: x.stat().st_mtime)
        return latest_file
    
    # Fall back to fixed name
    benchmark_file = results_dir / "benchmark_results.csv"
    if benchmark_file.exists():
        return benchmark_file
    
    raise FileNotFoundError("benchmark_results.csv not found!")


def filter_fastest_aca_negative_districts(df):
    """
    Remove districts where fastest_aca method has negative on-time delivery rates.
    This targets districts with fundamental simulation issues.
    """

    initial_count = len(df)
    initial_districts = df["district"].nunique()
    initial_datasets = len(df.groupby(["district", "day"]))

    logger.info(
        f"Starting with {initial_count} records from {initial_districts} districts ({initial_datasets} district-day combinations)"
    )

    # Find districts where fastest_aca has negative on-time rates
    fastest_aca_data = df[df["method"] == "fastest_aca"]

    if len(fastest_aca_data) == 0:
        logger.warning(
            "No 'fastest_aca' method found in data! Available methods: {}".format(list(df["method"].unique()))
        )
        return df, []

    negative_fastest_aca = fastest_aca_data[fastest_aca_data["on_time_delivery_rate"] < 0]
    problematic_districts = negative_fastest_aca["district"].unique()

    logger.info(f"Found {len(problematic_districts)} districts where fastest_aca has negative on-time rates")

    if len(problematic_districts) > 0:
        # Show details of problematic districts
        for district in sorted(problematic_districts):
            district_fastest_aca = fastest_aca_data[fastest_aca_data["district"] == district]
            min_rate = district_fastest_aca["on_time_delivery_rate"].min()
            avg_rate = district_fastest_aca["on_time_delivery_rate"].mean()
            days_affected = district_fastest_aca["day"].nunique()
            logger.info(
                f"  District {district}: avg={avg_rate:.1f}%, min={min_rate:.1f}%, {days_affected} days affected"
            )

        # Remove all data from these problematic districts
        filtered_df = df[~df["district"].isin(problematic_districts)]

        final_count = len(filtered_df)
        final_districts = filtered_df["district"].nunique()
        final_datasets = len(filtered_df.groupby(["district", "day"]))

        removed_records = initial_count - final_count
        removed_districts = initial_districts - final_districts
        removed_datasets = initial_datasets - final_datasets

        logger.info(
            f"Removed {removed_records} records from {removed_districts} districts with problematic fastest_aca"
        )
        logger.info(f"Removed {removed_datasets} district-day combinations")
        logger.info(
            f"Remaining: {final_count} records from {final_districts} districts ({final_datasets} combinations)"
        )

        return filtered_df, problematic_districts.tolist()

    else:
        logger.info("No districts with negative fastest_aca on-time rates found - no filtering needed")
        return df, []


def analyze_remaining_performance(df):
    """Analyze performance in remaining districts after filtering."""

    if len(df) == 0:
        logger.warning("No data remaining after filtering!")
        return

    logger.info("\nPerformance analysis of remaining districts:")

    for method in df["method"].unique():
        method_data = df[df["method"] == method]
        if len(method_data) > 0:
            avg_ontime = method_data["on_time_delivery_rate"].mean()
            min_ontime = method_data["on_time_delivery_rate"].min()
            max_ontime = method_data["on_time_delivery_rate"].max()
            districts_count = method_data["district"].nunique()

            logger.info(
                f"  {method:15s}: avg={avg_ontime:6.1f}%, min={min_ontime:6.1f}%, max={max_ontime:6.1f}% ({districts_count} districts)"
            )


def generate_fastest_aca_filter_report(original_df, filtered_df, removed_districts):
    """Generate a report on fastest_aca filtering."""

    report = []
    report.append("# Fastest ACA District Filtering Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Summary statistics
    original_count = len(original_df)
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count

    original_districts = original_df["district"].nunique()
    filtered_districts = filtered_df["district"].nunique() if filtered_count > 0 else 0

    report.append("## Summary")
    report.append(f"- **Filtering Criterion**: Remove districts where fastest_aca has negative on-time rates")
    report.append(f"- **Rationale**: fastest_aca should be most reliable baseline method")
    report.append(f"- **Original Records**: {original_count:,}")
    report.append(f"- **Filtered Records**: {filtered_count:,}")
    report.append(f"- **Removed Records**: {removed_count:,} ({(removed_count/original_count)*100:.1f}%)")
    report.append(f"- **Original Districts**: {original_districts}")
    report.append(f"- **Remaining Districts**: {filtered_districts}")
    report.append(f"- **Removed Districts**: {len(removed_districts)}")
    report.append("")

    if removed_districts:
        report.append("## Removed Districts (Problematic fastest_aca Performance)")
        report.append("")

        fastest_aca_data = original_df[original_df["method"] == "fastest_aca"]

        for district in sorted(removed_districts):
            district_data = fastest_aca_data[fastest_aca_data["district"] == district]

            if len(district_data) > 0:
                avg_rate = district_data["on_time_delivery_rate"].mean()
                min_rate = district_data["on_time_delivery_rate"].min()
                days_count = district_data["day"].nunique()

                report.append(f"- **District {district}**:")
                report.append(f"  - fastest_aca avg on-time rate: {avg_rate:.1f}%")
                report.append(f"  - fastest_aca min on-time rate: {min_rate:.1f}%")
                report.append(f"  - Days affected: {days_count}")
                report.append("")

    # Performance comparison (remaining districts only)
    if filtered_count > 0:
        report.append("## Performance in Remaining Districts")
        report.append("Average performance across districts with valid fastest_aca baseline:")
        report.append("")
        report.append("| Method | Avg On-Time Rate | Min Rate | Max Rate | Districts | Records |")
        report.append("|--------|------------------|----------|----------|-----------|---------|")

        for method in filtered_df["method"].unique():
            method_data = filtered_df[filtered_df["method"] == method]
            avg_ontime = method_data["on_time_delivery_rate"].mean()
            min_ontime = method_data["on_time_delivery_rate"].min()
            max_ontime = method_data["on_time_delivery_rate"].max()
            districts_count = method_data["district"].nunique()
            record_count = len(method_data)

            report.append(
                f"| {method} | {avg_ontime:.1f}% | {min_ontime:.1f}% | {max_ontime:.1f}% | {districts_count} | {record_count} |"
            )

        report.append("")
        report.append("## Key Insights")

        # Compare RL vs fastest_aca in remaining districts
        if "rl_aca" in filtered_df["method"].values and "fastest_aca" in filtered_df["method"].values:
            rl_performance = filtered_df[filtered_df["method"] == "rl_aca"]["on_time_delivery_rate"].mean()
            fastest_performance = filtered_df[filtered_df["method"] == "fastest_aca"]["on_time_delivery_rate"].mean()
            gap = rl_performance - fastest_performance

            report.append(f"- **RL vs Fastest ACA Gap**: {gap:.1f} percentage points")
            report.append(
                f"- **RL Performance**: {rl_performance:.1f}% (vs {fastest_performance:.1f}% for fastest_aca)"
            )

        if "meituan_baseline" in filtered_df["method"].values:
            meituan_performance = filtered_df[filtered_df["method"] == "meituan_baseline"][
                "on_time_delivery_rate"
            ].mean()
            report.append(f"- **Meituan Baseline**: {meituan_performance:.1f}% on-time rate")

    else:
        report.append("## ⚠️ WARNING")
        report.append("**ALL DISTRICTS WERE FILTERED OUT!**")
        report.append("")
        report.append("This indicates a fundamental issue with the simulation or data processing.")
        report.append("All districts show negative fastest_aca performance, which is impossible in reality.")
        report.append("")
        report.append("**Recommended Actions:**")
        report.append("1. Check simulation logic for bugs")
        report.append("2. Verify KPI calculation methods")
        report.append("3. Validate input data quality")
        report.append("4. Review algorithm implementations")

    return "\n".join(report)


def main():
    """Main function to filter districts with negative fastest_aca performance."""

    print("🔍 FILTERING: NEGATIVE FASTEST ACA DISTRICTS")
    print("=" * 60)

    try:
        # 1. Find and load latest benchmark file
        input_file = find_latest_benchmark_file()
        print(f"📁 Found benchmark file: {input_file}")

        df = pd.read_csv(input_file)
        print(f"📊 Loaded {len(df)} records from {df['district'].nunique()} districts")
        print(f"🔧 Methods: {', '.join(df['method'].unique())}")

        # 2. Check if fastest_aca exists
        if "fastest_aca" not in df["method"].values:
            print(f"❌ ERROR: 'fastest_aca' method not found!")
            print(f"Available methods: {list(df['method'].unique())}")
            return None

        # 3. Apply filtering
        print(f"\n🚿 Filtering districts where fastest_aca has negative on-time rates...")
        filtered_df, removed_districts = filter_fastest_aca_negative_districts(df)

        # 4. Analyze results
        analyze_remaining_performance(filtered_df)

        # 5. Save results with fixed file names
        output_dir = Path("data/simulation_results")

        # Save filtered data with both timestamped and fixed filename patterns
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Timestamped version for history
        timestamped_file = output_dir / f"fastest_aca_filtered_results_{timestamp}.csv"
        filtered_df.to_csv(timestamped_file, index=False)
        print(f"💾 Saved timestamped filtered data: {timestamped_file}")
        
        # Fixed name version for current use
        fixed_file = output_dir / "fastest_aca_filtered_results.csv"
        filtered_df.to_csv(fixed_file, index=False)
        print(f"💾 Saved current filtered data: {fixed_file}")

        # Save filtering report with both versions
        report = generate_fastest_aca_filter_report(df, filtered_df, removed_districts)
        report_file_timestamped = output_dir / f"filtering_report_{timestamp}.md"
        report_file_fixed = output_dir / "filtering_report.md"
        
        # Save both versions of the report
        with open(report_file_timestamped, "w") as f:
            f.write(report)
        with open(report_file_fixed, "w") as f:
            f.write(report)
        print(f"📋 Saved filtering reports: {report_file_timestamped} and {report_file_fixed}")

        # Summary
        if len(filtered_df) > 0:
            print(f"\n📈 SUMMARY OF REMAINING DATA")
            print("=" * 40)

            for method in filtered_df["method"].unique():
                method_data = filtered_df[filtered_df["method"] == method]
                avg_ontime = method_data["on_time_delivery_rate"].mean()
                districts = method_data["district"].nunique()
                print(f"{method:20s}: {avg_ontime:6.1f}% ({districts} districts)")

            print(f"\n✅ Filtering completed!")
            print(f"🔗 Use this file for analysis: {fixed_file}")
        else:
            print(f"\n⚠️  WARNING: ALL DISTRICTS FILTERED OUT!")
            print(f"This suggests fundamental simulation issues.")
            print(f"Review the filtering report for details.")

        return fixed_file

    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        raise


if __name__ == "__main__":
    main()
