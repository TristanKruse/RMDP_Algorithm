import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import json
from pathlib import Path
from scipy import stats  # Add this import at the top

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class BenchmarkPipeline:
    """
    Enhanced benchmarking pipeline with automated result processing,
    statistical analysis, and integration capabilities.
    """

    def __init__(self, base_results_dir: str = "data/simulation_results"):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)
        self.current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_latest_results(self, apply_filtering: bool = True) -> pd.DataFrame:
        """Load the most recent benchmark results with optional filtering."""
        # Look for filtered files first, then original files
        filtered_files = list(self.base_results_dir.glob("fastest_aca_filtered_results_*.csv"))
        csv_files = list(self.base_results_dir.glob("benchmark_results_*.csv"))
        combined_files = list(self.base_results_dir.glob("combined_with_baseline_*.csv"))

        # Prioritize filtered files if they exist
        if filtered_files:
            latest_file = max(filtered_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading FILTERED results from: {latest_file}")
            return pd.read_csv(latest_file)

        # Fall back to original files
        all_files = csv_files + combined_files

        if not all_files:
            raise FileNotFoundError(
                "No benchmark results found! Looking for 'benchmark_results_*.csv', 'combined_with_baseline_*.csv', or 'fastest_aca_filtered_results_*.csv'"
            )

        latest_file = max(all_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading results from: {latest_file}")
        df = pd.read_csv(latest_file)

        # Apply basic filtering if requested and no pre-filtered file was found
        if apply_filtering:
            logger.info("Applying basic data filtering to remove unrealistic results...")
            df = self._filter_unrealistic_data(df)

        return df

    def _filter_unrealistic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic filtering to remove unrealistic results."""
        initial_count = len(df)

        # Remove impossible on-time rates
        df = df[(df["on_time_delivery_rate"] >= -10) & (df["on_time_delivery_rate"] <= 105)]

        # Remove extreme delays
        median_delay = df["total_delay"].median()
        max_reasonable_delay = median_delay * 10
        df = df[(df["total_delay"] >= 0) & (df["total_delay"] <= max_reasonable_delay)]

        # Remove RL results with extremely negative performance
        rl_mask = df["method"].str.contains("rl", case=False, na=False)
        extreme_negative = (df["on_time_delivery_rate"] < -50) & rl_mask
        df = df[~extreme_negative]

        removed_count = initial_count - len(df)
        logger.info(
            f"Filtered out {removed_count}/{initial_count} unrealistic records ({(removed_count/initial_count)*100:.1f}%)"
        )

        return df

    def calculate_statistical_significance(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistical significance between methods.
        Performs paired t-tests for key metrics.
        """
        methods = df["method"].unique()
        metrics = ["total_delay", "on_time_delivery_rate", "avg_distance_per_order"]
        significance_results = {}

        # Group by dataset (district + day) for paired comparisons
        dataset_grouped = df.groupby(["district", "day"])

        for metric in metrics:
            significance_results[metric] = {}

            # Collect paired data for each method combination
            for i, method1 in enumerate(methods):
                for method2 in methods[i + 1 :]:
                    method1_values = []
                    method2_values = []

                    for (district, day), group in dataset_grouped:
                        m1_data = group[group["method"] == method1][metric]
                        m2_data = group[group["method"] == method2][metric]

                        if len(m1_data) > 0 and len(m2_data) > 0:
                            method1_values.append(m1_data.mean())
                            method2_values.append(m2_data.mean())

                    if len(method1_values) > 2:  # Need at least 3 paired observations
                        t_stat, p_value = stats.ttest_rel(method1_values, method2_values)
                        effect_size = (np.mean(method1_values) - np.mean(method2_values)) / np.std(method1_values)

                        significance_results[metric][f"{method1}_vs_{method2}"] = {
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "is_significant": p_value < 0.05,
                            "effect_size": effect_size,
                            "method1_mean": np.mean(method1_values),
                            "method2_mean": np.mean(method2_values),
                        }

        return significance_results

    def identify_problematic_datasets(self, df: pd.DataFrame, threshold_percentile: float = 90) -> Dict:
        """
        Identify datasets where RL performance is particularly poor.
        """
        problematic_datasets = {}

        # Calculate performance gaps for each dataset
        dataset_performance = []

        for (district, day), group in df.groupby(["district", "day"]):
            methods_data = {}
            for method in group["method"].unique():
                method_data = group[group["method"] == method]
                methods_data[method] = {
                    "total_delay": method_data["total_delay"].mean(),
                    "on_time_delivery_rate": method_data["on_time_delivery_rate"].mean(),
                }

            # Calculate performance gaps (RL vs best performing method)
            if "rl_aca" in methods_data and len(methods_data) > 1:
                rl_delay = methods_data["rl_aca"]["total_delay"]
                rl_on_time = methods_data["rl_aca"]["on_time_delivery_rate"]

                # Find best performing method (lowest delay)
                best_delay = min([methods_data[m]["total_delay"] for m in methods_data if m != "rl_aca"])
                best_on_time = max([methods_data[m]["on_time_delivery_rate"] for m in methods_data if m != "rl_aca"])

                delay_gap = (rl_delay - best_delay) / best_delay * 100  # Percentage worse
                on_time_gap = best_on_time - rl_on_time  # Percentage points worse

                dataset_performance.append(
                    {
                        "district": district,
                        "day": day,
                        "delay_gap_percent": delay_gap,
                        "on_time_gap_points": on_time_gap,
                        "rl_delay": rl_delay,
                        "best_delay": best_delay,
                        "rl_on_time": rl_on_time,
                        "best_on_time": best_on_time,
                    }
                )

        performance_df = pd.DataFrame(dataset_performance)

        # Identify problematic datasets (worst performing percentile)
        if len(performance_df) > 0:
            delay_threshold = np.percentile(performance_df["delay_gap_percent"], threshold_percentile)
            on_time_threshold = np.percentile(performance_df["on_time_gap_points"], threshold_percentile)

            problematic_datasets["worst_delay_performance"] = performance_df[
                performance_df["delay_gap_percent"] >= delay_threshold
            ].to_dict("records")

            problematic_datasets["worst_on_time_performance"] = performance_df[
                performance_df["on_time_gap_points"] >= on_time_threshold
            ].to_dict("records")
        else:
            problematic_datasets["worst_delay_performance"] = []
            problematic_datasets["worst_on_time_performance"] = []

        return problematic_datasets, performance_df

    def detect_unrealistic_results(self, df: pd.DataFrame) -> Dict:
        """
        Detect potentially unrealistic simulation results using statistical outliers.
        """
        unrealistic_results = {}

        for metric in ["total_delay", "on_time_delivery_rate", "avg_distance_per_order"]:
            # Skip if metric doesn't exist
            if metric not in df.columns:
                continue

            # Calculate Z-scores for each method
            method_outliers = {}

            for method in df["method"].unique():
                method_data = df[df["method"] == method][metric]

                # Skip if not enough data points
                if len(method_data) < 3:
                    continue

                z_scores = np.abs(stats.zscore(method_data))
                outlier_threshold = 3  # 3 standard deviations

                outlier_indices = method_data[z_scores > outlier_threshold].index
                outliers = df.loc[outlier_indices][["district", "day", metric]].to_dict("records")

                if outliers:
                    method_outliers[method] = outliers

            if method_outliers:
                unrealistic_results[metric] = method_outliers

        return unrealistic_results

    def generate_performance_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive performance analysis report."""

        # Calculate overall statistics
        overall_stats = (
            df.groupby("method")
            .agg(
                {
                    "total_delay": ["mean", "std", "min", "max"],
                    "on_time_delivery_rate": ["mean", "std", "min", "max"],
                    "avg_distance_per_order": ["mean", "std", "min", "max"],
                }
            )
            .round(2)
        )

        # Calculate statistical significance
        significance = self.calculate_statistical_significance(df)

        # Identify problematic datasets
        problematic, performance_df = self.identify_problematic_datasets(df)

        # Detect unrealistic results
        unrealistic = self.detect_unrealistic_results(df)

        report = f"""
# Algorithm Benchmarking Performance Report (FILTERED DATA)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
Total datasets analyzed: {len(df.groupby(['district', 'day']))}
Districts included: {df['district'].nunique()} (after filtering)
Methods compared: {', '.join(df['method'].unique())}

## Overall Performance Statistics

### Total Delay (minutes)
{overall_stats['total_delay'].to_string()}

### On-Time Delivery Rate (%)
{overall_stats['on_time_delivery_rate'].to_string()}

### Average Distance per Order (km)
{overall_stats['avg_distance_per_order'].to_string()}

## Key Performance Gaps

"""

        # Add performance comparison
        method_means = df.groupby("method")["on_time_delivery_rate"].mean().sort_values(ascending=False)
        report += "### On-Time Delivery Rate Ranking\n"
        for i, (method, rate) in enumerate(method_means.items(), 1):
            report += f"{i}. **{method}**: {rate:.1f}%\n"

        if "meituan_baseline" in method_means.index and "rl_aca" in method_means.index:
            baseline_rate = method_means["meituan_baseline"]
            rl_rate = method_means["rl_aca"]
            gap = baseline_rate - rl_rate
            report += f"\n**RL-ACA vs Meituan Gap**: {gap:.1f} percentage points\n"

        report += "\n## Statistical Significance Analysis\n"

        # Add significance results
        for metric, comparisons in significance.items():
            report += f"\n### {metric.replace('_', ' ').title()}\n"
            for comparison, stats_data in comparisons.items():
                significance_indicator = "**SIGNIFICANT**" if stats_data["is_significant"] else "Not significant"
                report += f"- {comparison}: p={stats_data['p_value']:.4f} ({significance_indicator})\n"
                report += f"  Effect size: {stats_data['effect_size']:.3f}\n"

        # Add problematic datasets section
        if len(problematic["worst_delay_performance"]) > 0:
            report += f"""
## Problematic Datasets Analysis

### Worst RL Performance (Delay)
Top {min(5, len(problematic['worst_delay_performance']))} datasets where RL-ACA performs worst:
"""
            for dataset in problematic["worst_delay_performance"][:5]:  # Show top 5
                report += f"- District {dataset['district']}, Day {dataset['day']}: {dataset['delay_gap_percent']:.1f}% worse delay\n"

        if len(problematic["worst_on_time_performance"]) > 0:
            report += f"""
### Worst RL Performance (On-time Rate)
Top {min(5, len(problematic['worst_on_time_performance']))} datasets where RL-ACA performs worst:
"""
            for dataset in problematic["worst_on_time_performance"][:5]:
                report += f"- District {dataset['district']}, Day {dataset['day']}: {dataset['on_time_gap_points']:.1f} percentage points worse\n"

        # Add unrealistic results section
        if unrealistic:
            report += "\n## Potentially Unrealistic Results (Statistical Outliers)\n"
            for metric, method_outliers in unrealistic.items():
                report += f"\n### {metric.replace('_', ' ').title()}\n"
                for method, outliers in method_outliers.items():
                    report += f"- {method}: {len(outliers)} outlier(s) detected\n"

        return report

    def save_analysis_results(self, df: pd.DataFrame):
        """Save comprehensive analysis results to files."""

        # Generate report
        report = self.generate_performance_report(df)

        # Save report
        report_path = self.base_results_dir / f"performance_report_{self.current_timestamp}.md"
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Saved performance report to: {report_path}")

        # Save statistical significance results
        significance = self.calculate_statistical_significance(df)
        significance_path = self.base_results_dir / f"statistical_significance_{self.current_timestamp}.json"
        with open(significance_path, "w") as f:
            json.dump(significance, f, indent=2, default=str)

        # Save problematic datasets
        problematic, performance_df = self.identify_problematic_datasets(df)
        problematic_path = self.base_results_dir / f"problematic_datasets_{self.current_timestamp}.json"
        with open(problematic_path, "w") as f:
            json.dump(problematic, f, indent=2, default=str)

        # Save performance gaps analysis
        if len(performance_df) > 0:
            performance_df.to_csv(self.base_results_dir / f"performance_gaps_{self.current_timestamp}.csv", index=False)

        return {
            "report_path": report_path,
            "significance_path": significance_path,
            "problematic_path": problematic_path,
        }


def main():
    """Main function to run the enhanced pipeline analysis."""

    pipeline = BenchmarkPipeline()

    try:
        # Load latest results
        df = pipeline.load_latest_results()
        logger.info(f"Loaded {len(df)} benchmark records")

        # Print quick summary
        print(f"\n📊 DATA SUMMARY")
        print("=" * 40)
        print(f"Records: {len(df)}")
        print(f"Districts: {df['district'].nunique()}")
        print(f"Methods: {', '.join(df['method'].unique())}")
        print(f"District-Day combinations: {len(df.groupby(['district', 'day']))}")

        # Run comprehensive analysis
        analysis_paths = pipeline.save_analysis_results(df)

        logger.info("Enhanced pipeline analysis completed!")
        logger.info(f"Results saved to: {pipeline.base_results_dir}")

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARKING PIPELINE ANALYSIS COMPLETE")
        print("=" * 60)

        print(f"\nAnalysis files generated:")
        for description, path in analysis_paths.items():
            print(f"- {description}: {path}")

        # Quick performance summary
        print(f"\n📈 QUICK PERFORMANCE SUMMARY")
        print("=" * 40)
        method_means = df.groupby("method")["on_time_delivery_rate"].mean().sort_values(ascending=False)
        for method, rate in method_means.items():
            print(f"{method:20s}: {rate:6.1f}%")

        return df, analysis_paths

    except Exception as e:
        logger.error(f"Pipeline analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
