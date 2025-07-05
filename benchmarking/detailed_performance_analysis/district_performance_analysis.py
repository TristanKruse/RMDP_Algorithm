#!/usr/bin/env python3
"""
District-Level Performance Analysis

Creates comprehensive visualizations comparing algorithm performance across districts,
including geographic patterns, demand correlation, and district characteristics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class DistrictPerformanceAnalyzer:
    """Analyze algorithm performance across different districts."""
    
    def __init__(self, results_dir: str = "data/simulation_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("benchmarking/detailed_performance_analysis/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for methods
        self.method_colors = {
            "RL-ACA": "#F18F01",                 # Orange - RL model
            "Fastest ACA": "#2E86AB",            # Blue - Baseline
            "ACA (Buffer=17)": "#A23B72",        # Purple - Heuristic
            "Meituan Baseline": "#63B600"        # Green - Meituan's method
        }

    def load_data(self):
        """Load benchmark results and district characteristics."""
        # Load benchmark results
        benchmark_files = list(self.results_dir.glob("fastest_aca_filtered_results_*.csv"))
        if not benchmark_files:
            benchmark_files = list(self.results_dir.glob("benchmark_results*.csv"))
        
        if not benchmark_files:
            raise FileNotFoundError("No benchmark results found!")
            
        latest_file = max(benchmark_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        print(f"📊 Loading data from: {latest_file.name}")
        
        # Method name mapping
        method_mapping = {
            "rl_aca": "RL-ACA",
            "fastest_aca": "Fastest ACA", 
            "aca_17": "ACA (Buffer=17)"
        }
        df["method_display"] = df["method"].map(method_mapping).fillna(df["method"])
        
        # Load Meituan baseline if available
        meituan_files = list(Path("data/meituan_benchmark").glob("meituan_ground_truth_performance*.csv"))
        if meituan_files:
            latest_meituan = max(meituan_files, key=lambda x: x.stat().st_mtime)
            meituan_df = pd.read_csv(latest_meituan)
            meituan_df["method"] = "meituan_baseline"
            meituan_df["method_display"] = "Meituan Baseline"
            
            # Align columns
            common_cols = ["district", "day", "on_time_delivery_rate", "total_delay", 
                          "avg_delay_late_orders", "avg_distance_per_order"]
            meituan_subset = meituan_df[common_cols + ["method", "method_display"]]
            df_subset = df[common_cols + ["method", "method_display"]]
            
            df = pd.concat([df_subset, meituan_subset], ignore_index=True)
        
        return df

    def create_district_heatmap(self, df, metric="on_time_delivery_rate"):
        """Create heatmap showing performance by district and method."""
        # Pivot data for heatmap
        pivot_data = df.groupby(["district", "method_display"])[metric].mean().unstack()
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(pivot_data, 
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlGn',
                   center=pivot_data.mean().mean(),
                   cbar_kws={'label': metric.replace('_', ' ').title()})
        
        plt.title(f'{metric.replace("_", " ").title()} by District and Method', 
                 fontsize=16, pad=20)
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('District', fontsize=12)
        plt.tight_layout()
        
        output_path = self.output_dir / f"district_heatmap_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")

    def create_district_ranking(self, df):
        """Create district ranking visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [
            ("on_time_delivery_rate", "On-Time Rate (%)", True),
            ("total_delay", "Total Delay (min)", False),
            ("avg_delay_late_orders", "Avg Delay Late Orders (min)", False),
            ("avg_distance_per_order", "Avg Distance (km)", False)
        ]
        
        for idx, (metric, title, higher_better) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Calculate district averages across all methods
            district_avg = df.groupby("district")[metric].mean().sort_values(ascending=not higher_better)
            
            # Create bar plot
            bars = ax.bar(range(len(district_avg)), district_avg.values, 
                         color='steelblue', alpha=0.7)
            
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel('District (Ranked)', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_xticks(range(len(district_avg)))
            ax.set_xticklabels([f'D{d}' for d in district_avg.index], rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, district_avg.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('District Performance Rankings', fontsize=16, y=0.95)
        plt.tight_layout()
        
        output_path = self.output_dir / "district_rankings.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")

    def create_algorithm_district_comparison(self, df):
        """Compare how different algorithms perform across districts."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [
            ("on_time_delivery_rate", "On-Time Rate (%)", True),
            ("total_delay", "Total Delay (min)", False),
            ("avg_delay_late_orders", "Avg Delay Late Orders (min)", False), 
            ("avg_distance_per_order", "Avg Distance (km)", False)
        ]
        
        for idx, (metric, title, higher_better) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Create boxplot for each method
            method_data = []
            method_labels = []
            colors = []
            
            for method in df["method_display"].unique():
                if method in self.method_colors:
                    method_subset = df[df["method_display"] == method]
                    district_values = method_subset.groupby("district")[metric].mean()
                    method_data.append(district_values.values)
                    method_labels.append(method)
                    colors.append(self.method_colors[method])
            
            bp = ax.boxplot(method_data, patch_artist=True, labels=method_labels)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_ylabel(title, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Algorithm Performance Distribution Across Districts', fontsize=16, y=0.95)
        plt.tight_layout()
        
        output_path = self.output_dir / "algorithm_district_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")

    def create_district_characteristics_analysis(self, df):
        """Analyze how district characteristics correlate with performance."""
        # Calculate district summary statistics
        district_stats = df.groupby("district").agg({
            "on_time_delivery_rate": ["mean", "std"],
            "total_delay": ["mean", "std"],
            "avg_distance_per_order": ["mean", "std"]
        }).round(2)
        
        # Flatten column names
        district_stats.columns = ['_'.join(col).strip() for col in district_stats.columns]
        
        # Create district variability analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Performance variability
        ax1 = axes[0]
        x = district_stats["on_time_delivery_rate_mean"]
        y = district_stats["on_time_delivery_rate_std"]
        scatter = ax1.scatter(x, y, s=100, alpha=0.7, c=range(len(x)), cmap='viridis')
        
        for i, district in enumerate(district_stats.index):
            ax1.annotate(f'D{district}', (x.iloc[i], y.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel('Mean On-Time Rate (%)', fontsize=12)
        ax1.set_ylabel('Std Dev On-Time Rate (%)', fontsize=12)
        ax1.set_title('District Performance Consistency', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Delay patterns
        ax2 = axes[1]
        x = district_stats["total_delay_mean"]
        y = district_stats["avg_distance_per_order_mean"]
        scatter = ax2.scatter(x, y, s=100, alpha=0.7, c=range(len(x)), cmap='viridis')
        
        for i, district in enumerate(district_stats.index):
            ax2.annotate(f'D{district}', (x.iloc[i], y.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Mean Total Delay (min)', fontsize=12)
        ax2.set_ylabel('Mean Distance per Order (km)', fontsize=12)
        ax2.set_title('District Delay vs Distance Patterns', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Algorithm advantage by district
        ax3 = axes[2]
        rl_aca_data = df[df["method_display"] == "RL-ACA"].groupby("district")["on_time_delivery_rate"].mean()
        fastest_aca_data = df[df["method_display"] == "Fastest ACA"].groupby("district")["on_time_delivery_rate"].mean()
        
        # Calculate advantage (positive = RL-ACA better)
        advantage = rl_aca_data - fastest_aca_data
        
        colors = ['green' if x > 0 else 'red' for x in advantage.values]
        bars = ax3.bar(range(len(advantage)), advantage.values, color=colors, alpha=0.7)
        
        ax3.set_xlabel('District', fontsize=12)
        ax3.set_ylabel('RL-ACA Advantage (%)', fontsize=12)
        ax3.set_title('RL-ACA vs Fastest ACA by District', fontsize=14)
        ax3.set_xticks(range(len(advantage)))
        ax3.set_xticklabels([f'D{d}' for d in advantage.index], rotation=45)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, advantage.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "district_characteristics_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")
        
        return district_stats

    def generate_district_summary_report(self, df):
        """Generate a comprehensive district analysis report."""
        report = []
        report.append("# District Performance Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Overall statistics
        total_districts = df["district"].nunique()
        total_days = df["day"].nunique()
        methods = df["method_display"].unique()
        
        report.append(f"## Dataset Overview")
        report.append(f"- Districts analyzed: {total_districts}")
        report.append(f"- Days of data: {total_days}")
        report.append(f"- Algorithms compared: {', '.join(methods)}")
        report.append("")
        
        # Performance by district
        report.append("## District Performance Summary")
        district_summary = df.groupby("district").agg({
            "on_time_delivery_rate": ["mean", "std"],
            "total_delay": ["mean", "std"],
            "avg_distance_per_order": "mean"
        }).round(2)
        
        report.append("```")
        report.append(district_summary.to_string())
        report.append("```")
        report.append("")
        
        # Best and worst performing districts
        district_ontime = df.groupby("district")["on_time_delivery_rate"].mean()
        best_districts = district_ontime.nlargest(3)
        worst_districts = district_ontime.nsmallest(3)
        
        report.append("## Key Findings")
        report.append(f"### Best Performing Districts (On-Time Rate)")
        for district, rate in best_districts.items():
            report.append(f"- District {district}: {rate:.1f}%")
        
        report.append(f"### Worst Performing Districts (On-Time Rate)")
        for district, rate in worst_districts.items():
            report.append(f"- District {district}: {rate:.1f}%")
        
        report.append("")
        
        # Algorithm comparison by district
        if "RL-ACA" in methods and "Fastest ACA" in methods:
            rl_performance = df[df["method_display"] == "RL-ACA"].groupby("district")["on_time_delivery_rate"].mean()
            aca_performance = df[df["method_display"] == "Fastest ACA"].groupby("district")["on_time_delivery_rate"].mean()
            advantage = rl_performance - aca_performance
            
            rl_wins = (advantage > 0).sum()
            aca_wins = (advantage < 0).sum()
            
            report.append("### RL-ACA vs Fastest ACA Performance")
            report.append(f"- Districts where RL-ACA wins: {rl_wins}/{total_districts}")
            report.append(f"- Districts where Fastest ACA wins: {aca_wins}/{total_districts}")
            report.append(f"- Average RL-ACA advantage: {advantage.mean():.2f}%")
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / "district_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Saved report: {report_path}")
        return report_text

    def run_full_analysis(self):
        """Run complete district performance analysis."""
        print("🔍 Starting District Performance Analysis")
        print("=" * 50)
        
        # Load data
        df = self.load_data()
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        self.create_district_heatmap(df, "on_time_delivery_rate")
        self.create_district_heatmap(df, "total_delay") 
        self.create_district_ranking(df)
        self.create_algorithm_district_comparison(df)
        district_stats = self.create_district_characteristics_analysis(df)
        
        # Generate report
        print("\n📄 Generating report...")
        self.generate_district_summary_report(df)
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
        return df, district_stats


def main():
    """Main execution function."""
    analyzer = DistrictPerformanceAnalyzer()
    df, stats = analyzer.run_full_analysis()
    return df, stats


if __name__ == "__main__":
    main()