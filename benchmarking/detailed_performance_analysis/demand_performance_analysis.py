#!/usr/bin/env python3
"""
Demand-Based Performance Analysis

Analyzes algorithm performance across different demand scenarios including:
- Peak vs non-peak hours
- Weekend vs weekday patterns  
- High vs low demand districts
- Order volume correlations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class DemandPerformanceAnalyzer:
    """Analyze algorithm performance across different demand scenarios."""
    
    def __init__(self, results_dir: str = "data/simulation_results"):
        self.results_dir = Path(results_dir)
        self.meituan_dir = Path("data/meituan_benchmark")
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
        """Load benchmark results and add demand context."""
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
        
        # Add demand context
        df = self._add_demand_context(df)
        
        return df

    def _add_demand_context(self, df):
        """Add demand-related context to the dataset."""
        # Convert day to datetime format (assuming format like 20221017)
        df["date"] = pd.to_datetime(df["day"], format="%Y%m%d")
        df["day_of_week"] = df["date"].dt.day_name()
        df["is_weekend"] = df["date"].dt.weekday >= 5
        
        # Load district characteristics if available
        try:
            peak_demand_file = Path("data/meituan_data/abb/peak_demand_by_district.csv")
            if peak_demand_file.exists():
                peak_data = pd.read_csv(peak_demand_file)
                df = df.merge(peak_data, on="district", how="left")
        except Exception as e:
            print(f"⚠️  Could not load peak demand data: {e}")
        
        # Categorize districts by performance level (proxy for demand complexity)
        district_performance = df.groupby("district")["on_time_delivery_rate"].mean()
        performance_terciles = district_performance.quantile([0.33, 0.67])
        
        def categorize_district(district):
            perf = district_performance[district]
            if perf <= performance_terciles.iloc[0]:
                return "High Complexity"
            elif perf <= performance_terciles.iloc[1]:
                return "Medium Complexity"
            else:
                return "Low Complexity"
        
        df["district_complexity"] = df["district"].map(categorize_district)
        
        # Add order volume proxy (using total_delay as indicator of system load)
        delay_terciles = df.groupby(["district", "day"])["total_delay"].mean().quantile([0.33, 0.67])
        
        def categorize_demand(row):
            delay = row["total_delay"]
            if delay >= delay_terciles.iloc[1]:
                return "High Demand"
            elif delay >= delay_terciles.iloc[0]:
                return "Medium Demand"
            else:
                return "Low Demand"
        
        df["demand_level"] = df.apply(categorize_demand, axis=1)
        
        return df

    def create_weekend_weekday_comparison(self, df):
        """Compare performance between weekends and weekdays."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [
            ("on_time_delivery_rate", "On-Time Rate (%)", True),
            ("total_delay", "Total Delay (min)", False),
            ("avg_delay_late_orders", "Avg Delay Late Orders (min)", False),
            ("avg_distance_per_order", "Avg Distance (km)", False)
        ]
        
        for idx, (metric, title, higher_better) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Create grouped bar plot
            weekend_weekday_data = df.groupby(["method_display", "is_weekend"])[metric].mean().unstack()
            weekend_weekday_data.columns = ["Weekday", "Weekend"]
            
            weekend_weekday_data.plot(kind="bar", ax=ax, 
                                    color=["steelblue", "orange"], 
                                    alpha=0.8, width=0.8)
            
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_xlabel("Algorithm", fontsize=10)
            ax.legend(title="Day Type", loc="best")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', fontsize=8)
        
        plt.suptitle('Weekend vs Weekday Performance Comparison', fontsize=16, y=0.95)
        plt.tight_layout()
        
        output_path = self.output_dir / "weekend_weekday_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")

    def create_demand_level_analysis(self, df):
        """Analyze performance across different demand levels."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [
            ("on_time_delivery_rate", "On-Time Rate (%)", True),
            ("total_delay", "Total Delay (min)", False),
            ("avg_delay_late_orders", "Avg Delay Late Orders (min)", False),
            ("avg_distance_per_order", "Avg Distance (km)", False)
        ]
        
        for idx, (metric, title, higher_better) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Create box plot by demand level
            demand_order = ["Low Demand", "Medium Demand", "High Demand"]
            
            method_data = []
            method_labels = []
            positions = []
            colors = []
            
            for i, method in enumerate(df["method_display"].unique()):
                if method in self.method_colors:
                    for j, demand_level in enumerate(demand_order):
                        subset = df[(df["method_display"] == method) & 
                                  (df["demand_level"] == demand_level)]
                        if not subset.empty:
                            method_data.append(subset[metric].values)
                            positions.append(j * len(df["method_display"].unique()) + i + 1)
                            colors.append(self.method_colors[method])
                            if j == 0:  # Only add label once per method
                                method_labels.append(method)
            
            if method_data:
                bp = ax.boxplot(method_data, positions=positions, patch_artist=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Set x-axis labels
                ax.set_xticks([i * len(df["method_display"].unique()) + 
                              len(df["method_display"].unique())/2 for i in range(len(demand_order))])
                ax.set_xticklabels(demand_order)
            
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Performance Across Demand Levels', fontsize=16, y=0.95)
        plt.tight_layout()
        
        output_path = self.output_dir / "demand_level_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")

    def create_district_complexity_analysis(self, df):
        """Analyze performance across district complexity levels."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [
            ("on_time_delivery_rate", "On-Time Rate (%)", True),
            ("total_delay", "Total Delay (min)", False), 
            ("avg_delay_late_orders", "Avg Delay Late Orders (min)", False),
            ("avg_distance_per_order", "Avg Distance (km)", False)
        ]
        
        for idx, (metric, title, higher_better) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Create grouped bar plot
            complexity_data = df.groupby(["method_display", "district_complexity"])[metric].mean().unstack()
            
            complexity_data.plot(kind="bar", ax=ax, 
                               color=["green", "orange", "red"], 
                               alpha=0.8, width=0.8)
            
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_xlabel("Algorithm", fontsize=10)
            ax.legend(title="District Complexity", loc="best")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', fontsize=8)
        
        plt.suptitle('Performance Across District Complexity Levels', fontsize=16, y=0.95)
        plt.tight_layout()
        
        output_path = self.output_dir / "district_complexity_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")

    def create_temporal_patterns_analysis(self, df):
        """Analyze temporal patterns in performance."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Daily performance trends
        ax1 = axes[0, 0]
        daily_performance = df.groupby(["day", "method_display"])["on_time_delivery_rate"].mean().unstack()
        
        for method in daily_performance.columns:
            if method in self.method_colors:
                ax1.plot(daily_performance.index, daily_performance[method], 
                        marker='o', label=method, color=self.method_colors[method], linewidth=2)
        
        ax1.set_title("Daily On-Time Rate Trends", fontsize=12, pad=10)
        ax1.set_xlabel("Day", fontsize=10)
        ax1.set_ylabel("On-Time Rate (%)", fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Day of week patterns
        ax2 = axes[0, 1]
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_performance = df.groupby(["day_of_week", "method_display"])["on_time_delivery_rate"].mean().unstack()
        dow_performance = dow_performance.reindex(dow_order)
        
        for method in dow_performance.columns:
            if method in self.method_colors:
                ax2.plot(dow_performance.index, dow_performance[method], 
                        marker='s', label=method, color=self.method_colors[method], linewidth=2)
        
        ax2.set_title("Day of Week Performance Patterns", fontsize=12, pad=10)
        ax2.set_xlabel("Day of Week", fontsize=10)
        ax2.set_ylabel("On-Time Rate (%)", fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Performance variance by day
        ax3 = axes[1, 0]
        daily_variance = df.groupby(["day", "method_display"])["on_time_delivery_rate"].std().unstack()
        
        for method in daily_variance.columns:
            if method in self.method_colors:
                ax3.plot(daily_variance.index, daily_variance[method], 
                        marker='^', label=method, color=self.method_colors[method], linewidth=2)
        
        ax3.set_title("Daily Performance Variability", fontsize=12, pad=10)
        ax3.set_xlabel("Day", fontsize=10)
        ax3.set_ylabel("On-Time Rate Std Dev", fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Algorithm advantage over time
        ax4 = axes[1, 1]
        if "RL-ACA" in daily_performance.columns and "Fastest ACA" in daily_performance.columns:
            advantage = daily_performance["RL-ACA"] - daily_performance["Fastest ACA"]
            colors = ['green' if x > 0 else 'red' for x in advantage.values]
            
            bars = ax4.bar(range(len(advantage)), advantage.values, color=colors, alpha=0.7)
            ax4.set_title("RL-ACA Daily Advantage vs Fastest ACA", fontsize=12, pad=10)
            ax4.set_xlabel("Day", fontsize=10)
            ax4.set_ylabel("Advantage (%)", fontsize=10)
            ax4.set_xticks(range(len(advantage)))
            ax4.set_xticklabels(advantage.index, rotation=45)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, advantage.values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}',
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        plt.suptitle('Temporal Performance Patterns', fontsize=16, y=0.95)
        plt.tight_layout()
        
        output_path = self.output_dir / "temporal_patterns_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")

    def create_demand_correlation_matrix(self, df):
        """Create correlation matrix between demand factors and performance."""
        # Select numeric columns for correlation
        numeric_cols = ["on_time_delivery_rate", "total_delay", "avg_delay_late_orders", 
                       "max_delay", "avg_distance_per_order"]
        
        # Add dummy variables for categorical columns
        df_corr = df.copy()
        df_corr["is_weekend_num"] = df_corr["is_weekend"].astype(int)
        df_corr["high_demand"] = (df_corr["demand_level"] == "High Demand").astype(int)
        df_corr["high_complexity"] = (df_corr["district_complexity"] == "High Complexity").astype(int)
        
        correlation_cols = numeric_cols + ["is_weekend_num", "high_demand", "high_complexity"]
        corr_data = df_corr[correlation_cols].corr()
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Demand Factors vs Performance Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        
        output_path = self.output_dir / "demand_correlation_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")

    def generate_demand_summary_report(self, df):
        """Generate comprehensive demand analysis report."""
        report = []
        report.append("# Demand-Based Performance Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Dataset overview
        total_records = len(df)
        methods = df["method_display"].unique()
        
        report.append("## Dataset Overview")
        report.append(f"- Total records: {total_records}")
        report.append(f"- Algorithms analyzed: {', '.join(methods)}")
        report.append(f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        report.append("")
        
        # Weekend vs Weekday analysis
        weekend_summary = df.groupby(["method_display", "is_weekend"])["on_time_delivery_rate"].mean().unstack()
        weekend_summary.columns = ["Weekday", "Weekend"]
        weekend_summary["Weekend_Advantage"] = weekend_summary["Weekend"] - weekend_summary["Weekday"]
        
        report.append("## Weekend vs Weekday Performance")
        report.append("```")
        report.append(weekend_summary.round(2).to_string())
        report.append("```")
        report.append("")
        
        # Demand level analysis
        demand_summary = df.groupby(["method_display", "demand_level"])["on_time_delivery_rate"].mean().unstack()
        
        report.append("## Performance by Demand Level")
        report.append("```")
        report.append(demand_summary.round(2).to_string())
        report.append("```")
        report.append("")
        
        # District complexity analysis
        complexity_summary = df.groupby(["method_display", "district_complexity"])["on_time_delivery_rate"].mean().unstack()
        
        report.append("## Performance by District Complexity")
        report.append("```")
        report.append(complexity_summary.round(2).to_string())
        report.append("```")
        report.append("")
        
        # Key findings
        report.append("## Key Findings")
        
        # Find best performing conditions for each algorithm
        for method in methods:
            method_data = df[df["method_display"] == method]
            
            # Best day type
            best_day_type = "Weekend" if method_data[method_data["is_weekend"]]["on_time_delivery_rate"].mean() > \
                           method_data[~method_data["is_weekend"]]["on_time_delivery_rate"].mean() else "Weekday"
            
            # Best demand level
            demand_performance = method_data.groupby("demand_level")["on_time_delivery_rate"].mean()
            best_demand = demand_performance.idxmax()
            
            # Best complexity level
            complexity_performance = method_data.groupby("district_complexity")["on_time_delivery_rate"].mean()
            best_complexity = complexity_performance.idxmax()
            
            report.append(f"### {method}")
            report.append(f"- Best day type: {best_day_type}")
            report.append(f"- Best demand level: {best_demand}")
            report.append(f"- Best complexity level: {best_complexity}")
            report.append("")
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / "demand_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Saved report: {report_path}")
        return report_text

    def run_full_analysis(self):
        """Run complete demand-based performance analysis."""
        print("🔍 Starting Demand-Based Performance Analysis")
        print("=" * 50)
        
        # Load data
        df = self.load_data()
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        self.create_weekend_weekday_comparison(df)
        self.create_demand_level_analysis(df)
        self.create_district_complexity_analysis(df)
        self.create_temporal_patterns_analysis(df)
        self.create_demand_correlation_matrix(df)
        
        # Generate report
        print("\n📄 Generating report...")
        self.generate_demand_summary_report(df)
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
        return df


def main():
    """Main execution function."""
    analyzer = DemandPerformanceAnalyzer()
    df = analyzer.run_full_analysis()
    return df


if __name__ == "__main__":
    main()