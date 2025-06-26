import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class AdvancedBenchmarkVisualizer:
    """
    Advanced visualization suite for algorithm benchmarking results.
    Creates publication-ready comparative charts and analysis plots.
    """

    def __init__(self, results_dir: str = "data/simulation_results"):
        self.results_dir = Path(results_dir)
        self.viz_dir = self.results_dir / "advanced_visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes for methods
        self.method_colors = {
            "RL-ACA (4-Phase)": "#F18F01",       # Orange - 4-phase RL model
            "RL-ACA (1-Phase)": "#FF6B6B",       # Red - 1-phase RL model
            "Fastest ACA": "#2E86AB",            # Blue - Baseline
            "ACA (Buffer=17)": "#A23B72",        # Purple - Heuristic
            "Meituan Baseline": "#63B600"        # Green - Meituan's method
        }

        self.method_labels = {
            "RL-ACA (4-Phase)": "RL-ACA (4-Phase)",
            "RL-ACA (1-Phase)": "RL-ACA (1-Phase)",
            "Fastest ACA": "Fastest ACA",
            "ACA (Buffer=17)": "ACA (Buffer=17)",
            "Meituan Baseline": "Meituan Baseline",
        }

    def load_data(self, csv_path: str = None) -> pd.DataFrame:
        """Load benchmark results data."""
        if csv_path is None:
            # Look for fixed filenames first, then timestamped files
            benchmark_file = self.results_dir / "benchmark_results.csv"
            filtered_file = self.results_dir / "fastest_aca_filtered_results.csv"
            
            if filtered_file.exists():
                csv_path = filtered_file
                print(f"📊 Using FILTERED data: {csv_path.name}")
            elif benchmark_file.exists():
                csv_path = benchmark_file
                print(f"📊 Using benchmark data: {csv_path.name}")
            else:
                # Fall back to timestamped files
                filtered_files = list(self.results_dir.glob("fastest_aca_filtered_results_*.csv"))
                csv_files = list(self.results_dir.glob("benchmark_results_*.csv"))
                combined_files = list(self.results_dir.glob("combined_with_baseline_*.csv"))

                # Prioritize filtered files if they exist
                if filtered_files:
                    csv_path = max(filtered_files, key=lambda x: x.stat().st_mtime)
                    print(f"📊 Using FILTERED data: {csv_path.name}")
                else:
                    # Fall back to original files
                    all_files = csv_files + combined_files

                    if not all_files:
                        raise FileNotFoundError(
                            "No benchmark results found! Looking for 'benchmark_results.csv', 'fastest_aca_filtered_results.csv', or timestamped versions"
                        )
                    csv_path = max(all_files, key=lambda x: x.stat().st_mtime)
                    print(f"📊 Using timestamped data: {csv_path.name}")

        df = pd.read_csv(csv_path)

        # Filter for only the methods we want to visualize
        desired_methods = [
            "rl_aca",              # Old RL model 
            "rl_aca_phase1_final", # New 4-phase RL model
            "fastest_aca",         # Fastest ACA baseline
            "aca_17",              # ACA heuristic with buffer=17
        ]
        
        # Filter the dataframe to only include desired methods
        df = df[df["method"].isin(desired_methods)]
        print(f"📊 Filtered to {len(desired_methods)} methods: {', '.join(desired_methods)}")
        print(f"📊 Available methods in data: {', '.join(df['method'].unique())}")

        # Handle method name mapping for your specific data
        method_mapping = {
            "rl_aca": "RL-ACA (4-Phase)",
            "rl_aca_phase1_final": "RL-ACA (1-Phase)",
            "fastest_aca": "Fastest ACA",
            "aca_17": "ACA (Buffer=17)",
        }

        # Add readable method labels
        df["method_label"] = df["method"].map(method_mapping).fillna(df["method"])

        return df

    def create_performance_comparison_grid(self, df: pd.DataFrame, save_name: str = "performance_grid"):
        """
        Create a comprehensive grid comparing all methods across key metrics.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Algorithm Performance Comparison Across All Metrics", fontsize=16, fontweight="bold")

        metrics = [
            ("total_delay", "Total Delay (minutes)", "lower_better"),
            ("on_time_delivery_rate", "On-Time Delivery Rate (%)", "higher_better"),
            ("avg_delay_late_orders", "Avg Delay (Late Orders)", "lower_better"),
            ("max_delay", "Maximum Delay (minutes)", "lower_better"),
            ("avg_distance_per_order", "Avg Distance per Order (km)", "lower_better"),
            ("active_period_idle_rate", "Vehicle Idle Rate (%)", "lower_better"),
        ]

        for idx, (metric, title, direction) in enumerate(metrics):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]

            # Create violin plot with box plot overlay
            sns.violinplot(data=df, x="method_label", y=metric, ax=ax, inner=None, alpha=0.6)
            sns.boxplot(
                data=df, x="method_label", y=metric, ax=ax, width=0.3, boxprops=dict(alpha=0.7), showfliers=False
            )

            # Highlight best performing method
            method_means = df.groupby("method_label")[metric].mean()
            if direction == "lower_better":
                best_method = method_means.idxmin()
                best_color = "lightgreen"
            else:
                best_method = method_means.idxmax()
                best_color = "lightgreen"

            # Add background color for best method
            best_idx = list(method_means.index).index(best_method)
            ax.axvspan(best_idx - 0.4, best_idx + 0.4, alpha=0.2, color=best_color)

            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)

            # Add mean value annotations
            for i, method in enumerate(method_means.index):
                mean_val = method_means[method]
                ax.annotate(
                    f"{mean_val:.1f}",
                    xy=(i, mean_val),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontweight="bold",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        plt.tight_layout()
        plt.savefig(self.viz_dir / f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(self.viz_dir / f"{save_name}.pdf", bbox_inches="tight")
        plt.close()

    def create_performance_radar_chart(self, df: pd.DataFrame, save_name: str = "performance_radar"):
        """
        Create radar chart comparing normalized performance across methods.
        """
        # Calculate mean performance for each method
        method_stats = (
            df.groupby("method_label")
            .agg(
                {
                    "total_delay": "mean",
                    "on_time_delivery_rate": "mean",
                    "avg_distance_per_order": "mean",
                    "max_delay": "mean",
                    "active_period_idle_rate": "mean",
                }
            )
            .round(2)
        )

        # Normalize metrics (0-1 scale, where 1 is best)
        normalized_stats = method_stats.copy()

        # For metrics where lower is better, invert the scale
        lower_better = ["total_delay", "avg_distance_per_order", "max_delay", "active_period_idle_rate"]
        for metric in lower_better:
            max_val = method_stats[metric].max()
            min_val = method_stats[metric].min()
            # Invert: best (lowest) becomes 1, worst (highest) becomes 0
            normalized_stats[metric] = (max_val - method_stats[metric]) / (max_val - min_val)

        # For on_time_delivery_rate, higher is better (normalize normally)
        metric = "on_time_delivery_rate"
        max_val = method_stats[metric].max()
        min_val = method_stats[metric].min()
        normalized_stats[metric] = (method_stats[metric] - min_val) / (max_val - min_val)

        # Create radar chart
        labels = ["Low Delay", "High On-Time Rate", "Short Distance", "Low Max Delay", "Low Idle Rate"]
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        for method in normalized_stats.index:
            values = normalized_stats.loc[method].tolist()
            values += values[:1]  # Complete the circle

            color = self.method_colors.get(method.split(" ")[0].lower(), "#333333")
            ax.plot(angles, values, "o-", linewidth=2, label=method, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
        ax.grid(True)

        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        plt.title("Normalized Performance Comparison\n(1.0 = Best Performance)", fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()
        plt.savefig(self.viz_dir / f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(self.viz_dir / f"{save_name}.pdf", bbox_inches="tight")
        plt.close()

    def create_district_performance_heatmap(
        self, df: pd.DataFrame, metric: str = "total_delay", save_name: str = "district_heatmap"
    ):
        """
        Create heatmap showing performance across districts for each method.
        """
        # Pivot data for heatmap
        pivot_data = df.pivot_table(values=metric, index="district", columns="method_label", aggfunc="mean")

        # Calculate relative performance (percentage difference from best method per district)
        relative_performance = pivot_data.copy()
        for district in pivot_data.index:
            if metric in ["total_delay", "avg_distance_per_order", "max_delay"]:
                # Lower is better
                best_val = pivot_data.loc[district].min()
                relative_performance.loc[district] = (pivot_data.loc[district] - best_val) / best_val * 100
            else:
                # Higher is better
                best_val = pivot_data.loc[district].max()
                relative_performance.loc[district] = (best_val - pivot_data.loc[district]) / best_val * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Absolute values heatmap
        sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax1, cbar_kws={"label": metric})
        ax1.set_title(f'Absolute {metric.replace("_", " ").title()} by District', fontweight="bold")
        ax1.set_xlabel("Method")
        ax1.set_ylabel("District")

        # Relative performance heatmap
        sns.heatmap(
            relative_performance,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            center=0,
            ax=ax2,
            cbar_kws={"label": "% Worse than Best"},
        )
        ax2.set_title(f"Relative Performance by District\n(% worse than best method)", fontweight="bold")
        ax2.set_xlabel("Method")
        ax2.set_ylabel("")

        plt.tight_layout()
        plt.savefig(self.viz_dir / f"{save_name}_{metric}.png", dpi=300, bbox_inches="tight")
        plt.savefig(self.viz_dir / f"{save_name}_{metric}.pdf", bbox_inches="tight")
        plt.close()

    def create_performance_ranking_chart(self, df: pd.DataFrame, save_name: str = "performance_ranking"):
        """
        Create chart showing how often each method ranks 1st, 2nd, 3rd across datasets.
        """
        metrics = ["total_delay", "on_time_delivery_rate", "avg_distance_per_order"]
        metric_titles = ["Total Delay", "On-Time Delivery Rate", "Average Distance"]
        lower_better = [True, False, True]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Method Rankings Across All Datasets", fontsize=16, fontweight="bold")

        for idx, (metric, title, is_lower_better) in enumerate(zip(metrics, metric_titles, lower_better)):
            ax = axes[idx]

            # Calculate rankings for each dataset
            rankings = []
            for (district, day), group in df.groupby(["district", "day"]):
                method_performance = group.groupby("method_label")[metric].mean()

                if is_lower_better:
                    ranked = method_performance.rank(method="min")
                else:
                    ranked = method_performance.rank(method="min", ascending=False)

                for method, rank in ranked.items():
                    rankings.append({"method": method, "rank": int(rank), "metric": metric})

            rankings_df = pd.DataFrame(rankings)

            # Create stacked bar chart
            rank_counts = rankings_df.groupby(["method", "rank"]).size().unstack(fill_value=0)

            # Ensure all ranks are present
            for rank in [1, 2, 3]:
                if rank not in rank_counts.columns:
                    rank_counts[rank] = 0

            rank_counts = rank_counts[[1, 2, 3]]  # Ensure order

            # Plot stacked bars
            rank_counts.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                color=["#2E8B57", "#DAA520", "#CD853F"],  # Green, Gold, Bronze
                alpha=0.8,
            )

            ax.set_title(f"{title}\n(Ranking Distribution)", fontweight="bold")
            ax.set_xlabel("Method")
            ax.set_ylabel("Number of Datasets")
            ax.legend(["1st Place", "2nd Place", "3rd Place"], title="Rank")
            ax.tick_params(axis="x", rotation=45)

            # Add percentage annotations
            total_datasets = len(df.groupby(["district", "day"]))
            for i, method in enumerate(rank_counts.index):
                first_place_count = rank_counts.loc[method, 1]
                percentage = (first_place_count / total_datasets) * 100
                ax.annotate(
                    f"{percentage:.1f}%",
                    xy=(i, first_place_count / 2),
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                )

        plt.tight_layout()
        plt.savefig(self.viz_dir / f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(self.viz_dir / f"{save_name}.pdf", bbox_inches="tight")
        plt.close()

    def create_variability_analysis(self, df: pd.DataFrame, save_name: str = "variability_analysis"):
        """
        Analyze and visualize performance variability across methods.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Performance Variability Analysis", fontsize=16, fontweight="bold")

        # 1. Coefficient of Variation comparison
        ax1 = axes[0, 0]
        cv_data = []
        for method in df["method_label"].unique():
            method_data = df[df["method_label"] == method]
            for metric in ["total_delay", "on_time_delivery_rate"]:
                cv = method_data[metric].std() / method_data[metric].mean()
                cv_data.append({"method": method, "metric": metric, "cv": cv})

        cv_df = pd.DataFrame(cv_data)
        sns.barplot(data=cv_df, x="metric", y="cv", hue="method", ax=ax1)
        ax1.set_title("Coefficient of Variation\n(Lower = More Consistent)", fontweight="bold")
        ax1.set_ylabel("Coefficient of Variation")
        ax1.tick_params(axis="x", rotation=45)

        # 2. Performance spread by district
        ax2 = axes[0, 1]
        district_std = df.groupby(["district", "method_label"])["total_delay"].std().reset_index()
        sns.boxplot(data=district_std, x="method_label", y="total_delay", ax=ax2)
        ax2.set_title(
            "Performance Variability Across Districts\n(Standard Deviation of Total Delay)", fontweight="bold"
        )
        ax2.set_ylabel("Std Dev of Total Delay")
        ax2.tick_params(axis="x", rotation=45)

        # 3. Day-to-day consistency
        ax3 = axes[1, 0]
        day_performance = df.groupby(["day", "method_label"])["total_delay"].mean().reset_index()
        for method in df["method_label"].unique():
            method_data = day_performance[day_performance["method_label"] == method]
            ax3.plot(method_data["day"], method_data["total_delay"], marker="o", label=method, linewidth=2)
        ax3.set_title("Day-to-Day Performance Consistency", fontweight="bold")
        ax3.set_ylabel("Average Total Delay")
        ax3.set_xlabel("Day")
        ax3.legend()
        ax3.tick_params(axis="x", rotation=45)

        # 4. Performance distribution comparison
        ax4 = axes[1, 1]
        for method in df["method_label"].unique():
            method_data = df[df["method_label"] == method]["total_delay"]
            ax4.hist(method_data, alpha=0.6, label=method, bins=20, density=True)
        ax4.set_title("Total Delay Distribution", fontweight="bold")
        ax4.set_xlabel("Total Delay (minutes)")
        ax4.set_ylabel("Density")
        ax4.legend()

        plt.tight_layout()
        plt.savefig(self.viz_dir / f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(self.viz_dir / f"{save_name}.pdf", bbox_inches="tight")
        plt.close()

    def create_root_cause_analysis_plots(self, df: pd.DataFrame, save_name: str = "root_cause_analysis"):
        """
        Create visualizations to help identify root causes of poor RL performance.
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Root Cause Analysis: Why is RL-ACA Underperforming?", fontsize=16, fontweight="bold")

        # 1. Performance vs Dataset Size (proxy: total delay)
        ax1 = axes[0, 0]
        dataset_complexity = (
            df.groupby(["district", "day", "method_label"])
            .agg({"total_delay": "mean", "avg_distance_per_order": "mean"})
            .reset_index()
        )

        rl_data = dataset_complexity[dataset_complexity["method_label"].str.contains("RL")]
        others_data = dataset_complexity[~dataset_complexity["method_label"].str.contains("RL")]

        ax1.scatter(
            others_data["avg_distance_per_order"],
            others_data["total_delay"],
            alpha=0.6,
            label="Heuristic Methods",
            s=30,
        )
        ax1.scatter(
            rl_data["avg_distance_per_order"], rl_data["total_delay"], alpha=0.8, label="RL-ACA", s=30, color="red"
        )
        ax1.set_xlabel("Average Distance per Order (km)")
        ax1.set_ylabel("Total Delay (minutes)")
        ax1.set_title("Performance vs Problem Complexity", fontweight="bold")
        ax1.legend()

        # 2. Relative performance by district (RL vs best heuristic)
        ax2 = axes[0, 1]
        performance_gaps = []
        for (district, day), group in df.groupby(["district", "day"]):
            methods_performance = group.groupby("method_label")["total_delay"].mean()
            if "RL-ACA (Reinforcement Learning)" in methods_performance.index:
                rl_delay = methods_performance["RL-ACA (Reinforcement Learning)"]
                heuristic_delays = [v for k, v in methods_performance.items() if "RL" not in k]
                if heuristic_delays:
                    best_heuristic = min(heuristic_delays)
                    gap = (rl_delay - best_heuristic) / best_heuristic * 100
                    performance_gaps.append({"district": district, "gap": gap})

        gap_df = pd.DataFrame(performance_gaps)
        if not gap_df.empty:
            ax2.bar(gap_df["district"], gap_df["gap"], color="orangered", alpha=0.7)
            ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax2.set_xlabel("District")
            ax2.set_ylabel("Performance Gap (%)")
            ax2.set_title("RL Performance Gap by District\n(% worse than best heuristic)", fontweight="bold")
            ax2.tick_params(axis="x", rotation=45)

        # 3. Method performance correlation matrix
        ax3 = axes[0, 2]
        correlation_data = df.pivot_table(
            values="total_delay", index=["district", "day"], columns="method_label", aggfunc="mean"
        )
        if not correlation_data.empty:
            corr_matrix = correlation_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax3)
            ax3.set_title("Method Performance Correlation", fontweight="bold")

        # 4. On-time rate vs total delay scatter
        ax4 = axes[1, 0]
        for method in df["method_label"].unique():
            method_data = df[df["method_label"] == method]
            ax4.scatter(method_data["total_delay"], method_data["on_time_delivery_rate"], alpha=0.6, label=method, s=20)
        ax4.set_xlabel("Total Delay (minutes)")
        ax4.set_ylabel("On-Time Delivery Rate (%)")
        ax4.set_title("Delay vs On-Time Rate Trade-off", fontweight="bold")
        ax4.legend()

        # 5. Performance trend by day
        ax5 = axes[1, 1]
        daily_performance = (
            df.groupby(["day", "method_label"])
            .agg({"total_delay": "mean", "on_time_delivery_rate": "mean"})
            .reset_index()
        )

        for method in df["method_label"].unique():
            method_daily = daily_performance[daily_performance["method_label"] == method]
            ax5.plot(range(len(method_daily)), method_daily["total_delay"], marker="o", label=method, linewidth=2)
        ax5.set_xlabel("Day (chronological order)")
        ax5.set_ylabel("Average Total Delay")
        ax5.set_title("Performance Trends Over Time", fontweight="bold")
        ax5.legend()

        # 6. Method efficiency comparison (delay per distance)
        ax6 = axes[1, 2]
        df["efficiency"] = df["total_delay"] / df["avg_distance_per_order"]
        efficiency_stats = df.groupby("method_label")["efficiency"].agg(["mean", "std"]).reset_index()

        x_pos = range(len(efficiency_stats))
        ax6.bar(x_pos, efficiency_stats["mean"], yerr=efficiency_stats["std"], capsize=5, alpha=0.7)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(efficiency_stats["method_label"], rotation=45)
        ax6.set_ylabel("Delay per Distance (min/km)")
        ax6.set_title("Method Efficiency\n(Lower = Better)", fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.viz_dir / f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(self.viz_dir / f"{save_name}.pdf", bbox_inches="tight")
        plt.close()

    def create_executive_summary_dashboard(self, df: pd.DataFrame, save_name: str = "executive_dashboard"):
        """
        Create a high-level executive summary dashboard.
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Main title
        fig.suptitle("Algorithm Benchmarking Executive Dashboard", fontsize=20, fontweight="bold", y=0.95)

        # 1. Overall performance summary (top left, large)
        ax1 = fig.add_subplot(gs[0:2, 0:2])

        summary_stats = (
            df.groupby("method_label")
            .agg({"total_delay": "mean", "on_time_delivery_rate": "mean", "avg_distance_per_order": "mean"})
            .round(1)
        )

        # Create performance score (normalized combination of metrics)
        normalized_summary = summary_stats.copy()
        normalized_summary["total_delay"] = (summary_stats["total_delay"].max() - summary_stats["total_delay"]) / (
            summary_stats["total_delay"].max() - summary_stats["total_delay"].min()
        )
        normalized_summary["on_time_delivery_rate"] = (
            summary_stats["on_time_delivery_rate"] - summary_stats["on_time_delivery_rate"].min()
        ) / (summary_stats["on_time_delivery_rate"].max() - summary_stats["on_time_delivery_rate"].min())
        normalized_summary["avg_distance_per_order"] = (
            summary_stats["avg_distance_per_order"].max() - summary_stats["avg_distance_per_order"]
        ) / (summary_stats["avg_distance_per_order"].max() - summary_stats["avg_distance_per_order"].min())

        normalized_summary["overall_score"] = normalized_summary.mean(axis=1) * 100

        bars = ax1.barh(
            range(len(normalized_summary)), normalized_summary["overall_score"], color=["#2E86AB", "#A23B72", "#F18F01"]
        )
        ax1.set_yticks(range(len(normalized_summary)))
        ax1.set_yticklabels(normalized_summary.index)
        ax1.set_xlabel("Overall Performance Score")
        ax1.set_title("Overall Algorithm Performance\n(Higher = Better)", fontweight="bold", fontsize=14)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.annotate(
                f"{width:.1f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontweight="bold",
            )

        # 2. Key metrics table (top right)
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.axis("tight")
        ax2.axis("off")

        table_data = summary_stats.round(1)
        table = ax2.table(
            cellText=table_data.values,
            rowLabels=table_data.index,
            colLabels=["Avg Delay (min)", "On-Time Rate (%)", "Avg Distance (km)"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax2.set_title("Key Performance Metrics", fontweight="bold", fontsize=12, pad=20)

        # 3. Win rate chart (middle right)
        ax3 = fig.add_subplot(gs[1, 2:4])

        # Calculate win rates
        win_rates = []
        total_datasets = len(df.groupby(["district", "day"]))

        for method in df["method_label"].unique():
            wins = 0
            for (district, day), group in df.groupby(["district", "day"]):
                method_performance = group.groupby("method_label")["total_delay"].mean()
                if method_performance.idxmin() == method:
                    wins += 1
            win_rate = (wins / total_datasets) * 100
            win_rates.append({"method": method, "win_rate": win_rate})

        win_df = pd.DataFrame(win_rates)
        bars = ax3.bar(range(len(win_df)), win_df["win_rate"], color=["#2E86AB", "#A23B72", "#F18F01"])
        ax3.set_xticks(range(len(win_df)))
        ax3.set_xticklabels(win_df["method"], rotation=45)
        ax3.set_ylabel("Win Rate (%)")
        ax3.set_title("Dataset Win Rate\n(Best Total Delay)", fontweight="bold")

        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. Performance distribution (bottom left)
        ax4 = fig.add_subplot(gs[2, 0:2])

        df.boxplot(column="total_delay", by="method_label", ax=ax4)
        ax4.set_title("Total Delay Distribution by Method", fontweight="bold")
        ax4.set_xlabel("Method")
        ax4.set_ylabel("Total Delay (minutes)")
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        # 5. Recommendations text box (bottom right)
        ax5 = fig.add_subplot(gs[2, 2:4])
        ax5.axis("off")

        # Generate recommendations based on data
        best_overall = normalized_summary["overall_score"].idxmax()
        worst_overall = normalized_summary["overall_score"].idxmin()
        best_delay = summary_stats["total_delay"].idxmin()
        best_ontime = summary_stats["on_time_delivery_rate"].idxmax()

        recommendations = f"""
KEY FINDINGS & RECOMMENDATIONS:

✓ Best Overall Performer: {best_overall}
✗ Needs Improvement: {worst_overall}

SPECIFIC INSIGHTS:
• Lowest Delay: {best_delay}
• Best On-Time Rate: {best_ontime}
• Total Datasets: {total_datasets}

NEXT STEPS:
1. Investigate RL-ACA training process
2. Analyze problematic datasets
3. Consider hybrid approaches
4. Validate simulation accuracy
        """

        ax5.text(
            0.05,
            0.95,
            recommendations,
            transform=ax5.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        plt.savefig(self.viz_dir / f"{save_name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(self.viz_dir / f"{save_name}.pdf", bbox_inches="tight")
        plt.close()

    def generate_all_visualizations(self, df: pd.DataFrame = None):
        """
        Generate all visualization types for comprehensive analysis.
        """
        if df is None:
            df = self.load_data()

        print("Generating advanced visualizations...")

        # Generate all visualization types
        self.create_performance_comparison_grid(df)
        print("✓ Performance comparison grid created")

        self.create_performance_radar_chart(df)
        print("✓ Performance radar chart created")

        self.create_district_performance_heatmap(df, "total_delay")
        self.create_district_performance_heatmap(df, "on_time_delivery_rate")
        print("✓ District performance heatmaps created")

        self.create_performance_ranking_chart(df)
        print("✓ Performance ranking chart created")

        self.create_variability_analysis(df)
        print("✓ Variability analysis created")

        self.create_root_cause_analysis_plots(df)
        print("✓ Root cause analysis plots created")

        self.create_executive_summary_dashboard(df)
        print("✓ Executive summary dashboard created")

        print(f"\nAll visualizations saved to: {self.viz_dir}")
        return self.viz_dir


def main():
    """Main function to generate all advanced visualizations."""
    visualizer = AdvancedBenchmarkVisualizer()

    try:
        df = visualizer.load_data()
        print(f"Loaded {len(df)} benchmark records")

        # Generate all visualizations
        viz_dir = visualizer.generate_all_visualizations(df)

        print("\n" + "=" * 60)
        print("ADVANCED VISUALIZATION GENERATION COMPLETE")
        print("=" * 60)
        print(f"All charts saved to: {viz_dir}")
        print("\nGenerated visualizations:")
        print("- Performance comparison grid")
        print("- Performance radar chart")
        print("- District performance heatmaps")
        print("- Performance ranking charts")
        print("- Variability analysis")
        print("- Root cause analysis plots")
        print("- Executive summary dashboard")

    except Exception as e:
        print(f"Visualization generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
