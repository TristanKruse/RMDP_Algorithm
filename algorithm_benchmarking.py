import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
from training.train import run_test_episode
from environment.meituan_data.meituan_data_config import MeituanDataConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def quick_benchmark_all_datasets():
    """
    Quick benchmarking: Run each method ONCE on each of the 176 datasets.
    This gives us a fast overview of relative performance across all scenarios.
    """

    # Define the four methods
    methods = {
        "fastest_aca": {
            "solver": "aca",
            "aca_buffer": 999,  # Max buffer = effectively fastest assignment
            "description": "ACA with maximum buffer (fastest assignment)",
        },
        "aca_17": {"solver": "aca", "aca_buffer": 17, "description": "ACA with buffer 17 (tuned)"},
        "rl_aca": {"solver": "rl_aca", "aca_buffer": 17, "description": "Latest RL-ACA trained model"},
    }

    # Define all 176 datasets
    districts = list(range(1, 23))  # Districts 1 to 22
    days = [f"202210{day:02d}" for day in range(17, 25)]  # October 17 to October 24, 2022

    # Storage for results
    all_results = []
    dataset_info = []
    failed_datasets = []

    total_datasets = len(districts) * len(days)
    logger.info(f"🚀 Starting Quick Benchmarking")
    logger.info(f"📊 Testing {len(methods)} methods on {total_datasets} datasets")
    logger.info(f"⚡ Single run per method per dataset for speed")
    logger.info("=" * 60)

    dataset_count = 0

    # Iterate through all datasets
    for district in districts:
        for day in days:
            dataset_count += 1
            logger.info(f"📍 Dataset {dataset_count}/{total_datasets}: District {district}, Day {day}")

            try:
                # Configure Meituan data with vehicle scaling
                meituan_config = MeituanDataConfig(
                    district_id=district,
                    day=day,
                    # Enable real data usage
                    use_restaurant_positions=True,
                    use_vehicle_count=True,
                    use_vehicle_positions=True,
                    use_service_area=True,
                    use_deadlines=True,
                    # Use real order data
                    order_generation_mode="replay",
                    temporal_pattern=None,
                    simulation_start_hour=10,
                    simulation_duration_hours=12,
                    # Vehicle scaling
                    scale_vehicles_to_restaurants=True,
                    vehicles_per_restaurant_ratio=0.54,
                )

                # Collect dataset characteristics
                dataset_info.append(
                    {
                        "district": district,
                        "day": day,
                        "num_restaurants": meituan_config.get_restaurant_count(),
                        "original_vehicles": meituan_config.get_vehicle_count(),
                        "scaled_vehicles": meituan_config.get_scaled_vehicle_count(),
                    }
                )

            except Exception as e:
                logger.warning(f"❌ Failed to load dataset District {district}, Day {day}: {e}")
                failed_datasets.append(f"District_{district}_Day_{day}")
                continue

            # Test each method on this dataset
            for method_name, method_config in methods.items():
                logger.info(f"  🤖 Testing: {method_name}")

                try:
                    # Single run with fixed seed for reproducibility
                    seed = district * 1000 + int(day[-2:]) * 100 + 42

                    # Prepare run parameters
                    run_params = {
                        "solver_name": method_config["solver"],
                        "meituan_config": meituan_config,
                        "seed": seed,
                        "reposition_idle_vehicles": False,
                        "visualize": False,
                        "warmup_duration": 0,
                        "exploration_rate": 0,
                        "save_results_to_disk": False,
                        "training_mode": False,  # Evaluation mode only
                    }

                    # Add method-specific parameters
                    if "aca_buffer" in method_config:
                        run_params["aca_buffer"] = method_config["aca_buffer"]

                    # Run the simulation
                    episode_stats = run_test_episode(**run_params)

                    # Calculate key metrics
                    total_delay = sum(episode_stats.get("delay_values", []))
                    late_orders_count = len(episode_stats.get("late_orders", set()))
                    total_orders = episode_stats.get("total_orders", 0)
                    orders_delivered = episode_stats.get("orders_delivered", 0)

                    # Calculate on-time delivery rate
                    on_time_orders = orders_delivered - late_orders_count
                    on_time_delivery_rate = (on_time_orders / total_orders * 100) if total_orders > 0 else 0

                    # Calculate other metrics
                    avg_delay_late_orders = total_delay / late_orders_count if late_orders_count > 0 else 0
                    active_period_idle_rate = episode_stats.get("active_period_idle_rate", 0) * 100
                    total_distance = episode_stats.get("total_distance", 0)
                    avg_distance_per_order = total_distance / total_orders if total_orders > 0 else 0

                    # Store results
                    result = {
                        "district": district,
                        "day": day,
                        "method": method_name,
                        "total_delay": total_delay,
                        "on_time_delivery_rate": on_time_delivery_rate,
                        "active_period_idle_rate": active_period_idle_rate,
                        "avg_delay_late_orders": avg_delay_late_orders,
                        "max_delay": episode_stats.get("max_delay", 0),
                        "avg_distance_per_order": avg_distance_per_order,
                        "total_orders": total_orders,
                        "orders_delivered": orders_delivered,
                        "late_orders_count": late_orders_count,
                        "undelivered_orders": total_orders - orders_delivered,
                    }

                    all_results.append(result)

                    # Quick performance summary
                    logger.info(
                        f"    ✅ Delay: {total_delay:.0f}min, On-time: {on_time_delivery_rate:.1f}%, Delivered: {orders_delivered}/{total_orders}"
                    )

                except Exception as e:
                    logger.error(f"    ❌ {method_name} failed: {e}")
                    continue

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        dataset_info_df = pd.DataFrame(dataset_info)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "data/simulation_results"
        os.makedirs(results_dir, exist_ok=True)

        # Save main results
        csv_path = os.path.join(results_dir, f"quick_benchmark_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"💾 Saved results to {csv_path}")

        # Save dataset info
        dataset_info_path = os.path.join(results_dir, f"dataset_characteristics_{timestamp}.csv")
        dataset_info_df.to_csv(dataset_info_path, index=False)

        # Analyze results
        analyze_quick_results(results_df, dataset_info_df, timestamp, failed_datasets)

        return results_df
    else:
        logger.error("❌ No results collected!")
        return None


def analyze_quick_results(results_df, dataset_info_df, timestamp, failed_datasets):
    """
    Analyze and visualize the quick benchmarking results.
    """
    logger.info(f"\n📊 ANALYZING RESULTS")
    logger.info("=" * 60)

    # Overall performance summary
    logger.info("🏆 OVERALL PERFORMANCE SUMMARY")
    logger.info("-" * 40)

    method_summary = (
        results_df.groupby("method")
        .agg(
            {
                "total_delay": ["mean", "std", "min", "max"],
                "on_time_delivery_rate": ["mean", "std", "min", "max"],
                "orders_delivered": "mean",
                "total_orders": "mean",
            }
        )
        .round(2)
    )

    for method in results_df["method"].unique():
        method_data = results_df[results_df["method"] == method]
        avg_delay = method_data["total_delay"].mean()
        avg_on_time = method_data["on_time_delivery_rate"].mean()
        wins = len(
            method_data[
                method_data["total_delay"] == method_data.groupby(["district", "day"])["total_delay"].transform("min")
            ]
        )

        logger.info(f"{method:<15}: Avg Delay={avg_delay:.0f}min, On-time={avg_on_time:.1f}%, Wins={wins}")

    # Check if fastest is always best
    logger.info(f"\n🥇 IS FASTEST ALWAYS BEST?")
    logger.info("-" * 40)

    # For each dataset, find the method with lowest delay
    best_by_dataset = results_df.loc[results_df.groupby(["district", "day"])["total_delay"].idxmin()]
    fastest_wins = len(best_by_dataset[best_by_dataset["method"] == "fastest_aca"])
    total_datasets = len(best_by_dataset)

    logger.info(f"Fastest ACA wins: {fastest_wins}/{total_datasets} datasets ({fastest_wins/total_datasets*100:.1f}%)")

    method_wins = best_by_dataset["method"].value_counts()
    for method, wins in method_wins.items():
        logger.info(f"{method}: {wins} wins ({wins/total_datasets*100:.1f}%)")

    # Dataset characteristics
    if not dataset_info_df.empty:
        logger.info(f"\n📋 DATASET CHARACTERISTICS")
        logger.info("-" * 40)
        logger.info(f"Total datasets processed: {len(dataset_info_df)}")
        logger.info(f"Average restaurants per district: {dataset_info_df['num_restaurants'].mean():.0f}")
        logger.info(f"Average scaled vehicles per district: {dataset_info_df['scaled_vehicles'].mean():.0f}")
        logger.info(f"Vehicle scaling ratio: 0.54 (constant)")

    if failed_datasets:
        logger.info(f"\n⚠️  FAILED DATASETS: {len(failed_datasets)}")
        for failed in failed_datasets[:5]:  # Show first 5
            logger.info(f"  - {failed}")
        if len(failed_datasets) > 5:
            logger.info(f"  ... and {len(failed_datasets) - 5} more")

    # Create visualizations
    create_quick_visualizations(results_df, dataset_info_df, timestamp)


def create_quick_visualizations(results_df, dataset_info_df, timestamp):
    """
    Create visualizations for quick benchmarking results.
    """
    viz_dir = os.path.join("data/simulation_results", "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. Method Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Total Delay Comparison
    sns.boxplot(data=results_df, x="method", y="total_delay", ax=axes[0, 0])
    axes[0, 0].set_title("Total Delay Distribution by Method")
    axes[0, 0].set_ylabel("Total Delay (minutes)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # On-time Rate Comparison
    sns.boxplot(data=results_df, x="method", y="on_time_delivery_rate", ax=axes[0, 1])
    axes[0, 1].set_title("On-time Delivery Rate by Method")
    axes[0, 1].set_ylabel("On-time Rate (%)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Performance by District
    district_summary = results_df.groupby(["district", "method"])["total_delay"].mean().reset_index()
    sns.lineplot(data=district_summary, x="district", y="total_delay", hue="method", ax=axes[1, 0])
    axes[1, 0].set_title("Average Total Delay by District")
    axes[1, 0].set_ylabel("Total Delay (minutes)")

    # Method Wins Distribution
    best_by_dataset = results_df.loc[results_df.groupby(["district", "day"])["total_delay"].idxmin()]
    win_counts = best_by_dataset["method"].value_counts()
    axes[1, 1].pie(win_counts.values, labels=win_counts.index, autopct="%1.1f%%")
    axes[1, 1].set_title("Method Performance (% of datasets won)")

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"quick_benchmark_overview_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Detailed Performance Heatmap
    plt.figure(figsize=(14, 10))

    # Create heatmap of total delay by district and method
    heatmap_data = results_df.pivot_table(values="total_delay", index="district", columns="method", aggfunc="mean")

    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Total Delay (minutes)"})
    plt.title("Total Delay Heatmap: Districts vs Methods")
    plt.xlabel("Method")
    plt.ylabel("District")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"performance_heatmap_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"📊 Visualizations saved to {viz_dir}")


if __name__ == "__main__":
    logger.info("🚀 Quick Algorithm Benchmarking - Single Run Per Dataset")
    logger.info("Testing whether the fastest algorithm consistently outperforms others")

    # Create necessary directories
    os.makedirs("data/simulation_results", exist_ok=True)
    os.makedirs("data/simulation_results/visualizations", exist_ok=True)

    # Run quick benchmarking
    results = quick_benchmark_all_datasets()

    if results is not None:
        logger.info("\n🎉 Quick benchmarking completed successfully!")
        logger.info("📁 Check data/simulation_results/ for detailed results and visualizations")

        # Quick answer to your question
        best_by_dataset = results.loc[results.groupby(["district", "day"])["total_delay"].idxmin()]
        fastest_wins = len(best_by_dataset[best_by_dataset["method"] == "fastest_aca"])
        total_datasets = len(best_by_dataset)

        logger.info(f"\n🏆 ANSWER TO YOUR QUESTION:")
        logger.info(
            f"Is fastest algorithm always outperforming? {fastest_wins}/{total_datasets} datasets ({fastest_wins/total_datasets*100:.1f}%)"
        )

        if fastest_wins == total_datasets:
            logger.info("✅ YES! Fastest ACA wins on ALL datasets")
        elif fastest_wins > total_datasets * 0.8:
            logger.info("⚡ MOSTLY! Fastest ACA wins on most datasets")
        else:
            logger.info("🤔 NO! Other methods sometimes perform better")

    else:
        logger.error("❌ Benchmarking failed!")
