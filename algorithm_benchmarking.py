import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
from train import run_test_episode, MeituanDataConfig, lunch_dinner_pattern

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def benchmark_methods():
    """
    Benchmarks FV, ACA, and RL-ACA across 176 Meituan datasets (22 districts x 8 days).
    Runs 10 simulations per dataset for each method to account for stochasticity.
    Collects KPIs, saves results, and creates visualizations.
    """
    # Define methods to benchmark
    methods = ["fastest", "aca", "rl_aca"]
    
    # Define districts and days
    districts = list(range(1, 23))  # Districts 1 to 22
    days = [f"202210{day:02d}" for day in range(17, 25)]  # October 17 to October 24, 2022
    
    # Define KPIs to collect
    kpis = [
        "on_time_delivery_rate", "active_period_idle_rate", "avg_delay_late_orders",
        "max_delay", "avg_distance_per_order", "total_delay"
    ]
    
    # Number of runs per dataset for each method
    num_runs = 10
    
    # Number of episodes per run (FV: 1 episode, ACA/RL-ACA: 10 episodes)
    episodes_per_run = {
        "fastest": 10,
        "aca": 10,
        "rl_aca": 10
    }
    
    # Initialize results storage
    all_results = []
    
    # Iterate over datasets
    for district in districts:
        for day in days:
            logger.info(f"Processing dataset: District {district}, Day {day}")
            
            # Configure Meituan data for this dataset
            meituan_config = MeituanDataConfig(
                district_id=district,
                day=day,
                use_restaurant_positions=False,
                use_vehicle_count=False,
                use_vehicle_positions=False,
                use_service_area=False,
                use_deadlines=False,
                order_generation_mode="pattern",
                temporal_pattern=lunch_dinner_pattern,
                simulation_start_hour=None,
                simulation_duration_hours=None
            )
            
            # Run simulations for each method
            for method in methods:
                logger.info(f"Running simulations for method: {method}")
                
                # Run multiple simulations to account for stochasticity
                method_results = []
                for run in range(num_runs):
                    run_results = []
                    num_episodes = episodes_per_run[method]
                    
                    for episode in range(num_episodes):
                        seed = (run * num_episodes + episode) + (district * 1000) + (int(day[-2:]) * 10000)  # Unique seed per run, episode, district, day
                        stats = run_test_episode(
                            solver_name=method,
                            meituan_config=meituan_config,
                            seed=seed,
                            reposition_idle_vehicles=False,
                            visualize=False,
                            warmup_duration=0,
                            aca_buffer=17,
                            exploration_rate=0  # For RL-ACA, set exploration rate to 0 for evaluation
                        )
                        
                        # Extract KPIs
                        result = {
                            "district": district,
                            "day": day,
                            "method": method,
                            "run": run,
                            "episode": episode
                        }
                        for kpi in kpis:
                            result[kpi] = stats.get(kpi, 0)
                        run_results.append(result)
                    
                    # Average KPIs over episodes within this run
                    avg_run_result = {
                        "district": district,
                        "day": day,
                        "method": method,
                        "run": run
                    }
                    for kpi in kpis:
                        values = [r[kpi] for r in run_results]
                        avg_run_result[kpi] = sum(values) / len(values) if values else 0
                    method_results.append(avg_run_result)
                
                # Average KPIs over runs for this dataset and method
                avg_result = {
                    "district": district,
                    "day": day,
                    "method": method
                }
                for kpi in kpis:
                    values = [r[kpi] for r in method_results]
                    avg_result[kpi] = sum(values) / len(values) if values else 0
                all_results.append(avg_result)

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "data/simulation_results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"benchmark_results_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved benchmark results to {csv_path}")

    # Aggregate results by district
    district_summary = results_df.groupby(["district", "method"]).mean().reset_index()
    district_summary = district_summary.drop(columns=["day"])

    # Aggregate results by day
    day_summary = results_df.groupby(["day", "method"]).mean().reset_index()
    day_summary = day_summary.drop(columns=["district"])

    # Create visualizations
    create_plots(district_summary, day_summary, timestamp)

    return all_results

def create_plots(district_summary: pd.DataFrame, day_summary: pd.DataFrame, timestamp: str):
    """
    Creates two plots:
    1. Total delay by district (bars) with on-time delivery rate (lines).
    2. Total delay by day (bars) with on-time delivery rate (lines).
    """
    viz_dir = os.path.join("data/simulation_results", "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Plot 1: Total Delay by District
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")
    
    # Bar plot for total delay
    sns.barplot(data=district_summary, x="district", y="total_delay", hue="method")
    
    # Create a second y-axis for on-time delivery rate
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Line plot for on-time delivery rate
    sns.lineplot(data=district_summary, x="district", y="on_time_delivery_rate", hue="method",
                 style="method", markers=True, ax=ax2)
    
    # Customize the plot
    ax1.set_xlabel("District")
    ax1.set_ylabel("Total Delay (minutes)")
    ax2.set_ylabel("On-Time Delivery Rate (%)")
    ax1.set_title("Total Delay and On-Time Delivery Rate by District")
    
    # Adjust legend to avoid overlap
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, title="Method", loc="upper left", bbox_to_anchor=(1.15, 1))
    ax2.get_legend().remove()  # Remove duplicate legend
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"delay_by_district_{timestamp}.png"), dpi=300)
    plt.close()
    logger.info(f"Saved plot: delay_by_district_{timestamp}.png")

    # Plot 2: Total Delay by Day
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Bar plot for total delay
    sns.barplot(data=day_summary, x="day", y="total_delay", hue="method")
    
    # Create a second y-axis for on-time delivery rate
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Line plot for on-time delivery rate
    sns.lineplot(data=day_summary, x="day", y="on_time_delivery_rate", hue="method",
                 style="method", markers=True, ax=ax2)
    
    # Customize the plot
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Total Delay (minutes)")
    ax2.set_ylabel("On-Time Delivery Rate (%)")
    ax1.set_title("Total Delay and On-Time Delivery Rate by Day")
    
    # Adjust legend to avoid overlap
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, title="Method", loc="upper left", bbox_to_anchor=(1.15, 1))
    ax2.get_legend().remove()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"delay_by_day_{timestamp}.png"), dpi=300)
    plt.close()
    logger.info(f"Saved plot: delay_by_day_{timestamp}.png")

if __name__ == "__main__":
    logger.info("Starting benchmarking of methods...")
    results = benchmark_methods()
    logger.info("Benchmarking completed!")