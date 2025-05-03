# tune_vehicle_buffer.py
import os
import logging
from train import run_test_episode  # Import from the original train.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Lunch dinner pattern for 12 hours, starting at 10 a.m
lunch_dinner_pattern = {
    'type': 'hourly',
    'hourly_rates': {
        0: 1.59,  # Maps to 10:00
        1: 4.38,  # Maps to 11:00
        2: 2.38,  # Maps to 12:00
        3: 1.15,  # Maps to 13:00
        4: 0.84,  # Maps to 14:00
        5: 0.76,  # Maps to 15:00
        6: 1.00,  # Maps to 16:00
        7: 2.25,  # Maps to 17:00
        8: 2.86,  # Maps to 18:00
        9: 1.94,  # Maps to 19:00
        10: 1.23, # Maps to 20:00
        11: 0.85,  # Maps to 21:00
        12: 0.55, # Maps to 22:00
        13: 0.37,  # Maps to 23:00
        14: 0.21, # Maps to 00:00
        15: 0.13, # Maps to 01:00
        16: 0.08, # Maps to 02:00
        17: 0.05, # Maps to 03:00
        18: 0.04, # Maps to 04:00
        19: 0.04, # Maps to 05:00
        20: 0.12, # Maps to 06:00
        21: 0.24, # Maps to 07:00
        22: 0.41, # Maps to 08:00
        23: 0.53  # Maps to 09:00
    }
}

def tune_vehicle_buffer():
    """
    Tunes the ACA vehicle buffer using sample average approximation (SAA).
    Tests buffer sizes from 0 to 35, running 100 episodes per size.
    Tracks multiple metrics (fill rate, total delay, number of late orders, etc.).
    Selects the buffer size with the lowest total delay.
    Creates a bar chart of total delay vs. buffer sizes.
    """
    # Simulation environment configuration (aligned with Section 5.3)
    env_config = {
        "num_restaurants": 20,
        "num_vehicles": 10,      # 0.5 * num_restaurants
        "mean_prep_time": 13.4,
        "prep_time_var": 2.0,
        "delivery_window": 15,   # Reduced to produce more late orders
        "simulation_duration": 600,  # 10 hours (600 minutes)
        "cooldown_duration": 0,
        "mean_interarrival_time": 8, # Scaled to fit couriers
        "service_area_dimensions": (6.0, 6.0),
        "downtown_concentration": 0.71,
        "service_time": 4.0,
        "movement_per_step": (11.5 / 60) / 1.0,  # Same as in train.py
        "visualize": False,
        "update_interval": 0.01,
        "reposition_idle_vehicles": True,
        "seed": None,
        "demand_pattern": lunch_dinner_pattern,  # Use the lunch_dinner_pattern
    }

    # Buffer sizes to test: 0 to 35
    buffer_sizes = list(range(36))  # [0, 1, 2, ..., 35]
    num_episodes = 100  # Number of episodes per buffer size

    # Store results: buffer size -> list of metrics per episode
    metrics_to_track = [
        "on_time_delivery_rate",  # Fill rate
        "total_delay",            # Total delay
        "late_orders",            # Number of late orders
        "percentage_late_orders", # Percentage of late orders
        "avg_delay_late_orders",  # Average delay of late orders
        "bundling_rate",          # Bundling efficiency
        "avg_distance_per_order", # Average distance per order
        "active_period_idle_rate" # Active period idle rate
    ]
    results = {b: {metric: [] for metric in metrics_to_track} for b in buffer_sizes}

    # Run SAA for each buffer size
    for buffer_size in buffer_sizes:
        logger.info(f"Testing buffer size: {buffer_size}")
        for episode in range(num_episodes):
            # Use a unique seed for each episode
            seed = episode + buffer_size * num_episodes

            # Run episode with ACA solver
            stats = run_test_episode(
                solver_name="aca",
                meituan_config=None,
                seed=seed,
                reposition_idle_vehicles=True,
                visualize=False,
                warmup_duration=0,
                save_rl_model=False,
                rl_model_path=None,
                save_results_to_disk=False,
                env_config=env_config,
                aca_buffer=buffer_size
            )

            # Use precomputed metrics from train.py
            on_time_rate = stats["on_time_delivery_rate"]
            late_orders = len(stats["late_orders"])
            percentage_late = stats["percentage_late_orders"]
            avg_delay = stats["avg_delay_late_orders"]
            avg_distance = stats["avg_distance_per_order"]
            bundling_rate = stats["bundling_rate"] if "bundling_rate" in stats else 0

            # Store metrics
            results[buffer_size]["on_time_delivery_rate"].append(on_time_rate)
            results[buffer_size]["total_delay"].append(stats["total_delay"])
            results[buffer_size]["late_orders"].append(late_orders)
            results[buffer_size]["percentage_late_orders"].append(percentage_late)
            results[buffer_size]["avg_delay_late_orders"].append(avg_delay)
            results[buffer_size]["bundling_rate"].append(bundling_rate)
            results[buffer_size]["avg_distance_per_order"].append(avg_distance)
            results[buffer_size]["active_period_idle_rate"].append(stats["active_period_idle_rate"])

            if episode % 10 == 0:
                logger.info(
                    f"  Episode {episode}/{num_episodes}, "
                    f"Total Orders: {stats['total_orders']}, "
                    f"Delivered: {stats['orders_delivered']}, "
                    f"Late Orders: {late_orders}, "
                    f"Total Delay: {stats['total_delay']:.2f} minutes, "
                    f"Fill Rate: {on_time_rate:.1f}%"
                )

    # Compute sample averages for each metric
    sample_averages = {b: {metric: np.mean(values) for metric, values in metrics.items()} for b, metrics in results.items()}

    # Select the best buffer size based on total delay
    best_buffer = min(sample_averages, key=lambda b: sample_averages[b]["total_delay"])
    best_metrics = sample_averages[best_buffer]

    logger.info(f"\nBest buffer size: {best_buffer}")
    logger.info(f"Metrics for best buffer size:")
    for metric, value in best_metrics.items():
        logger.info(f"  {metric}: {value:.2f}")

    # Create bar chart of total delay vs. buffer sizes
    total_delays = [sample_averages[b]["total_delay"] for b in buffer_sizes]
    plt.figure(figsize=(10, 6))
    plt.bar(buffer_sizes, total_delays, color='skyblue')
    plt.xlabel('Buffer Size')
    plt.ylabel('Total Delay (minutes)')
    plt.title('Total Delay vs. ACA Vehicle Buffer Size')
    plt.xticks(buffer_sizes)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig("data/simulation_results/buffer_total_delay_bar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved bar chart to data/simulation_results/buffer_total_delay_bar_chart.png")

    # Save results to CSV with all metrics
    results_df = pd.DataFrame([
        {
            "buffer_size": b,
            **{metric: np.mean(values) for metric, values in metrics.items()},
            **{f"{metric}_std": np.std(values) for metric, values in metrics.items()}
        }
        for b, metrics in results.items()
    ])
    results_df.to_csv("data/simulation_results/vehicle_buffer_tuning.csv", index=False)
    logger.info("Saved tuning results to data/simulation_results/vehicle_buffer_tuning.csv")

    return best_buffer, best_metrics

if __name__ == "__main__":
    best_buffer, best_metrics = tune_vehicle_buffer()
    logger.info(f"Final Result: Selected buffer size {best_buffer} with metrics: {best_metrics}")