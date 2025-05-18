import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def ensure_visualization_dir() -> str:
    """Ensure visualization directory exists and return its path."""
    viz_dir = os.path.join("data", "visualizations")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    return viz_dir


def visualize_restaurant_distribution(
    episode_stats: Dict[str, Any], solver_name: str, timestamp: Optional[str] = None
) -> None:
    """Visualize the distribution of orders per restaurant."""
    if not episode_stats.get("orders_per_restaurant"):
        logger.warning("No restaurant order data available for visualization")
        return

    # Create figure
    plt.figure(figsize=(12, 6))

    # Get restaurant data
    restaurants = list(episode_stats["orders_per_restaurant"].keys())
    orders = list(episode_stats["orders_per_restaurant"].values())

    # Sort by number of orders
    sorted_indices = np.argsort(orders)[::-1]
    restaurants = [restaurants[i] for i in sorted_indices]
    orders = [orders[i] for i in sorted_indices]

    # Limit to top 30 restaurants for readability
    if len(restaurants) > 30:
        restaurants = restaurants[:30]
        orders = orders[:30]

    # Create bar plot
    bars = plt.bar(range(len(restaurants)), orders)

    # Customize plot
    plt.title(f"Orders per Restaurant - {solver_name}")
    plt.xlabel("Restaurant ID")
    plt.ylabel("Number of Orders")
    plt.xticks(range(len(restaurants)), restaurants, rotation=45, ha="right")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom")

    plt.tight_layout()

    # Save figure
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"restaurant_distribution_{solver_name}_{timestamp}.png"
    save_path = os.path.join(ensure_visualization_dir(), filename)
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Restaurant distribution visualization saved to {save_path}")


def visualize_delay_distribution(
    episode_stats: Dict[str, Any], solver_name: str, timestamp: Optional[str] = None
) -> None:
    """Visualize the distribution of delivery delays."""
    if not episode_stats.get("delay_values"):
        logger.warning("No delay data available for visualization")
        return

    plt.figure(figsize=(10, 6))

    # Create histogram of delays
    plt.hist(episode_stats["delay_values"], bins=30, alpha=0.7)

    # Add vertical line for mean delay
    mean_delay = np.mean(episode_stats["delay_values"])
    plt.axvline(mean_delay, color="r", linestyle="dashed", linewidth=1, label=f"Mean Delay: {mean_delay:.2f} min")

    # Customize plot
    plt.title(f"Delivery Delay Distribution - {solver_name}")
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Number of Orders")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save figure
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"delay_distribution_{solver_name}_{timestamp}.png"
    save_path = os.path.join(ensure_visualization_dir(), filename)
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Delay distribution visualization saved to {save_path}")


def visualize_bundle_statistics(
    episode_stats: Dict[str, Any], solver_name: str, timestamp: Optional[str] = None
) -> None:
    """Visualize bundle-related statistics."""
    if not episode_stats.get("bundle_sizes"):
        logger.warning("No bundle data available for visualization")
        return

    plt.figure(figsize=(10, 6))

    # Create histogram of bundle sizes
    plt.hist(
        episode_stats["bundle_sizes"], bins=range(1, max(episode_stats["bundle_sizes"]) + 2), alpha=0.7, rwidth=0.85
    )

    # Customize plot
    plt.title(f"Bundle Size Distribution - {solver_name}")
    plt.xlabel("Bundle Size")
    plt.ylabel("Number of Bundles")
    plt.grid(True, alpha=0.3)

    # Save figure
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bundle_statistics_{solver_name}_{timestamp}.png"
    save_path = os.path.join(ensure_visualization_dir(), filename)
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Bundle statistics visualization saved to {save_path}")


def visualize_vehicle_utilization(
    episode_stats: Dict[str, Any], solver_name: str, timestamp: Optional[str] = None
) -> None:
    """Visualize vehicle utilization over time."""
    if not episode_stats.get("active_period_idle_rates_by_vehicle"):
        logger.warning("No vehicle utilization data available for visualization")
        return

    plt.figure(figsize=(12, 6))

    # Plot utilization for each vehicle
    for vehicle_id, idle_rates in episode_stats["active_period_idle_rates_by_vehicle"].items():
        utilization = [1 - rate for rate in idle_rates]
        plt.plot(utilization, label=f"Vehicle {vehicle_id}", alpha=0.7)

    # Customize plot
    plt.title(f"Vehicle Utilization Over Time - {solver_name}")
    plt.xlabel("Time Step")
    plt.ylabel("Utilization Rate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    # Save figure
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vehicle_utilization_{solver_name}_{timestamp}.png"
    save_path = os.path.join(ensure_visualization_dir(), filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Vehicle utilization visualization saved to {save_path}")


def create_summary_visualization(
    episode_stats: Dict[str, Any], solver_name: str, timestamp: Optional[str] = None
) -> None:
    """Create a summary visualization with multiple subplots."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Restaurant Distribution
    if episode_stats.get("orders_per_restaurant"):
        restaurants = list(episode_stats["orders_per_restaurant"].keys())
        orders = list(episode_stats["orders_per_restaurant"].values())
        sorted_indices = np.argsort(orders)[::-1][:10]  # Top 10 restaurants
        ax1.bar(range(len(sorted_indices)), [orders[i] for i in sorted_indices])
        ax1.set_title("Top 10 Restaurants by Orders")
        ax1.set_xlabel("Restaurant ID")
        ax1.set_ylabel("Number of Orders")
        ax1.set_xticks(range(len(sorted_indices)))
        ax1.set_xticklabels([restaurants[i] for i in sorted_indices], rotation=45)

    # 2. Delay Distribution
    if episode_stats.get("delay_values"):
        ax2.hist(episode_stats["delay_values"], bins=30, alpha=0.7)
        ax2.set_title("Delivery Delay Distribution")
        ax2.set_xlabel("Delay (minutes)")
        ax2.set_ylabel("Number of Orders")

    # 3. Bundle Size Distribution
    if episode_stats.get("bundle_sizes"):
        ax3.hist(
            episode_stats["bundle_sizes"], bins=range(1, max(episode_stats["bundle_sizes"]) + 2), alpha=0.7, rwidth=0.85
        )
        ax3.set_title("Bundle Size Distribution")
        ax3.set_xlabel("Bundle Size")
        ax3.set_ylabel("Number of Bundles")

    # 4. Vehicle Utilization
    if episode_stats.get("active_period_idle_rates_by_vehicle"):
        for vehicle_id, idle_rates in episode_stats["active_period_idle_rates_by_vehicle"].items():
            utilization = [1 - rate for rate in idle_rates]
            ax4.plot(utilization, label=f"Vehicle {vehicle_id}", alpha=0.7)
        ax4.set_title("Vehicle Utilization Over Time")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Utilization Rate")
        ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust layout and save
    plt.suptitle(f"Simulation Summary - {solver_name}", fontsize=16)
    plt.tight_layout()

    filename = f"summary_visualization_{solver_name}_{timestamp}.png"
    save_path = os.path.join(ensure_visualization_dir(), filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Summary visualization saved to {save_path}")
