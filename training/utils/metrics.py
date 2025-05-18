# training/utils/metrics.py

import numpy as np
from typing import Dict, List, Any


def calculate_delay_metrics(delay_values: List[float]) -> Dict[str, float]:
    """Calculate various delay-related metrics."""
    if not delay_values:
        return {
            "mean_delay": 0.0,
            "median_delay": 0.0,
            "p90_delay": 0.0,
            "p95_delay": 0.0,
            "p99_delay": 0.0,
            "max_delay": 0.0,
        }

    return {
        "mean_delay": np.mean(delay_values),
        "median_delay": np.median(delay_values),
        "p90_delay": np.percentile(delay_values, 90),
        "p95_delay": np.percentile(delay_values, 95),
        "p99_delay": np.percentile(delay_values, 99),
        "max_delay": np.max(delay_values),
    }


def calculate_bundle_metrics(episode_stats: Dict[str, Any]) -> Dict[str, float]:
    """Calculate bundle-related metrics."""
    total_orders = episode_stats["total_orders"]
    if total_orders == 0:
        return {"bundle_rate": 0.0, "avg_bundle_size": 0.0, "max_bundle_size": 0.0}

    bundled_orders = len(episode_stats["bundled_orders"])
    bundle_sizes = episode_stats["bundle_sizes"]

    return {
        "bundle_rate": bundled_orders / total_orders if total_orders > 0 else 0.0,
        "avg_bundle_size": np.mean(bundle_sizes) if bundle_sizes else 0.0,
        "max_bundle_size": np.max(bundle_sizes) if bundle_sizes else 0.0,
    }


def calculate_vehicle_metrics(episode_stats: Dict[str, Any]) -> Dict[str, float]:
    """Calculate vehicle utilization metrics."""
    active_period_steps = episode_stats["active_period_steps"]
    if active_period_steps == 0:
        return {"avg_vehicle_utilization": 0.0, "min_vehicle_utilization": 0.0, "max_vehicle_utilization": 0.0}

    idle_rates_by_vehicle = episode_stats["active_period_idle_rates_by_vehicle"]
    if not idle_rates_by_vehicle:
        return {"avg_vehicle_utilization": 0.0, "min_vehicle_utilization": 0.0, "max_vehicle_utilization": 0.0}

    # Calculate utilization as (1 - idle_rate) for each vehicle
    vehicle_utilizations = []
    for vehicle_id, idle_rates in idle_rates_by_vehicle.items():
        if idle_rates:  # Check if there are any idle rates recorded
            avg_idle_rate = np.mean(idle_rates)
            utilization = 1 - avg_idle_rate
            vehicle_utilizations.append(utilization)

    if not vehicle_utilizations:
        return {"avg_vehicle_utilization": 0.0, "min_vehicle_utilization": 0.0, "max_vehicle_utilization": 0.0}

    return {
        "avg_vehicle_utilization": np.mean(vehicle_utilizations),
        "min_vehicle_utilization": np.min(vehicle_utilizations),
        "max_vehicle_utilization": np.max(vehicle_utilizations),
    }


def calculate_restaurant_metrics(episode_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate restaurant-specific metrics."""
    orders_per_restaurant = episode_stats["orders_per_restaurant"]
    if not orders_per_restaurant:
        return {"restaurant_load_balance": 0.0, "restaurant_utilization": 0.0}

    # Calculate load balance (standard deviation of orders per restaurant)
    order_counts = list(orders_per_restaurant.values())
    load_balance = np.std(order_counts) if order_counts else 0.0

    # Calculate restaurant utilization
    total_orders = sum(order_counts)
    num_restaurants = len(orders_per_restaurant)
    restaurant_utilization = (
        total_orders / (num_restaurants * episode_stats["total_orders"]) if episode_stats["total_orders"] > 0 else 0.0
    )

    return {"restaurant_load_balance": load_balance, "restaurant_utilization": restaurant_utilization}


def calculate_all_metrics(episode_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate all metrics for the episode."""
    metrics = {}

    # Add delay metrics
    metrics.update(calculate_delay_metrics(episode_stats["delay_values"]))

    # Add bundle metrics
    metrics.update(calculate_bundle_metrics(episode_stats))

    # Add vehicle metrics
    metrics.update(calculate_vehicle_metrics(episode_stats))

    # Add restaurant metrics
    metrics.update(calculate_restaurant_metrics(episode_stats))

    # Add basic statistics
    metrics.update(
        {
            "total_orders": episode_stats["total_orders"],
            "orders_delivered": episode_stats["orders_delivered"],
            "total_distance": episode_stats["total_distance"],
            "total_reward": episode_stats["total_reward"],
            "late_orders_count": len(episode_stats["late_orders"]),
            "postponed_orders_count": len(episode_stats["postponed_orders"]),
        }
    )

    return metrics
