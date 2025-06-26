def initialize_episode_stats():
    """Initialize the statistics dictionary for tracking episode metrics."""
    return {
        "total_orders": 0,
        "orders_delivered": 0,
        "total_delay": 0,
        "late_orders": set(),  # Use set to avoid duplicates
        "max_delay": 0,
        "delay_values": [],
        "total_distance": 0,
        "postponed_orders": set(),
        "average_idle_rate": 0,
        "idle_rates_by_vehicle": {},
        "total_idle_time": 0,
        "orders_per_hour": 0,
        "active_period_orders_per_hour": 0,
        "system_capacity": 0,
        "active_period_capacity": 0,
        "active_period_idle_time": 0,
        "active_period_steps": 0,
        "active_period_idle_rate": 0,
        "active_period_idle_rates_by_vehicle": {},
        "orders_per_restaurant": {},  # Track orders by restaurant ID
        # New meal prep time metrics
        "true_prep_times": [],  # Actual meal preparation times (ready_time - request_time)
        "avg_true_prep_time": 0.0,  # Average true preparation time
        "max_true_prep_time": 0.0,  # Maximum true preparation time
        "order_wait_times": [],  # Time orders waited after being ready (pickup_time - ready_time)
        "avg_order_wait_time": 0.0,  # Average time orders waited after being ready
        "max_order_wait_time": 0.0,  # Maximum time orders waited after being ready
        "total_pickup_times": [],  # Total time from order to pickup (pickup_time - request_time)
        "avg_total_pickup_time": 0.0,  # Average total time to pickup
        "max_total_pickup_time": 0.0,  # Maximum total time to pickup
        "total_driver_wait_time": 0.0,  # Total time drivers waited for food
        "driver_wait_orders": 0,  # Number of orders where driver had to wait
        "true_prep_by_restaurant": {},  # Track true prep times by restaurant {restaurant_id: [true_prep_times]}
        "driver_wait_by_restaurant": {},  # Keep this as is
        "order_wait_by_restaurant": {},
        "total_pickup_by_restaurant": {},  # Track total pickup times by restaurant {restaurant_id: [total_pickup_times]}
        # Bundle tracking
        "bundles_formed": 0,  # Total number of bundles formed
        "bundle_sizes": [],  # List of bundle sizes for calculating average
        "bundled_orders": set(),  # Set of orders that were part of a bundle
        "same_restaurant_bundles": 0,  # Bundles where all orders come from same restaurant
        "bundle_delays": [],  # Delays for bundled orders
        "non_bundle_delays": [],  # Delays for non-bundled orders
    }


def calculate_capacity_metrics(stats, simulation_duration, cooldown_duration, warmup_duration):
    """Calculate capacity-related metrics for the simulation."""
    # Convert minutes to hours
    total_hours = simulation_duration / 60
    active_hours = (simulation_duration - cooldown_duration - warmup_duration) / 60

    # Calculate orders per hour
    stats["orders_per_hour"] = stats["orders_delivered"] / total_hours
    stats["active_period_orders_per_hour"] = stats["orders_delivered"] / active_hours if active_hours > 0 else 0
    # Calculate theoretical system capacity (orders/hour * 24 hours)
    stats["system_capacity"] = stats["orders_per_hour"] * 24
    stats["active_period_capacity"] = stats["active_period_orders_per_hour"] * 24

    return stats


def calculate_idle_rate_distance(stats, simulation_duration, vehicle_speed_kmh=16):
    """Calculate distance using idle-rate method for more realistic estimates."""
    idle_rates = stats.get("active_period_idle_rates_by_vehicle", {})
    
    if not idle_rates:
        return 0
    
    # Calculate average vehicle utilization
    vehicle_utilizations = [1 - sum(rates)/len(rates) for rates in idle_rates.values() if rates]
    avg_utilization = sum(vehicle_utilizations) / len(vehicle_utilizations) if vehicle_utilizations else 0
    
    # Calculate productive distance
    vehicle_speed_km_per_minute = vehicle_speed_kmh / 60
    num_vehicles = len(idle_rates)
    total_productive_distance = avg_utilization * simulation_duration * num_vehicles * vehicle_speed_km_per_minute
    
    return total_productive_distance
