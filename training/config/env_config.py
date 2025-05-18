from .demand_patterns import lunch_dinner_pattern


def get_env_config(movement_per_step):
    """Environment configuration with explanatory documentation"""
    return {
        # System size parameters
        "num_restaurants": 80,  # 20, 80, 320
        "num_vehicles": 40,  # 10, 40, 160
        # Time parameters
        "mean_prep_time": 13.4,  # 13.4 # Gamma distributed preparation time (minutes)
        "prep_time_var": 20.9,  # 2.0 # Preparation time variance
        "delivery_window": 39,  # Delivery time window (minutes)
        "simulation_duration": 600,  # 420 # Total simulation time (minutes)
        "cooldown_duration": 60,  # No new orders in final period (minutes)
        # "warmup_duration": 60,  # 1 hour warmup
        # Workload parameters
        "mean_interarrival_time": 2,  # 8, 2, 0.5
        # Area parameters
        "service_area_dimensions": (6.0, 6.0),  # 10km x 10km area
        "downtown_concentration": 0.71,  # Restaurant concentration downtown
        # Service parameters
        "service_time": 3.0,  # 4.0 # Time at pickup/delivery locations
        "movement_per_step": movement_per_step,
        # Visualization
        "visualize": False,
        "update_interval": 0.01,  # Update frequency (0.01 or 1)
        # Optional behavior flags (set by run_test_episode)
        "reposition_idle_vehicles": False,  # Whether vehicles reposition when idle
        "seed": None,  # Random seed for reproducibility
        "demand_pattern": lunch_dinner_pattern,  # e.g., lunch_dinner_pattern,  # Pass your demand pattern here
    }
