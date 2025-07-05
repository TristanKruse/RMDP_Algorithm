from .demand_patterns import lunch_dinner_pattern


def get_env_config(movement_per_step):
    """Environment configuration with explanatory documentation"""
    return {
        # System size parameters
        "num_restaurants": 80,  # 20, 80, 320
        "num_vehicles": 40,  # 10, 40, 160
        # Time parameters
        "mean_prep_time": 26.8,  # 26.8 # Gamma distributed preparation time (timesteps, 30-sec interval)
        "prep_time_var": 41.8,  # 41.8 # Preparation time variance (30-sec interval)
        "delivery_window": 78,  # Delivery time window (timesteps, 30-sec interval)
        "simulation_duration": 1200,  # 1200 # Total simulation time (timesteps, 30-sec interval)
        "cooldown_duration": 120,  # No new orders in final period (timesteps, 30-sec interval)
        # "warmup_duration": 60,  # 1 hour warmup
        # Workload parameters
        "mean_interarrival_time": 4,  # 4 # Order arrival interval (timesteps, 30-sec interval)
        # Area parameters
        "service_area_dimensions": (6.0, 6.0),  # 10km x 10km area
        "downtown_concentration": 0.71,  # Restaurant concentration downtown
        # Service parameters
        "service_time": 6.0,  # 6.0 # Time at pickup/delivery locations (timesteps, 30-sec interval)
        "movement_per_step": movement_per_step,
        # Visualization
        "visualize": False,
        "update_interval": 0.01,  # Update frequency (0.01 or 1)
        # Optional behavior flags (set by run_test_episode)
        "reposition_idle_vehicles": False,  # Whether vehicles reposition when idle
        "seed": None,  # Random seed for reproducibility
        "demand_pattern": lunch_dinner_pattern,  # e.g., lunch_dinner_pattern,  # Pass your demand pattern here
    }
