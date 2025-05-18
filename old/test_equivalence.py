import os
import sys
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Add the parent directory to Python path to import from training module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from training module
from training.train import run_training as run_new_version
from old.old_train import run_test_episode as run_old_version  # This is in the current directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_episode_stats(new_stats: Dict[str, Any], old_stats: Dict[str, Any], tolerance: float = 1e-6) -> bool:
    """Compare two episode statistics dictionaries for equivalence."""
    # List of keys to compare
    numeric_keys = [
        "total_orders",
        "orders_delivered",
        "total_distance",
        "total_reward",
        "max_delay",
        "mean_delay",
        "median_delay",
        "p90_delay",
        "p95_delay",
        "p99_delay",
        "bundle_rate",
        "avg_bundle_size",
        "max_bundle_size",
        "avg_vehicle_utilization",
        "min_vehicle_utilization",
        "max_vehicle_utilization",
        "restaurant_load_balance",
        "restaurant_utilization",
    ]

    # Compare numeric values
    for key in numeric_keys:
        if key in new_stats and key in old_stats:
            new_val = float(new_stats[key])
            old_val = float(old_stats[key])
            if abs(new_val - old_val) > tolerance:
                logger.error(f"Mismatch in {key}: new={new_val}, old={old_val}")
                return False

    # Compare sets
    set_keys = ["late_orders", "postponed_orders", "bundled_orders"]
    for key in set_keys:
        if key in new_stats and key in old_stats:
            if set(new_stats[key]) != set(old_stats[key]):
                logger.error(f"Mismatch in {key}: new={new_stats[key]}, old={old_stats[key]}")
                return False

    # Compare lists with tolerance
    list_keys = ["delay_values", "bundle_sizes"]
    for key in list_keys:
        if key in new_stats and key in old_stats:
            if len(new_stats[key]) != len(old_stats[key]):
                logger.error(f"Length mismatch in {key}: new={len(new_stats[key])}, old={len(old_stats[key])}")
                return False
            for new_val, old_val in zip(new_stats[key], old_stats[key]):
                if abs(float(new_val) - float(old_val)) > tolerance:
                    logger.error(f"Value mismatch in {key}: new={new_val}, old={old_val}")
                    return False

    return True


def run_equivalence_test(
    solver_name: str = "rl_aca",
    seed: int = 42,
    num_episodes: int = 1,
    use_meituan_data: bool = False,
    meituan_data_path: str = None,
    # Add all the environment parameters
    num_restaurants: int = 50,
    num_vehicles: int = 20,
    mean_prep_time: float = 10.0,
    delivery_window: int = 30,
    # Add RL parameters with default values
    rl_learning_rate: float = 0.0005,
    rl_discount_factor: float = 0.95,
    rl_exploration_rate: float = 0.9,
    rl_exploration_decay: float = 0.99999,
    rl_min_exploration_rate: float = 0.2,
    rl_batch_size: int = 64,
    rl_target_update_frequency: int = 50,
    rl_replay_buffer_capacity: int = 10000,
):
    """Run both versions and compare their results."""
    logger.info("Starting equivalence test...")

    # Create args object for new version
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Create args with all parameters
    args = Args(
        solver=solver_name,
        seed=seed,
        num_episodes=num_episodes,
        use_meituan_data=use_meituan_data,
        meituan_data_path=meituan_data_path,
        visualize=False,
        save_rl_model=False,
        rl_model_path=None,
        num_restaurants=num_restaurants,
        num_vehicles=num_vehicles,
        mean_prep_time=mean_prep_time,
        delivery_window=delivery_window,
        rl_learning_rate=rl_learning_rate,
        rl_discount_factor=rl_discount_factor,
        rl_exploration_rate=rl_exploration_rate,
        rl_exploration_decay=rl_exploration_decay,
        rl_min_exploration_rate=rl_min_exploration_rate,
        rl_batch_size=rl_batch_size,
        rl_target_update_frequency=rl_target_update_frequency,
        rl_replay_buffer_capacity=rl_replay_buffer_capacity,
    )

    # Run new version
    logger.info("Running new modular version...")
    new_results = []
    for episode in range(num_episodes):
        new_stats = run_new_version(args)
        new_results.append(new_stats)

    # Run old version with only the parameters it accepts
    logger.info("Running old monolithic version...")
    old_results = []
    for episode in range(num_episodes):
        old_stats = run_old_version(
            solver_name=solver_name,
            seed=seed,
            meituan_config=None if not use_meituan_data else MeituanDataConfig(meituan_data_path),
        )
        old_results.append(old_stats)

    # Compare results
    logger.info("Comparing results...")
    all_equivalent = True
    for episode in range(num_episodes):
        logger.info(f"Comparing episode {episode + 1}/{num_episodes}")
        if not compare_episode_stats(new_results[episode], old_results[episode]):
            logger.error(f"Results for episode {episode + 1} are not equivalent!")
            all_equivalent = False

    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("data", "test_results")
    os.makedirs(results_dir, exist_ok=True)

    # Convert sets to lists in the results before saving
    def convert_sets_to_lists(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {key: convert_sets_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets_to_lists(item) for item in obj]
        return obj

    comparison_results = {
        "timestamp": timestamp,
        "parameters": args.__dict__,
        "equivalent": all_equivalent,
        "new_results": convert_sets_to_lists(new_results),
        "old_results": convert_sets_to_lists(old_results),
    }

    results_file = os.path.join(results_dir, f"equivalence_test_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(comparison_results, f, indent=2)

    if all_equivalent:
        logger.info("✅ All results are equivalent!")
    else:
        logger.error("❌ Results are not equivalent!")

    return all_equivalent


def main():
    """Main entry point for the equivalence test."""
    # Test with different configurations
    test_configs = [
        {
            "solver_name": "fastest",
            "seed": 42,
            "num_episodes": 1,
            "use_meituan_data": False,
            "num_restaurants": 50,
            "num_vehicles": 20,
            "mean_prep_time": 10.0,
            "delivery_window": 30,
        },
        {
            "solver_name": "rl_aca",
            "seed": 42,
            "num_episodes": 1,
            "use_meituan_data": False,
            "num_restaurants": 50,
            "num_vehicles": 20,
            "mean_prep_time": 10.0,
            "delivery_window": 30,
            "rl_learning_rate": 0.0005,
            "rl_discount_factor": 0.95,
            "rl_exploration_rate": 0.9,
            "rl_exploration_decay": 0.99999,
            "rl_min_exploration_rate": 0.2,
            "rl_batch_size": 64,
            "rl_target_update_frequency": 50,
            "rl_replay_buffer_capacity": 10000,
        },
    ]

    all_passed = True
    for config in test_configs:
        logger.info(f"\nTesting configuration: {config}")
        if not run_equivalence_test(**config):
            all_passed = False

    if all_passed:
        logger.info("\n✅ All tests passed!")
    else:
        logger.error("\n❌ Some tests failed!")


if __name__ == "__main__":
    main()
