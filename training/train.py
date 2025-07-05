import os
import logging
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List

from environment.environment import RestaurantMealDeliveryEnv
from environment.meituan_data.meituan_data_config import MeituanDataConfig

from training.core.episode import run_test_episode
from training.core.stats import initialize_episode_stats, calculate_capacity_metrics
from training.utils.metrics import calculate_all_metrics
from training.utils.visualization import (
    visualize_restaurant_distribution,
    visualize_delay_distribution,
    visualize_bundle_statistics,
    visualize_vehicle_utilization,
    create_summary_visualization,
)
from training.utils.file_io import save_results
from training.config.env_config import get_env_config
from training.config.solver_config import SOLVERS

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and test delivery optimization models")

    # Basic parameters
    parser.add_argument(
        "--solver",
        type=str,
        default="fastest",
        choices=["fastest", "aca", "rl_aca", "bundler"],
        help="Solver to use for optimization",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")

    # Environment parameters
    parser.add_argument("--num_restaurants", type=int, default=50, help="Number of restaurants in the environment")
    parser.add_argument("--num_vehicles", type=int, default=20, help="Number of delivery vehicles")
    parser.add_argument("--mean_prep_time", type=float, default=10.0, help="Mean preparation time for orders")
    parser.add_argument("--delivery_window", type=int, default=30, help="Delivery time window in minutes")

    # RL-specific parameters
    parser.add_argument("--rl_learning_rate", type=float, default=0.0005, help="Learning rate for RL model")
    parser.add_argument("--rl_discount_factor", type=float, default=0.95, help="Discount factor for RL model")
    parser.add_argument("--rl_exploration_rate", type=float, default=0.9, help="Initial exploration rate for RL model")
    parser.add_argument(
        "--rl_exploration_decay", type=float, default=0.99999, help="Exploration rate decay for RL model"
    )
    parser.add_argument(
        "--rl_min_exploration_rate", type=float, default=0.2, help="Minimum exploration rate for RL model"
    )
    parser.add_argument("--rl_batch_size", type=int, default=64, help="Batch size for RL training")
    parser.add_argument(
        "--rl_target_update_frequency", type=int, default=50, help="Frequency of target network updates"
    )
    parser.add_argument("--rl_replay_buffer_capacity", type=int, default=10000, help="Capacity of replay buffer")

    # Meituan data parameters
    parser.add_argument("--use_meituan_data", action="store_true", help="Use Meituan data for simulation")
    parser.add_argument("--meituan_data_path", type=str, default=None, help="Path to Meituan data file")

    # Visualization and saving parameters
    parser.add_argument("--visualize", action="store_true", default=False, help="Enable visualization")
    parser.add_argument("--save_rl_model", action="store_true", help="Save RL model after training")
    parser.add_argument("--rl_model_path", type=str, default=None, help="Path to save/load RL model")

    return parser.parse_args()


def create_solver(args, env_config, movement_per_step):
    """Create and initialize the solver with proper environment setup."""
    # Create environment first to get location_manager
    env = RestaurantMealDeliveryEnv(**env_config)

    # Initialize solver with both movement_per_step and location_manager
    return SOLVERS[args.solver](movement_per_step, env.location_manager), env


def run_training(args) -> List[Dict[str, Any]]:
    """Run the training process with the given arguments.

    Returns:
        List[Dict[str, Any]]: List of episode statistics for each episode
    """
    # Calculate movement_per_step first
    speed = 8  # 8 km/h for 30-second intervals (was 16 km/h for 1-minute intervals)
    street_network_factor = 1.0  # 1.4 in paper, we calculated the average speed over the euclidic distance
    movement_per_step = (speed / 60) / street_network_factor  # km per minute adjusted for street network

    # Initialize environment configuration
    env_config = get_env_config(movement_per_step)
    env_config.update(
        {
            "num_restaurants": args.num_restaurants,
            "num_vehicles": args.num_vehicles,
            "mean_prep_time": args.mean_prep_time,
            "delivery_window": args.delivery_window,
        }
    )

    # Initialize Meituan data configuration if needed
    meituan_config = None
    if args.use_meituan_data:
        if not args.meituan_data_path:
            raise ValueError("Meituan data path must be provided when using Meituan data")
        meituan_config = MeituanDataConfig(args.meituan_data_path)

    # Create solver and environment
    solver, env = create_solver(args, env_config, movement_per_step)

    # Run episodes
    all_episode_stats = []
    for episode in range(args.num_episodes):
        logger.info(f"Starting episode {episode + 1}/{args.num_episodes}")

        # Run episode
        episode_stats = run_test_episode(
            solver_name=args.solver,
            solver=solver,
            meituan_config=meituan_config,
            seed=args.seed,
            reposition_idle_vehicles=True,
            visualize=args.visualize,
            warmup_duration=60,
            save_rl_model=args.save_rl_model,
            rl_model_path=args.rl_model_path,
            save_results_to_disk=True,
            env_config=env_config,
            exploration_rate=None,
            training_mode=True,
            rl_learning_rate=args.rl_learning_rate,
            rl_discount_factor=args.rl_discount_factor,
            rl_exploration_rate=args.rl_exploration_rate,
            rl_exploration_decay=args.rl_exploration_decay,
            rl_min_exploration_rate=args.rl_min_exploration_rate,
            rl_batch_size=args.rl_batch_size,
            rl_target_update_frequency=args.rl_target_update_frequency,
            rl_replay_buffer_capacity=args.rl_replay_buffer_capacity,
        )

        # Calculate metrics
        metrics = calculate_all_metrics(episode_stats)
        all_episode_stats.append(episode_stats)  # Store the raw episode stats, not just metrics

        # Create visualizations only if explicitly requested
        if args.visualize:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            visualize_restaurant_distribution(episode_stats, args.solver, timestamp)
            visualize_delay_distribution(episode_stats, args.solver, timestamp)
            visualize_bundle_statistics(episode_stats, args.solver, timestamp)
            visualize_vehicle_utilization(episode_stats, args.solver, timestamp)
            create_summary_visualization(episode_stats, args.solver, timestamp)

        # Save results
        save_results(
            episode_stats, args.solver, args.seed, meituan_config, solver_params=vars(args), env_params=env_config
        )

        logger.info(f"Completed episode {episode + 1}/{args.num_episodes}")
        logger.info(f"Episode metrics: {metrics}")

    # Calculate and log average metrics across all episodes
    if args.num_episodes > 1:
        avg_metrics = {
            key: sum(episode[key] for episode in all_episode_stats) / len(all_episode_stats)
            for key in all_episode_stats[0].keys()
        }
        logger.info(f"Average metrics across {args.num_episodes} episodes: {avg_metrics}")

    return all_episode_stats  # Return the raw episode stats for testing


def main():
    """Main entry point for the training script."""
    args = parse_args()

    # Create necessary directories
    os.makedirs("data/results", exist_ok=True)
    os.makedirs("data/visualizations", exist_ok=True)
    if args.save_rl_model:
        os.makedirs("data/models", exist_ok=True)

    try:
        run_training(args)
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
