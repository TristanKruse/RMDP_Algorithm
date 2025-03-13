# train_rl_aca.py
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm

from train import run_test_episode, get_env_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def train_rl_aca(
    num_episodes: int = 200,
    save_interval: int = 20,
    district_day=None,
    seed: int = 42,
    visualize: bool = False,
    reposition_idle_vehicles: bool = False,
    model_dir: str = "data/models",
):
    """
    Train the RL-based ACA solver over multiple episodes.
    
    Args:
        num_episodes: Number of episodes to train
        save_interval: Save model every N episodes
        district_day: Tuple of (district_id, day) for Meituan data
        seed: Random seed for reproducibility
        visualize: Whether to enable visualization
        reposition_idle_vehicles: Whether to reposition idle vehicles
        model_dir: Directory to save models
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Metrics to track
    metrics = {
        "rewards": [],
        "delays": [],
        "on_time_rates": [],
        "postponement_rates": []
    }
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training RL-ACA"):
        # Run an episode with training enabled
        model_path = f"{model_dir}/rl_aca_{timestamp}_latest.pt" if episode > 0 else None
        
        # Each episode uses the model from the previous episode
        stats = run_test_episode(
            solver_name="rl_aca",
            district_day=district_day,
            seed=seed + episode,  # Different seed each episode 
            reposition_idle_vehicles=reposition_idle_vehicles,
            visualize=visualize and episode % 20 == 0,  # Only visualize occasionally
            save_rl_model=True,
            rl_model_path=model_path
        )
        
        # Track metrics
        metrics["rewards"].append(stats["total_reward"])
        metrics["delays"].append(sum(stats["delay_values"]) if stats["delay_values"] else 0)
        
        total_orders = max(1, stats["orders_delivered"])
        late_orders = len(stats["late_orders"])
        on_time_rate = ((total_orders - late_orders) / total_orders) * 100
        metrics["on_time_rates"].append(on_time_rate)
        
        postponement_rate = len(stats["postponed_orders"]) / max(1, stats["total_orders"]) * 100
        metrics["postponement_rates"].append(postponement_rate)
        
        # Log progress
        if (episode + 1) % 5 == 0:
            logger.info(f"Episode {episode+1}/{num_episodes}: "
                       f"Reward: {stats['total_reward']:.2f}, "
                       f"On-time rate: {on_time_rate:.2f}%, "
                       f"Postponement rate: {postponement_rate:.2f}%")
        
        # Save model at specified intervals
        if (episode + 1) % save_interval == 0:
            checkpoint_path = f"{model_dir}/rl_aca_{timestamp}_ep{episode+1}.pt"
            # Copy the latest model to a checkpoint file
            if os.path.exists(model_path):
                import shutil
                shutil.copy(model_path, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Plot training metrics
    plot_training_metrics(metrics, timestamp, model_dir)
    
    logger.info(f"Training completed. Final model saved to {model_path}")
    return model_path

def plot_training_metrics(metrics, timestamp, model_dir):
    """Plot training metrics and save the figures."""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics["rewards"])
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    # Plot delays
    plt.subplot(2, 2, 2)
    plt.plot(metrics["delays"])
    plt.title("Total Delay per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Delay (minutes)")
    
    # Plot on-time rates
    plt.subplot(2, 2, 3)
    plt.plot(metrics["on_time_rates"])
    plt.title("On-Time Delivery Rate")
    plt.xlabel("Episode")
    plt.ylabel("On-Time Rate (%)")
    plt.ylim(0, 100)
    
    # Plot postponement rates
    plt.subplot(2, 2, 4)
    plt.plot(metrics["postponement_rates"])
    plt.title("Postponement Rate")
    plt.xlabel("Episode")
    plt.ylabel("Postponement Rate (%)")
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f"{model_dir}/rl_aca_training_{timestamp}.png", dpi=300)
    plt.close()

def evaluate_model(
    model_path: str,
    num_episodes: int = 10,
    district_day=None,
    seed: int = 100,
    visualize: bool = False,
):
    """
    Evaluate a trained RL-ACA model over multiple episodes.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of evaluation episodes
        district_day: Tuple of (district_id, day) for Meituan data
        seed: Random seed for reproducibility
        visualize: Whether to enable visualization
    """
    # Metrics to track
    metrics = {
        "total_rewards": [],
        "total_delays": [],
        "on_time_rates": [],
        "max_delays": []
    }
    
    logger.info(f"Evaluating model: {model_path}")
    
    # Evaluation loop
    for episode in tqdm(range(num_episodes), desc="Evaluating RL-ACA"):
        # Run an episode with training disabled
        stats = run_test_episode(
            solver_name="rl_aca",
            district_day=district_day,
            seed=seed + episode,  # Different seed each episode 
            reposition_idle_vehicles=False,
            visualize=visualize,
            rl_model_path=model_path
        )
        
        # Track metrics
        metrics["total_rewards"].append(stats["total_reward"])
        metrics["total_delays"].append(sum(stats["delay_values"]) if stats["delay_values"] else 0)
        metrics["max_delays"].append(stats["max_delay"])
        
        total_orders = max(1, stats["orders_delivered"])
        late_orders = len(stats["late_orders"])
        on_time_rate = ((total_orders - late_orders) / total_orders) * 100
        metrics["on_time_rates"].append(on_time_rate)
    
    # Calculate average metrics
    avg_reward = np.mean(metrics["total_rewards"])
    avg_delay = np.mean(metrics["total_delays"])
    avg_on_time = np.mean(metrics["on_time_rates"])
    avg_max_delay = np.mean(metrics["max_delays"])
    
    logger.info(f"Evaluation Results ({num_episodes} episodes):")
    logger.info(f"Average Reward: {avg_reward:.2f}")
    logger.info(f"Average Total Delay: {avg_delay:.2f} minutes")
    logger.info(f"Average On-Time Rate: {avg_on_time:.2f}%")
    logger.info(f"Average Max Delay: {avg_max_delay:.2f} minutes")
    
    return metrics

def compare_models(
    heuristic_episodes: int = 5,
    rl_episodes: int = 5,
    rl_model_path: str = None,
    district_day=None,
    seed: int = 200,
):
    """
    Compare the RL-based ACA with the original heuristic ACA.
    
    Args:
        heuristic_episodes: Number of episodes to run with heuristic ACA
        rl_episodes: Number of episodes to run with RL-based ACA
        rl_model_path: Path to the trained RL model
        district_day: Tuple of (district_id, day) for Meituan data
        seed: Random seed for reproducibility
    """
    logger.info("Comparing heuristic ACA vs RL-based ACA")
    
    # Run heuristic ACA episodes
    heuristic_metrics = {
        "total_rewards": [],
        "total_delays": [],
        "on_time_rates": [],
        "postponement_rates": []
    }
    
    for episode in tqdm(range(heuristic_episodes), desc="Running Heuristic ACA"):
        stats = run_test_episode(
            solver_name="aca",  # Original ACA
            district_day=district_day,
            seed=seed + episode,
            reposition_idle_vehicles=False,
            visualize=False
        )
        
        heuristic_metrics["total_rewards"].append(stats["total_reward"])
        heuristic_metrics["total_delays"].append(sum(stats["delay_values"]) if stats["delay_values"] else 0)
        
        total_orders = max(1, stats["orders_delivered"])
        late_orders = len(stats["late_orders"])
        on_time_rate = ((total_orders - late_orders) / total_orders) * 100
        heuristic_metrics["on_time_rates"].append(on_time_rate)
        
        postponement_rate = len(stats["postponed_orders"]) / max(1, stats["total_orders"]) * 100
        heuristic_metrics["postponement_rates"].append(postponement_rate)
    
    # Run RL-based ACA episodes
    rl_metrics = {
        "total_rewards": [],
        "total_delays": [],
        "on_time_rates": [],
        "postponement_rates": []
    }
    
    for episode in tqdm(range(rl_episodes), desc="Running RL-based ACA"):
        stats = run_test_episode(
            solver_name="rl_aca",  # RL-based ACA
            district_day=district_day,
            seed=seed + heuristic_episodes + episode,
            reposition_idle_vehicles=False,
            visualize=False,
            rl_model_path=rl_model_path
        )
        
        rl_metrics["total_rewards"].append(stats["total_reward"])
        rl_metrics["total_delays"].append(sum(stats["delay_values"]) if stats["delay_values"] else 0)
        
        total_orders = max(1, stats["orders_delivered"])
        late_orders = len(stats["late_orders"])
        on_time_rate = ((total_orders - late_orders) / total_orders) * 100
        rl_metrics["on_time_rates"].append(on_time_rate)
        
        postponement_rate = len(stats["postponed_orders"]) / max(1, stats["total_orders"]) * 100
        rl_metrics["postponement_rates"].append(postponement_rate)
    
    # Calculate average metrics
    heuristic_avg_reward = np.mean(heuristic_metrics["total_rewards"])
    heuristic_avg_delay = np.mean(heuristic_metrics["total_delays"])
    heuristic_avg_on_time = np.mean(heuristic_metrics["on_time_rates"])
    heuristic_avg_postpone = np.mean(heuristic_metrics["postponement_rates"])
    
    rl_avg_reward = np.mean(rl_metrics["total_rewards"])
    rl_avg_delay = np.mean(rl_metrics["total_delays"])
    rl_avg_on_time = np.mean(rl_metrics["on_time_rates"])
    rl_avg_postpone = np.mean(rl_metrics["postponement_rates"])
    logger.info("\nComparison Results:")
    logger.info(f"{'Metric':<25} {'Heuristic ACA':<20} {'RL-based ACA':<20} {'Improvement':<15}")
    logger.info(f"{'-'*70}")
    logger.info(f"{'Average Reward':<25} {heuristic_avg_reward:<20.2f} {rl_avg_reward:<20.2f} {((rl_avg_reward - heuristic_avg_reward) / abs(heuristic_avg_reward)) * 100:<15.2f}%")
    logger.info(f"{'Average Total Delay':<25} {heuristic_avg_delay:<20.2f} {rl_avg_delay:<20.2f} {((heuristic_avg_delay - rl_avg_delay) / heuristic_avg_delay) * 100:<15.2f}%")
    logger.info(f"{'Average On-Time Rate':<25} {heuristic_avg_on_time:<20.2f}% {rl_avg_on_time:<20.2f}% {(rl_avg_on_time - heuristic_avg_on_time):<15.2f}pp")
    logger.info(f"{'Postponement Rate':<25} {heuristic_avg_postpone:<20.2f}% {rl_avg_postpone:<20.2f}% {(rl_avg_postpone - heuristic_avg_postpone):<15.2f}pp")
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    # Plot rewards comparison
    plt.subplot(2, 2, 1)
    plt.bar(["Heuristic", "RL"], [heuristic_avg_reward, rl_avg_reward])
    plt.title("Average Reward")
    plt.ylabel("Reward")
    
    # Plot delay comparison
    plt.subplot(2, 2, 2)
    plt.bar(["Heuristic", "RL"], [heuristic_avg_delay, rl_avg_delay])
    plt.title("Average Total Delay")
    plt.ylabel("Delay (minutes)")
    
    # Plot on-time rate comparison
    plt.subplot(2, 2, 3)
    plt.bar(["Heuristic", "RL"], [heuristic_avg_on_time, rl_avg_on_time])
    plt.title("Average On-Time Rate")
    plt.ylabel("On-Time Rate (%)")
    plt.ylim(0, 100)
    
    # Plot postponement rate comparison
    plt.subplot(2, 2, 4)
    plt.bar(["Heuristic", "RL"], [heuristic_avg_postpone, rl_avg_postpone])
    plt.title("Postponement Rate")
    plt.ylabel("Postponement Rate (%)")
    plt.ylim(0, 100)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = f"data/models/aca_comparison_{timestamp}.png"
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    
    logger.info(f"\nComparison chart saved to {comparison_path}")
    return {
        "heuristic": heuristic_metrics,
        "rl": rl_metrics
    }

if __name__ == "__main__":
    # Example usage for training
    model_path = train_rl_aca(
        num_episodes=50,       # Number of training episodes
        save_interval=10,      # Save model every 10 episodes
        seed=42,               # Random seed
        visualize=False,       # Disable visualization during training
        model_dir="data/models"  # Directory to save models
    )
    
    # Evaluate the trained model
    evaluate_model(
        model_path=model_path,   # Path to the trained model
        num_episodes=5,          # Number of evaluation episodes
        seed=100,                # Different seed for evaluation
        visualize=True           # Enable visualization during evaluation
    )
    
    # Compare with heuristic ACA
    compare_models(
        heuristic_episodes=5,    # Number of episodes for heuristic ACA
        rl_episodes=5,           # Number of episodes for RL-based ACA
        rl_model_path=model_path,  # Path to the trained RL model
        seed=200                 # Different seed for comparison
    )