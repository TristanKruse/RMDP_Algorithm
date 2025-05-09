# train_rl.py - Simplified to focus on phased training
from models.aca_policy.rl_postponement import RLPostponementDecision
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple
from train import run_test_episode, get_env_config
import json
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_rl_aca(
    phases: List[Dict],
    save_interval: int = 20,
    stability_window: int = 10,       # Window for stability assessment
    stability_threshold: float = 3.0, # Max percent change considered stable
    seed: int = 1,
    visualize: bool = False,
    reposition_idle_vehicles: bool = False,
    model_dir: str = "data/models",
    resume_from_model: str = None,    # Support resuming
    start_phase: int = 0,             # Phase to start from when resuming
    start_episode: int = 0,           # Episode to start from when resuming
    exploration_start: float = 0.9,   # Initial exploration rate
    exploration_end: float = 0.05,     # Final exploration rate
    decay_method: str = "exponential",     # "linear" or "exponential"
    decay_rate: float = 0.999,         # For exponential decay
    # RL hyperparameters
    rl_learning_rate: float = 0.0005,
    rl_discount_factor: float = 0.95,
    rl_exploration_rate: float = 0.9,
    rl_exploration_decay: float = 0.99,
    rl_min_exploration_rate: float = 0.2,
    rl_batch_size: int = 64,
    rl_target_update_frequency: int = 50,
    rl_replay_buffer_capacity: int = 50000,
    rl_bundling_reward: float = 0.05,
    rl_postponement_penalty: float = -0.005,
    rl_on_time_reward: float = 0.2
    ):
    """
    Train the RL-based ACA solver through multiple progressive phases.
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory for phase-specific data
    phase_dir = os.path.join(model_dir, f"phased_training_{timestamp}")
    os.makedirs(phase_dir, exist_ok=True)
    
    # Use a single latest model path that will be updated throughout training
    latest_model_path = os.path.join(phase_dir, "rl_aca_latest.pt")
    
    # Set initial model path if resuming
    current_model_path = resume_from_model if resume_from_model else latest_model_path
    
    # Save phase configurations for reference
    with open(os.path.join(phase_dir, "phase_configs.json"), "w") as f:
        json.dump(phases, f, indent=4)
    
    # Initialize overall metrics
    all_metrics = {
        "phase_transitions": [],
        "rewards": [],
        "delays": [],
        "on_time_rates": [],
        "postponement_rates": [],
        "losses": []  # Add a list to track all losses
    }

    # Initialize exploration rate at the start
    current_exploration_rate = exploration_start
    
    # Calculate total episodes across all phases
    max_total_episodes = sum(phase.get("max_episodes", 100) for phase in phases)
    
    # If resuming, attempt to load metrics from previous runs
    if resume_from_model:
        resume_dir = os.path.dirname(resume_from_model)
        metrics_file = os.path.join(resume_dir, f"phased_metrics_{os.path.basename(resume_dir).split('_')[-1]}.npz")
        if os.path.exists(metrics_file):
            saved_metrics = np.load(metrics_file, allow_pickle=True)
            all_metrics = {
                "phase_transitions": saved_metrics["phase_transitions"].tolist(),
                "rewards": saved_metrics["rewards"].tolist(),
                "delays": saved_metrics["delays"].tolist(),
                "on_time_rates": saved_metrics["on_time_rates"].tolist(),
                "postponement_rates": saved_metrics["postponement_rates"].tolist(),
                "losses": saved_metrics.get("losses", []).tolist()
            }
            logger.info(f"Resuming training from phase {start_phase}, episode {start_episode}")
    
    # Training loop through phases
    current_phase_idx = start_phase
    total_completed_episodes = 0  # For exploration rate

    while current_phase_idx < len(phases):
        current_phase = phases[current_phase_idx]
        phase_name = current_phase.get("name", f"Phase {current_phase_idx + 1}")
        
        logger.info(f"\n{'=' * 80}\nStarting phase {current_phase_idx + 1}/{len(phases)}: {phase_name}")
        logger.info(f"Environment: {current_phase['env_config']}")
        logger.info(f"Performance criteria: {current_phase['performance_criteria']}\n{'=' * 80}")
        
        # Initialize phase-specific metrics
        phase_metrics = {
            "rewards": [],
            "delays": [],
            "on_time_rates": [],
            "postponement_rates": []
        }
        
        # Process episodes for the current phase
        episode_in_phase = start_episode if current_phase_idx == start_phase else 0
        phase_complete = False
        min_episodes = current_phase.get("min_episodes", 20)
        max_episodes = current_phase.get("max_episodes", 100)
        
        # Create progress bar for this phase
        total_possible = max_episodes - episode_in_phase
        with tqdm(total=total_possible, desc=f"Phase {current_phase_idx + 1}/{len(phases)}: {phase_name}") as pbar:
            while not phase_complete:
                # Pass phase-specific environment parameters
                env_config = current_phase["env_config"]
                
                # Log current model status
                if episode_in_phase == 0:
                    logger.info(f"Starting phase with model: {current_model_path}")
                else:
                    logger.info(f"Using model: {latest_model_path}")
      
                # Update exploration rate based on selected decay method (continuous across phases)
                if decay_method == "linear":
                    progress = total_completed_episodes / max_total_episodes
                    current_exploration_rate = max(
                        exploration_end,
                        exploration_start - (exploration_start - exploration_end) * progress
                    )
                elif decay_method == "exponential":
                    # If this is not the first episode, apply exponential decay
                    if current_phase_idx > 0 or episode_in_phase > 0:
                        current_exploration_rate = max(
                            exploration_end,
                            current_exploration_rate * decay_rate
                        )

                # Run test episode - always save to the same latest model path
                stats = run_test_episode(
                    solver_name="rl_aca",
                    # seed=seed + episode_in_phase,  # Different seed each episode for diversity
                    seed=seed,
                    reposition_idle_vehicles=reposition_idle_vehicles,
                    visualize=visualize and episode_in_phase % 20 == 0,
                    save_rl_model=True,
                    rl_model_path=latest_model_path,  # Always use the latest model path
                    save_results_to_disk=False,
                    env_config=env_config,
                    exploration_rate=current_exploration_rate,
                    # Pass RL hyperparameters
                    rl_learning_rate=rl_learning_rate,
                    rl_discount_factor=rl_discount_factor,
                    rl_exploration_rate=rl_exploration_rate,
                    rl_exploration_decay=rl_exploration_decay,
                    rl_min_exploration_rate=rl_min_exploration_rate,
                    rl_batch_size=rl_batch_size,
                    rl_target_update_frequency=rl_target_update_frequency,
                    rl_replay_buffer_capacity=rl_replay_buffer_capacity,
                    rl_bundling_reward=rl_bundling_reward,
                    rl_postponement_penalty=rl_postponement_penalty,
                    rl_on_time_reward=rl_on_time_reward
                )
                
                # Update metrics
                reward = stats["total_reward"]
                delay = sum(stats["delay_values"]) if stats["delay_values"] else 0
                
                total_orders = max(1, stats["orders_delivered"])
                late_orders = len(stats["late_orders"])
                on_time_rate = ((total_orders - late_orders) / total_orders) * 100
                
                postponement_rate = len(stats["postponed_orders"]) / max(1, stats["total_orders"]) * 100
                
                # Update phase metrics
                phase_metrics["rewards"].append(reward)
                phase_metrics["delays"].append(delay)
                phase_metrics["on_time_rates"].append(on_time_rate)
                phase_metrics["postponement_rates"].append(postponement_rate)
                
                # Update overall metrics
                all_metrics["rewards"].append(reward)
                all_metrics["delays"].append(delay)
                all_metrics["on_time_rates"].append(on_time_rate)
                all_metrics["postponement_rates"].append(postponement_rate)
                

                # Extract losses from the solver and append to all_metrics["losses"]
                try:
                    solver = RLPostponementDecision()
                    solver.load_model(latest_model_path)
                    episode_losses = solver.batch_losses  # Get the losses for this episode
                    all_metrics["losses"].extend(episode_losses)  # Append to the cumulative list
                except Exception as e:
                    logger.warning(f"Failed to extract losses: {e}")

                # Plot losses at save intervals, overwriting the same file
                if (episode_in_phase + 1) % save_interval == 0:
                    loss_plot_path = os.path.join(phase_dir, "loss_plot.png")  # Single file, overwritten
                    plot_losses(
                        losses=all_metrics["losses"],
                        save_path=loss_plot_path,
                        window_size=20,
                        phase_idx=current_phase_idx + 1,
                        episode_idx=episode_in_phase + 1,
                        total_steps=len(all_metrics["losses"])
                    )
                    logger.info(f"Updated loss plot at {loss_plot_path}")



                # Update progress bar with key metrics             
                pbar.set_postfix({
                    'reward': f"{reward:.2f}".ljust(10), 
                    'on-time': f"{on_time_rate:.1f}%".ljust(10),
                    'delay': f"{delay:.1f}".ljust(8),
                    #'seed': f"{seed + episode_in_phase}",
                    'seed': f"{seed}".ljust(6),
                    'explore': f"{current_exploration_rate:.3f}".ljust(8),  # Add exploration rate
                    'postponed': f"{postponement_rate:.1f}%".ljust(8)
                })
                pbar.update(1)
                
                # Save checkpoints at regular intervals
                if (episode_in_phase + 1) % save_interval == 0:
                    # Create checkpoint name
                    checkpoint_path = os.path.join(phase_dir, f"rl_aca_phase{current_phase_idx+1}_ep{episode_in_phase+1}.pt")
                    
                    # Create resuming info
                    resume_info = {
                        "phase": current_phase_idx,
                        "episode": episode_in_phase + 1,
                        "timestamp": timestamp,
                    }
                    
                    # Save resuming info
                    with open(os.path.join(phase_dir, "resuming_info.json"), "w") as f:
                        json.dump(resume_info, f)
                    
                    # Copy latest model to checkpoint
                    if os.path.exists(latest_model_path):
                        try:
                            shutil.copy(latest_model_path, checkpoint_path)
                            logger.info(f"Saved checkpoint to {checkpoint_path}")
                        except Exception as e:
                            logger.error(f"Failed to save checkpoint: {e}")
                    else:
                        logger.warning(f"Could not save checkpoint: Model file {latest_model_path} does not exist")
                    
                    # Save metrics
                    metrics_file = os.path.join(phase_dir, f"phased_metrics_{timestamp}.npz")
                    np.savez(
                        metrics_file, 
                        phase_transitions=np.array(all_metrics["phase_transitions"]),
                        rewards=np.array(all_metrics["rewards"]),
                        delays=np.array(all_metrics["delays"]),
                        on_time_rates=np.array(all_metrics["on_time_rates"]),
                        postponement_rates=np.array(all_metrics["postponement_rates"])
                    )
                
                # Check if phase completion criteria are met
                criteria_met, reason = check_phase_criteria(
                    phase_metrics=phase_metrics,
                    episode_count=episode_in_phase + 1,
                    min_episodes=min_episodes,
                    max_episodes=max_episodes,
                    stability_window=stability_window,
                    stability_threshold=stability_threshold,
                    performance_criteria=current_phase["performance_criteria"]
                )
                
                # Check if phase is complete
                if criteria_met or episode_in_phase >= max_episodes:
                    phase_complete = True
                    
                    # Save phase transition point
                    all_metrics["phase_transitions"].append(len(all_metrics["rewards"]) - 1)
                    
                    # Save phase final model
                    phase_final_path = os.path.join(phase_dir, f"rl_aca_phase{current_phase_idx+1}_final.pt")
                    if os.path.exists(latest_model_path):
                        try:
                            shutil.copy(latest_model_path, phase_final_path)
                            logger.info(f"Phase {current_phase_idx + 1} final model saved to {phase_final_path}")
                        except Exception as e:
                            logger.error(f"Failed to create final model: {e}")
                    else:
                        logger.warning(f"Cannot create final model - source file {latest_model_path} not found")
                    
                    # Plot phase results
                    plot_phase_results(phase_metrics, current_phase_idx, phase_name, phase_dir, timestamp)
                    
                    # Log completion message
                    if criteria_met:
                        logger.info(f"Phase {current_phase_idx + 1} completed after {episode_in_phase + 1} episodes")
                        logger.info(f"Reason: {reason}")
                
                # Increment episode counter
                episode_in_phase += 1
                total_completed_episodes += 1  # Increment total completed episodes

        # Move to next phase
        current_phase_idx += 1
        start_episode = 0  # Reset episode counter for next phase
    
    # Training complete - plot overall results
    plot_training_results(all_metrics, phases, phase_dir, timestamp)
    
    # Get path to final model
    final_model_path = os.path.join(phase_dir, f"rl_aca_phase{len(phases)}_final.pt")
    
    logger.info(f"Phased training completed. Final model saved to {final_model_path}")
    return final_model_path


def check_phase_criteria(
    phase_metrics: Dict,
    episode_count: int,
    min_episodes: int,
    max_episodes: int,
    stability_window: int,
    stability_threshold: float,
    performance_criteria: Dict
) -> Tuple[bool, str]:
    """
    Check if the performance criteria for the current phase have been met.
    
    Args:
        phase_metrics: Dictionary of metrics for the current phase
        episode_count: Number of episodes run in this phase
        min_episodes: Minimum episodes required before advancement
        max_episodes: Maximum episodes before forced advancement
        stability_window: Number of recent episodes to check for stability
        stability_threshold: Maximum percentage change in reward
        performance_criteria: Dictionary of criteria to meet
        
    Returns:
        Tuple of (criteria_met, reason)
    """
    # Ensure minimum episodes have been run
    if episode_count < min_episodes:
        return False, f"Minimum episodes not reached ({episode_count}/{min_episodes})"
    
    # Get metrics from recent episodes
    if len(phase_metrics["rewards"]) < stability_window:
        return False, f"Not enough data for stability check ({len(phase_metrics['rewards'])}/{stability_window})"
    
    recent_rewards = phase_metrics["rewards"][-stability_window:]
    recent_delays = phase_metrics["delays"][-stability_window:]
    recent_on_time = phase_metrics["on_time_rates"][-stability_window:]
    
    # Check reward stability using max-min range
    min_reward = min(recent_rewards)
    max_reward = max(recent_rewards)
    
    # For negative rewards (common in RL), we use absolute values to calculate percentage
    if min_reward < 0 and max_reward < 0:
        # Both rewards are negative, so use absolute values
        min_abs = abs(min_reward)
        max_abs = abs(max_reward)
        # Since rewards are negative, min_reward has the larger absolute value
        percent_range = (min_abs - max_abs) / min_abs * 100.0
    elif min_reward >= 0 and max_reward >= 0:
        # Both rewards are positive
        if min_reward == 0:
            percent_range = float('inf') if max_reward > 0 else 0.0
        else:
            percent_range = (max_reward - min_reward) / min_reward * 100.0
    else:
        # Rewards cross zero, which indicates high instability
        percent_range = 100.0  # Just set to a high value
    
    if percent_range > stability_threshold:
        return False, f"Reward not stable: {percent_range:.2f}% range > {stability_threshold:.2f}% threshold"
    
    # Check average reward criteria
    if "min_avg_reward" in performance_criteria:
        avg_reward = np.mean(recent_rewards)
        if avg_reward < performance_criteria["min_avg_reward"]:
            return False, f"Average reward too low: {avg_reward:.2f} < {performance_criteria['min_avg_reward']:.2f}"
    
    # Check on-time delivery rate criteria
    if "min_on_time_rate" in performance_criteria:
        avg_on_time = np.mean(recent_on_time)
        if avg_on_time < performance_criteria["min_on_time_rate"]:
            return False, f"On-time rate too low: {avg_on_time:.2f}% < {performance_criteria['min_on_time_rate']:.2f}%"
    
    # Check maximum average delay criteria
    if "max_avg_delay" in performance_criteria:
        avg_delay = np.mean(recent_delays)
        if avg_delay > performance_criteria["max_avg_delay"]:
            return False, f"Average delay too high: {avg_delay:.2f} > {performance_criteria['max_avg_delay']:.2f}"
    
    # Force phase transition if max episodes reached
    if episode_count >= max_episodes:
        return True, f"Maximum episodes reached ({max_episodes})"
            
    # All criteria met
    return True, "All criteria met"


def plot_phase_results(metrics, phase_idx, phase_name, output_dir, timestamp):
    """Plot results for a single training phase with trend lines."""
    plt.figure(figsize=(15, 10))
    
    # Number of episodes for x-axis
    episodes = np.arange(len(metrics["rewards"]))

    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(episodes, metrics["rewards"], label="Rewards")
    # Add trend line
    z = np.polyfit(episodes, metrics["rewards"], 1)
    p = np.poly1d(z)
    plt.plot(episodes, p(episodes), "r--", label=f"Trend (slope={z[0]:.2f})")
    plt.title(f"Phase {phase_idx + 1}: {phase_name} - Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    # Plot delays
    plt.subplot(2, 2, 2)
    plt.plot(episodes, metrics["delays"], label="Delays")
    # Add trend line
    z = np.polyfit(episodes, metrics["delays"], 1)
    p = np.poly1d(z)
    plt.plot(episodes, p(episodes), "r--", label=f"Trend (slope={z[0]:.2f})")
    plt.title(f"Phase {phase_idx + 1}: {phase_name} - Delays")
    plt.xlabel("Episode")
    plt.ylabel("Total Delay (minutes)")
    plt.legend()

    # Plot on-time rates
    plt.subplot(2, 2, 3)
    plt.plot(episodes, metrics["on_time_rates"], label="On-Time Rate")
    # Add trend line
    z = np.polyfit(episodes, metrics["on_time_rates"], 1)
    p = np.poly1d(z)
    plt.plot(episodes, p(episodes), "r--", label=f"Trend (slope={z[0]:.2f})")
    plt.title(f"Phase {phase_idx + 1}: {phase_name} - On-Time Rate")
    plt.xlabel("Episode")
    plt.ylabel("On-Time Rate (%)")
    plt.ylim(0, 100)
    plt.legend()

    # Plot postponement rates
    plt.subplot(2, 2, 4)
    plt.plot(episodes, metrics["postponement_rates"], label="Postponement Rate")
    # Add trend line
    z = np.polyfit(episodes, metrics["postponement_rates"], 1)
    p = np.poly1d(z)
    plt.plot(episodes, p(episodes), "r--", label=f"Trend (slope={z[0]:.2f})")
    plt.title(f"Phase {phase_idx + 1}: {phase_name} - Postponement Rate")
    plt.xlabel("Episode")
    plt.ylabel("Postponement Rate (%)")
    plt.ylim(0, 100)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"phase{phase_idx+1}_results_{timestamp}.png"), dpi=300)
    plt.close()


def plot_training_results(metrics, phases, output_dir, timestamp):
    """Plot overall training results across all phases with trend lines."""
    plt.figure(figsize=(15, 12))
    
    # Number of episodes for x-axis
    episodes = np.arange(len(metrics["rewards"]))
    
    # Get phase transition points
    phase_transitions = metrics["phase_transitions"]

    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(episodes, metrics["rewards"], label="Rewards")
    # Add trend line
    z = np.polyfit(episodes, metrics["rewards"], 1)
    p = np.poly1d(z)
    plt.plot(episodes, p(episodes), "r--", label=f"Trend (slope={z[0]:.2f})")
    plt.title("Total Reward Across All Phases")
    plt.xlabel("Total Episodes")
    plt.ylabel("Reward")
    for i, transition in enumerate(phase_transitions):
        plt.axvline(x=transition, color='g', linestyle='--')
        plt.text(transition, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.1, 
                 f"P{i+1}→P{i+2}", rotation=90, verticalalignment='bottom')
    plt.legend()

    # Plot delays
    plt.subplot(2, 2, 2)
    plt.plot(episodes, metrics["delays"], label="Delays")
    # Add trend line
    z = np.polyfit(episodes, metrics["delays"], 1)
    p = np.poly1d(z)
    plt.plot(episodes, p(episodes), "r--", label=f"Trend (slope={z[0]:.2f})")
    plt.title("Total Delay Across All Phases")
    plt.xlabel("Total Episodes")
    plt.ylabel("Delay (minutes)")
    for i, transition in enumerate(phase_transitions):
        plt.axvline(x=transition, color='g', linestyle='--')
        plt.text(transition, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.1, 
                 f"P{i+1}→P{i+2}", rotation=90, verticalalignment='bottom')
    plt.legend()

    # Plot on-time rates
    plt.subplot(2, 2, 3)
    plt.plot(episodes, metrics["on_time_rates"], label="On-Time Rate")
    # Add trend line
    z = np.polyfit(episodes, metrics["on_time_rates"], 1)
    p = np.poly1d(z)
    plt.plot(episodes, p(episodes), "r--", label=f"Trend (slope={z[0]:.2f})")
    plt.title("On-Time Delivery Rate Across All Phases")
    plt.xlabel("Total Episodes")
    plt.ylabel("On-Time Rate (%)")
    plt.ylim(0, 100)
    for i, transition in enumerate(phase_transitions):
        plt.axvline(x=transition, color='g', linestyle='--')
        plt.text(transition, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.1, 
                 f"P{i+1}→P{i+2}", rotation=90, verticalalignment='bottom')
    plt.legend()

    # Plot postponement rates
    plt.subplot(2, 2, 4)
    plt.plot(episodes, metrics["postponement_rates"], label="Postponement Rate")
    # Add trend line
    z = np.polyfit(episodes, metrics["postponement_rates"], 1)
    p = np.poly1d(z)
    plt.plot(episodes, p(episodes), "r--", label=f"Trend (slope={z[0]:.2f})")
    plt.title("Postponement Rate Across All Phases")
    plt.xlabel("Total Episodes")
    plt.ylabel("Postponement Rate (%)")
    plt.ylim(0, 100)
    for i, transition in enumerate(phase_transitions):
        plt.axvline(x=transition, color='g', linestyle='--')
        plt.text(transition, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.1, 
                 f"P{i+1}→P{i+2}", rotation=90, verticalalignment='bottom')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"overall_results_{timestamp}.png"), dpi=300)
    plt.close()


def find_latest_model(model_dir="data/models"):
    """Find the latest phased training model and its resume information."""
    if not os.path.exists(model_dir):
        return None, 0, 0
    
    # Find phased training directories
    phased_dirs = [d for d in os.listdir(model_dir) if d.startswith("phased_training_")]
    if not phased_dirs:
        return None, 0, 0
    
    # Find the most recent directory
    phased_dirs.sort(reverse=True)
    latest_dir = os.path.join(model_dir, phased_dirs[0])
    
    # Try to find the latest final model from the highest phase
    for phase in range(10, 0, -1):  # Look from phase 10 down to 1
        final_model = os.path.join(latest_dir, f"rl_aca_phase{phase}_final.pt")
        if os.path.exists(final_model):
            logger.info(f"Found final model for phase {phase}: {final_model}")
            return final_model, phase-1, 0
    
    # If no final models found, try the latest.pt file
    latest_model = os.path.join(latest_dir, "rl_aca_latest.pt")
    if os.path.exists(latest_model):
        # Try to determine phase from resuming_info.json
        if os.path.exists(os.path.join(latest_dir, "resuming_info.json")):
            with open(os.path.join(latest_dir, "resuming_info.json"), "r") as f:
                resume_info = json.load(f)
                phase = resume_info.get("phase", 0)
                episode = resume_info.get("episode", 0)
                return latest_model, phase, episode
        return latest_model, 0, 0
    
    # Fallback: look for any pt files in the directory
    model_files = [f for f in os.listdir(latest_dir) if f.endswith('.pt')]
    if model_files:
        model_files.sort(reverse=True)  # Sort to get newest first
        model_path = os.path.join(latest_dir, model_files[0])
        
        # Try to extract phase from filename using regex
        import re
        phase_match = re.search(r'phase(\d+)', model_files[0])
        phase = int(phase_match.group(1))-1 if phase_match else 0
        
        return model_path, phase, 0
    
    return None, 0, 0


def evaluate_model(
    model_path: str,
    num_episodes: int = 10,
    seed: int = 100,
    visualize: bool = False,
    env_config=None,  # Added parameter for environment config
    ):
    """
    Evaluate a trained RL-ACA model over multiple episodes.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of evaluation episodes
        district_day: Tuple of (district_id, day) for Meituan data
        seed: Random seed for reproducibility
        visualize: Whether to enable visualization
        env_config: Optional environment configuration to use for evaluation
    """
    # Metrics to track
    metrics = {
        "total_rewards": [],
        "total_delays": [],
        "on_time_rates": [],
        "max_delays": []
    }
    
    logger.info(f"Evaluating model: {model_path}")
    
    # Use default environment if none provided
    env_params = env_config or {}
    
    # Evaluation loop
    for episode in tqdm(range(num_episodes), desc="Evaluating RL-ACA"):
        # Run an episode with training disabled
        stats = run_test_episode(
            solver_name="rl_aca",
            seed=seed + episode,
            reposition_idle_vehicles=False,
            visualize=visualize,
            rl_model_path=model_path,
            exploration_rate=0.00,
            env_config=env_params  # Pass as a single dictionary
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
    seed: int = 200,
    visualize: bool = False,
    env_config=None,  # Added parameter for environment config  
    ):
    """
    Compare the RL-based ACA with the original heuristic ACA.
    """
    logger.info("Comparing heuristic ACA vs RL-based ACA")
    
    # Use default environment if none provided
    env_params = env_config or {}
    
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
            seed=seed + episode,
            reposition_idle_vehicles=False,
            visualize=visualize,
            env_config=env_params  # Pass as a single parameter instead of unpacking
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
            seed=seed + episode,
            reposition_idle_vehicles=False,
            visualize=visualize,
            rl_model_path=rl_model_path,
            exploration_rate=0.00,
            env_config=env_params  # Pass as a single parameter instead of unpacking
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
    
    # Create comparison plots as line charts
    plt.figure(figsize=(15, 12))

    # Get seed numbers for x-axis
    seed_numbers = [seed + i for i in range(max(heuristic_episodes, rl_episodes))]
    heuristic_x = seed_numbers[:heuristic_episodes]
    rl_x = seed_numbers[:rl_episodes]

    # Plot rewards comparison
    plt.subplot(2, 2, 1)
    plt.plot(heuristic_x, heuristic_metrics["total_rewards"], 'o-', label="Heuristic")
    plt.plot(rl_x, rl_metrics["total_rewards"], 'o-', label="RL")

    # Add trend lines
    if len(heuristic_x) > 1:
        h_z = np.polyfit(heuristic_x, heuristic_metrics["total_rewards"], 1)
        h_p = np.poly1d(h_z)
        plt.plot(heuristic_x, h_p(heuristic_x), "r--", label=f"Heuristic Trend (slope={h_z[0]:.2f})")

    if len(rl_x) > 1:
        rl_z = np.polyfit(rl_x, rl_metrics["total_rewards"], 1)
        rl_p = np.poly1d(rl_z)
        plt.plot(rl_x, rl_p(rl_x), "g--", label=f"RL Trend (slope={rl_z[0]:.2f})")

    plt.title("Rewards by Seed")
    plt.xlabel("Seed")
    plt.ylabel("Total Reward")
    plt.legend()

    # Plot delay comparison
    plt.subplot(2, 2, 2)
    plt.plot(heuristic_x, heuristic_metrics["total_delays"], 'o-', label="Heuristic")
    plt.plot(rl_x, rl_metrics["total_delays"], 'o-', label="RL")

    # Add trend lines
    if len(heuristic_x) > 1:
        h_z = np.polyfit(heuristic_x, heuristic_metrics["total_delays"], 1)
        h_p = np.poly1d(h_z)
        plt.plot(heuristic_x, h_p(heuristic_x), "r--", label=f"Heuristic Trend (slope={h_z[0]:.2f})")

    if len(rl_x) > 1:
        rl_z = np.polyfit(rl_x, rl_metrics["total_delays"], 1)
        rl_p = np.poly1d(rl_z)
        plt.plot(rl_x, rl_p(rl_x), "g--", label=f"RL Trend (slope={rl_z[0]:.2f})")

    plt.title("Total Delay by Seed")
    plt.xlabel("Seed")
    plt.ylabel("Total Delay (minutes)")
    plt.legend()

    # Plot on-time rate comparison
    plt.subplot(2, 2, 3)
    plt.plot(heuristic_x, heuristic_metrics["on_time_rates"], 'o-', label="Heuristic")
    plt.plot(rl_x, rl_metrics["on_time_rates"], 'o-', label="RL")

    # Add trend lines
    if len(heuristic_x) > 1:
        h_z = np.polyfit(heuristic_x, heuristic_metrics["on_time_rates"], 1)
        h_p = np.poly1d(h_z)
        plt.plot(heuristic_x, h_p(heuristic_x), "r--", label=f"Heuristic Trend (slope={h_z[0]:.2f})")

    if len(rl_x) > 1:
        rl_z = np.polyfit(rl_x, rl_metrics["on_time_rates"], 1)
        rl_p = np.poly1d(rl_z)
        plt.plot(rl_x, rl_p(rl_x), "g--", label=f"RL Trend (slope={rl_z[0]:.2f})")

    plt.title("On-Time Rate by Seed")
    plt.xlabel("Seed")
    plt.ylabel("On-Time Rate (%)")
    plt.ylim(0, 100)
    plt.legend()

    # Plot postponement rate comparison
    plt.subplot(2, 2, 4)
    plt.plot(heuristic_x, heuristic_metrics["postponement_rates"], 'o-', label="Heuristic")
    plt.plot(rl_x, rl_metrics["postponement_rates"], 'o-', label="RL")

    # Add trend lines
    if len(heuristic_x) > 1:
        h_z = np.polyfit(heuristic_x, heuristic_metrics["postponement_rates"], 1)
        h_p = np.poly1d(h_z)
        plt.plot(heuristic_x, h_p(heuristic_x), "r--", label=f"Heuristic Trend (slope={h_z[0]:.2f})")

    if len(rl_x) > 1:
        rl_z = np.polyfit(rl_x, rl_metrics["postponement_rates"], 1)
        rl_p = np.poly1d(rl_z)
        plt.plot(rl_x, rl_p(rl_x), "g--", label=f"RL Trend (slope={rl_z[0]:.2f})")

    plt.title("Postponement Rate by Seed")
    plt.xlabel("Seed")
    plt.ylabel("Postponement Rate (%)")
    plt.ylim(0, 100)
    plt.legend()
    
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


def plot_losses(losses, save_path, window_size=20, phase_idx=None, episode_idx=None, total_steps=None):
    """
    Plot the training losses and overwrite the existing plot.
    
    Args:
        losses: List of loss values
        save_path: Path to save the plot (will be overwritten)
        window_size: Size of window for smoothing
        phase_idx: Current phase index (for labeling)
        episode_idx: Current episode index (for labeling)
        total_steps: Total number of training steps (for labeling)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    if not losses:
        logger.warning("No loss data available to plot")
        return
    
    iterations = list(range(len(losses)))
    
    plt.figure(figsize=(12, 6))
    
    # Plot raw losses
    plt.plot(iterations, losses, 'b-', alpha=0.3, label='Raw Loss')
    
    # Dynamic window size: use the smaller of window_size or 10% of total steps
    dynamic_window = min(window_size, max(1, len(losses) // 10))
    if len(losses) > dynamic_window:
        smoothed_losses = []
        for i in range(len(losses) - dynamic_window + 1):
            smoothed_losses.append(sum(losses[i:i+dynamic_window]) / dynamic_window)
        
        smoothed_x = list(range(dynamic_window - 1, len(losses)))
        plt.plot(smoothed_x, smoothed_losses, 'r-', linewidth=2, 
                label=f'Moving Average (window={dynamic_window})')
    
    # Add titles and labels
    title = f'Training Loss Over Time (Total Steps: {total_steps})'
    if phase_idx is not None and episode_idx is not None:
        title += f'\nPhase {phase_idx}, Episode {episode_idx}'
    plt.title(title)
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    if len(losses) > 0:
        recent_losses = losses[-min(100, len(losses)):]
        plt.figtext(0.01, 0.01, 
                    f'Recent stats (last {len(recent_losses)}):\n'
                    f'Min: {min(recent_losses):.4f}\n'
                    f'Max: {max(recent_losses):.4f}\n'
                    f'Mean: {sum(recent_losses)/len(recent_losses):.4f}',
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the plot with error handling and retry
    retries = 3
    base, ext = os.path.splitext(save_path)  # Split into 'loss_plot' and '.png'
    for attempt in range(retries):
        try:
            # Use a temporary file with proper extension
            temp_path = f"{base}_temp_{attempt}{ext}"  # e.g., 'loss_plot_temp_0.png'
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            os.replace(temp_path, save_path)  # Move to final path
            logger.debug(f"Loss plot saved to {save_path}")
            plt.close()
            break
        except Exception as e:
            logger.warning(f"Failed to save loss plot (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                logger.error(f"Could not save loss plot after {retries} attempts. Continuing training.")
                plt.close()
                return
    
    plt.close()  # Ensure figure is closed


def define_training_phases():
    """
    Define the progressive training phases with increasing complexity.
    """
    phases = [
        # Phase 1: Simple Environment
        {
            "name": "Simple Environment",
            "env_config": {
                "num_vehicles": 10,  # Start with just 2 vehicles
                "num_restaurants": 20,  # Limited restaurants
                "service_area_dimensions": (6.0, 6.0),  # Small area
                "mean_interarrival_time": 8,  # Low order density
            },
            "performance_criteria": {
                # No performance criteria - phase will run until max_episodes
            },
            "min_episodes": 20, 
            "max_episodes": 500    # More episodes for initial learning
        }
        # ,
        
        # # Phase 2: Intermediate Environment
        # {
        #     "name": "Intermediate Environment",
        #     "env_config": {
        #         "num_vehicles": 15,  # Increase to 5 vehicles
        #         "num_restaurants": 15,  # More restaurants
        #         "service_area_dimensions": (6.0, 6.0),  # Larger area
        #         "mean_interarrival_time": 1.5,  # Medium order density
        #     },
        #     "performance_criteria": {
        #         # No performance criteria - phase will run until max_episodes
        #     },
        #     "min_episodes": 20,  # 30, 100
        #     "max_episodes": 200   # Substantial training in intermediate complexity
        # },
        
        # # Phase 3: Full Environment
        # {
        #     "name": "Full Environment",
        #     "env_config": {
        #         "num_vehicles": 30,  # Full fleet
        #         "num_restaurants": 30,  # All restaurants
        #         "service_area_dimensions": (6.0, 6.0),  # Complete service area
        #         "mean_interarrival_time": 0.65,  # High order density
        #     },
        #     "performance_criteria": {
        #         # No performance criteria - phase will run until max_episodes
        #     },
        #     "min_episodes": 20,  # 50, 300
        #     "max_episodes": 200   # Extensive training in full complexity
        # }
    ]
    
    return phases


if __name__ == "__main__":
    # Check for command-line arguments for custom configuration
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Train RL-ACA with phased curriculum learning")
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--stability_window', type=int, default=10, help='Window size for stability check')
    parser.add_argument('--stability_threshold', type=float, default=3.0, help='Percentage threshold for stability')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compare', action='store_true', help='Compare with heuristic ACA after training')
    parser.add_argument('--compare-only', action='store_true', help='Only compare existing model with heuristic ACA (no training)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to existing model for comparison (required with --compare-only)')
    parser.add_argument('--no-compare', action='store_true', help='Do not compare with heuristic ACA after training')
    parser.add_argument('--initial-exploration', type=float, default=0.9, help='Initial exploration rate (default: 0.9)')
    parser.add_argument('--min-exploration', type=float, default=0.01, help='Minimum exploration rate (default: 0.05)')
    parser.add_argument('--decay-method', type=str, choices=['linear', 'exponential'], default='exponential', help='Exploration rate decay method (default: linear)')
    parser.add_argument('--decay-rate', type=float, default=0.99, help='Decay rate for exponential decay (default: 0.995)')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    seed = args.seed
    
    # Define phases for either training or comparison environment setup
    phases = define_training_phases()
    
    if args.compare_only:
        # Skip training and go straight to comparison
        model_path = args.model_path
        
        if not model_path:
            # Find latest model if no specific path provided
            model_path, _, _ = find_latest_model()
            if not model_path:
                logger.error("No model path provided and no existing model found. Cannot compare.")
                sys.exit(1)
            else:
                logger.info(f"Using latest model found: {model_path}")
        
        # Use final phase environment for comparison
        env_config = phases[-1]["env_config"]
        
        logger.info(f"Comparing models using existing model at: {model_path}")
        compare_models(
            heuristic_episodes=100,
            rl_episodes=100,
            rl_model_path=model_path,
            seed=seed,
            visualize=args.visualize,
            env_config=env_config
        )
    else:
        # Regular training mode
        # Check for existing models to resume from if requested
        if args.resume:
            latest_model, phase, episode = find_latest_model()
            
            if latest_model and os.path.exists(latest_model):
                logger.info(f"Found existing model: {latest_model}")
                logger.info(f"Resuming from phase {phase+1}, episode {episode}")
                resume = True
            else:
                logger.warning("No existing model found. Starting from scratch.")
                resume = False
                latest_model = None
                phase = 0
                episode = 0
        else:
            resume = False
            latest_model = None
            phase = 0
            episode = 0
        
        # Run phased training
        final_model_path = train_rl_aca(
            phases=phases,
            save_interval=args.save_interval,
            stability_window=args.stability_window,
            stability_threshold=args.stability_threshold,
            seed=seed,
            visualize=args.visualize,
            reposition_idle_vehicles=True,
            model_dir="data/models",
            resume_from_model=latest_model if resume else None,
            start_phase=phase if resume else 0,
            start_episode=episode if resume else 0,
            exploration_start=args.initial_exploration,
            decay_method=args.decay_method,
            # Tuned RL hyperparameters
            decay_rate=0.99,
            rl_learning_rate=0.0004,
            rl_batch_size=32,
            rl_target_update_frequency=75,
            rl_discount_factor=0.9,
            rl_exploration_decay=0.99,
            exploration_end=0.05,
            rl_bundling_reward=0.0,
            rl_postponement_penalty=-0.0,
            rl_on_time_reward=0.0
        )

        # Always compare after training, unless explicitly turned off
        if not hasattr(args, 'no_compare') or not args.no_compare:
            compare_models(
                heuristic_episodes=100,
                rl_episodes=100,
                rl_model_path=final_model_path,
                seed=seed,
                visualize=args.visualize,
                env_config=phases[-1]["env_config"]
            )


# --------------- Usage ---------------
# # Basic usage
# python train_rl.py

# # Resume from previous checkpoint
# python train_rl.py --resume

# # Customize stability criteria
# python train_rl.py --stability_window 15 --stability_threshold 2.5

# # Enable visualization
# python train_rl.py --visualize

# # Compare with heuristic ACA after training
# python train_rl.py --compare

# # Only compare using latest model without training
# python train_rl.py --compare-only

# # Only compare using a specific model without training
# python train_rl.py --compare-only --model-path path/to/your/model.pt
# Model path example: 