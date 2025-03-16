# train_rl.py - Simplified to focus on phased training
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
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


# def train_rl_aca(
#     phases: List[Dict],
#     save_interval: int = 20,
#     stability_window: int = 10,       # Window for stability assessment
#     stability_threshold: float = 3.0,  # Max percent change considered stable
#     seed: int = 42,
#     visualize: bool = False,
#     reposition_idle_vehicles: bool = False,
#     model_dir: str = "data/models",
#     resume_from_model: str = None,    # Support resuming
#     start_phase: int = 0,             # Phase to start from when resuming
#     start_episode: int = 0,           # Episode to start from when resuming
# ):
#     """
#     Train the RL-based ACA solver through multiple progressive phases.
    
#     Args:
#         phases: List of phase configurations defining the progression
#         save_interval: Save model every N episodes
#         stability_window: Number of recent episodes to check for stability
#         stability_threshold: Maximum percentage change in reward to be considered stable
#         district_day: Tuple of (district_id, day) for Meituan data
#         seed: Random seed for reproducibility
#         visualize: Whether to enable visualization
#         reposition_idle_vehicles: Whether to reposition idle vehicles
#         model_dir: Directory to save models
#         resume_from_model: Path to model to resume training from
#         start_phase: Phase index to start from when resuming
#         start_episode: Episode number to start from within the start phase
#     """
#     # Create model directory if it doesn't exist
#     os.makedirs(model_dir, exist_ok=True)
    
#     # Generate timestamp for this training run
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # Set initial model path if resuming
#     initial_model_path = resume_from_model if resume_from_model else None
#     current_model_path = initial_model_path
    
#     # Create directory for phase-specific data
#     phase_dir = os.path.join(model_dir, f"phased_training_{timestamp}")
#     os.makedirs(phase_dir, exist_ok=True)
    
#     # Save phase configurations for reference
#     with open(os.path.join(phase_dir, "phase_configs.json"), "w") as f:
#         json.dump(phases, f, indent=4)
    
#     # Initialize overall metrics
#     all_metrics = {
#         "phase_transitions": [],
#         "rewards": [],
#         "delays": [],
#         "on_time_rates": [],
#         "postponement_rates": []
#     }
    
#     # If resuming, attempt to load metrics from previous runs
#     if resume_from_model:
#         metrics_file = os.path.join(model_dir, f"phased_metrics_{timestamp}.npz")
#         if os.path.exists(metrics_file):
#             saved_metrics = np.load(metrics_file, allow_pickle=True)
#             all_metrics = {
#                 "phase_transitions": saved_metrics["phase_transitions"].tolist(),
#                 "rewards": saved_metrics["rewards"].tolist(),
#                 "delays": saved_metrics["delays"].tolist(),
#                 "on_time_rates": saved_metrics["on_time_rates"].tolist(),
#                 "postponement_rates": saved_metrics["postponement_rates"].tolist(),
#             }
#             logger.info(f"Resuming training from phase {start_phase}, episode {start_episode}")
    
#     # Training loop through phases
#     current_phase_idx = start_phase
    
#     while current_phase_idx < len(phases):
#         current_phase = phases[current_phase_idx]
#         phase_name = current_phase.get("name", f"Phase {current_phase_idx + 1}")
        
#         logger.info(f"\n{'=' * 80}\nStarting phase {current_phase_idx + 1}/{len(phases)}: {phase_name}")
#         logger.info(f"Environment: {current_phase['env_config']}")
#         logger.info(f"Performance criteria: {current_phase['performance_criteria']}\n{'=' * 80}")
        
#         # Initialize phase-specific metrics
#         phase_metrics = {
#             "rewards": [],
#             "delays": [],
#             "on_time_rates": [],
#             "postponement_rates": []
#         }
        
#         # Process episodes for the current phase
#         episode_in_phase = start_episode if current_phase_idx == start_phase else 0
#         phase_complete = False
#         min_episodes = current_phase.get("min_episodes", 20)
#         max_episodes = current_phase.get("max_episodes", 100)
        
#         while not phase_complete:
#             # Prepare model path for this episode
#             model_path = f"{phase_dir}/rl_aca_phase{current_phase_idx+1}_{timestamp}_latest.pt"
            
#             # Use the appropriate model
#             if episode_in_phase == 0:
#                 model_to_use = current_model_path
#             else:
#                 model_to_use = model_path if os.path.exists(model_path) else current_model_path
            
#             # Run an episode with the current phase configuration
#             logger.info(f"Running episode {episode_in_phase + 1} in phase {current_phase_idx + 1}")
            
#             # Pass phase-specific environment parameters
#             env_config = current_phase["env_config"]
            
#             # Run test episode
#             stats = run_test_episode(
#                 solver_name="rl_aca",
#                 seed=seed + episode_in_phase,
#                 reposition_idle_vehicles=reposition_idle_vehicles,
#                 visualize=visualize and episode_in_phase % 20 == 0,
#                 save_rl_model=True,
#                 rl_model_path=model_to_use,
#                 save_results_to_disk=False,
#                 env_config=env_config  # Pass as single parameter
#             )
                        
#             # Update phase metrics
#             reward = stats["total_reward"]
#             delay = sum(stats["delay_values"]) if stats["delay_values"] else 0
            
#             total_orders = max(1, stats["orders_delivered"])
#             late_orders = len(stats["late_orders"])
#             on_time_rate = ((total_orders - late_orders) / total_orders) * 100
            
#             postponement_rate = len(stats["postponed_orders"]) / max(1, stats["total_orders"]) * 100
            
#             phase_metrics["rewards"].append(reward)
#             phase_metrics["delays"].append(delay)
#             phase_metrics["on_time_rates"].append(on_time_rate)
#             phase_metrics["postponement_rates"].append(postponement_rate)
            
#             # Update overall metrics
#             all_metrics["rewards"].append(reward)
#             all_metrics["delays"].append(delay)
#             all_metrics["on_time_rates"].append(on_time_rate)
#             all_metrics["postponement_rates"].append(postponement_rate)
            
#             # Log progress
#             if (episode_in_phase + 1) % 5 == 0:
#                 logger.info(f"Phase {current_phase_idx + 1} - Episode {episode_in_phase + 1}: "
#                            f"Reward: {reward:.2f}, "
#                            f"On-time rate: {on_time_rate:.2f}%, "
#                            f"Postponement rate: {postponement_rate:.2f}%")
            
#             # Save model at specified intervals
#             if (episode_in_phase + 1) % save_interval == 0:
#                 checkpoint_path = f"{phase_dir}/rl_aca_phase{current_phase_idx+1}_ep{episode_in_phase+1}.pt"
#                 if os.path.exists(model_path):
#                     shutil.copy(model_path, checkpoint_path)
#                     logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
#                     # Save progress information for resuming
#                     resume_info = {
#                         "phase": current_phase_idx,
#                         "episode": episode_in_phase + 1,
#                         "timestamp": timestamp,
#                     }
#                     with open(f"{phase_dir}/resuming_info.json", "w") as f:
#                         json.dump(resume_info, f)
                    
#                     # Save all metrics
#                     np.savez(
#                         os.path.join(phase_dir, f"phased_metrics_{timestamp}.npz"), 
#                         phase_transitions=np.array(all_metrics["phase_transitions"]),
#                         rewards=np.array(all_metrics["rewards"]),
#                         delays=np.array(all_metrics["delays"]),
#                         on_time_rates=np.array(all_metrics["on_time_rates"]),
#                         postponement_rates=np.array(all_metrics["postponement_rates"])
#                     )
            
#             # Check if phase completion criteria are met
#             criteria_met, reason = check_phase_criteria(
#                 phase_metrics=phase_metrics,
#                 episode_count=episode_in_phase + 1,
#                 min_episodes=min_episodes,
#                 max_episodes=max_episodes,
#                 stability_window=stability_window,
#                 stability_threshold=stability_threshold,
#                 performance_criteria=current_phase["performance_criteria"]
#             )
            
#             if criteria_met:
#                 phase_complete = True
#                 current_model_path = model_path  # Use this model for the next phase
                
#                 # Save phase transition point
#                 all_metrics["phase_transitions"].append(len(all_metrics["rewards"]) - 1)
                
#                 # Save phase-completion model
#                 phase_final_path = f"{phase_dir}/rl_aca_phase{current_phase_idx+1}_final.pt"
#                 if os.path.exists(model_path):
#                     shutil.copy(model_path, phase_final_path)
                    
#                 logger.info(f"Phase {current_phase_idx + 1} completed after {episode_in_phase + 1} episodes")
#                 logger.info(f"Reason: {reason}")
#                 logger.info(f"Phase model saved to {phase_final_path}")
                
#                 # Plot phase results
#                 plot_phase_results(phase_metrics, current_phase_idx, phase_name, phase_dir, timestamp)
            
#             # Increment episode counter
#             episode_in_phase += 1
        
#         # Move to next phase
#         current_phase_idx += 1
#         start_episode = 0  # Reset episode counter for next phase
    
#     # Training complete - plot overall results
#     plot_training_results(all_metrics, phases, phase_dir, timestamp)
    
#     # Get path to final model
#     final_model_path = f"{phase_dir}/rl_aca_phase{len(phases)}_final.pt"
    
#     logger.info(f"Phased training completed. Final model saved to {final_model_path}")
#     return final_model_path



def train_rl_aca(
    phases: List[Dict],
    save_interval: int = 20,
    stability_window: int = 10,       # Window for stability assessment
    stability_threshold: float = 3.0,  # Max percent change considered stable
    seed: int = 42,
    visualize: bool = False,
    reposition_idle_vehicles: bool = False,
    model_dir: str = "data/models",
    resume_from_model: str = None,    # Support resuming
    start_phase: int = 0,             # Phase to start from when resuming
    start_episode: int = 0,           # Episode to start from when resuming
):
    """
    Train the RL-based ACA solver through multiple progressive phases.
    
    Args:
        phases: List of phase configurations defining the progression
        save_interval: Save model every N episodes
        stability_window: Number of recent episodes to check for stability
        stability_threshold: Maximum percentage change in reward to be considered stable
        seed: Random seed for reproducibility
        visualize: Whether to enable visualization
        reposition_idle_vehicles: Whether to reposition idle vehicles
        model_dir: Directory to save models
        resume_from_model: Path to model to resume training from
        start_phase: Phase index to start from when resuming
        start_episode: Episode number to start from within the start phase
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set initial model path if resuming
    initial_model_path = resume_from_model if resume_from_model else None
    current_model_path = initial_model_path
    
    # Create directory for phase-specific data
    phase_dir = os.path.join(model_dir, f"phased_training_{timestamp}")
    os.makedirs(phase_dir, exist_ok=True)
    
    # Save phase configurations for reference
    with open(os.path.join(phase_dir, "phase_configs.json"), "w") as f:
        json.dump(phases, f, indent=4)
    
    # Initialize overall metrics
    all_metrics = {
        "phase_transitions": [],
        "rewards": [],
        "delays": [],
        "on_time_rates": [],
        "postponement_rates": []
    }
    
    # If resuming, attempt to load metrics from previous runs
    if resume_from_model:
        metrics_file = os.path.join(model_dir, f"phased_metrics_{timestamp}.npz")
        if os.path.exists(metrics_file):
            saved_metrics = np.load(metrics_file, allow_pickle=True)
            all_metrics = {
                "phase_transitions": saved_metrics["phase_transitions"].tolist(),
                "rewards": saved_metrics["rewards"].tolist(),
                "delays": saved_metrics["delays"].tolist(),
                "on_time_rates": saved_metrics["on_time_rates"].tolist(),
                "postponement_rates": saved_metrics["postponement_rates"].tolist(),
            }
            logger.info(f"Resuming training from phase {start_phase}, episode {start_episode}")
    
    # Training loop through phases
    current_phase_idx = start_phase
    
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
        # The total is max_episodes - episode_in_phase to account for resuming
        total_possible = max_episodes - episode_in_phase
        with tqdm(total=total_possible, desc=f"Phase {current_phase_idx + 1}/{len(phases)}: {phase_name}") as pbar:
            while not phase_complete:
                # Prepare model path for this episode
                model_path = f"{phase_dir}/rl_aca_phase{current_phase_idx+1}_{timestamp}_latest.pt"
                
                # Use the appropriate model
                if episode_in_phase == 0:
                    model_to_use = current_model_path
                else:
                    model_to_use = model_path if os.path.exists(model_path) else current_model_path
                
                # Run an episode with the current phase configuration
                # We'll remove this log since we have the progress bar now
                # logger.info(f"Running episode {episode_in_phase + 1} in phase {current_phase_idx + 1}")
                
                # Pass phase-specific environment parameters
                env_config = current_phase["env_config"]
                
                # Run test episode
                stats = run_test_episode(
                    solver_name="rl_aca",
                    seed=seed + episode_in_phase,
                    reposition_idle_vehicles=reposition_idle_vehicles,
                    visualize=visualize and episode_in_phase % 20 == 0,
                    save_rl_model=True,
                    rl_model_path=model_to_use,
                    save_results_to_disk=False,
                    env_config=env_config  # Pass as single parameter
                )
                            
                # Update phase metrics
                reward = stats["total_reward"]
                delay = sum(stats["delay_values"]) if stats["delay_values"] else 0
                
                total_orders = max(1, stats["orders_delivered"])
                late_orders = len(stats["late_orders"])
                on_time_rate = ((total_orders - late_orders) / total_orders) * 100
                
                postponement_rate = len(stats["postponed_orders"]) / max(1, stats["total_orders"]) * 100
                
                phase_metrics["rewards"].append(reward)
                phase_metrics["delays"].append(delay)
                phase_metrics["on_time_rates"].append(on_time_rate)
                phase_metrics["postponement_rates"].append(postponement_rate)
                
                # Update overall metrics
                all_metrics["rewards"].append(reward)
                all_metrics["delays"].append(delay)
                all_metrics["on_time_rates"].append(on_time_rate)
                all_metrics["postponement_rates"].append(postponement_rate)
                
                # Update progress bar with key metrics instead of logging
                pbar.set_postfix({
                    'reward': f"{reward:.2f}", 
                    'on-time': f"{on_time_rate:.1f}%",
                    'delay': f"{delay:.1f}"
                })
                pbar.update(1)
                
                # We can keep occasional logging for save points
                if (episode_in_phase + 1) % save_interval == 0:
                    checkpoint_path = f"{phase_dir}/rl_aca_phase{current_phase_idx+1}_ep{episode_in_phase+1}.pt"
                    if os.path.exists(model_path):
                        shutil.copy(model_path, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                        
                        # Save progress information for resuming
                        resume_info = {
                            "phase": current_phase_idx,
                            "episode": episode_in_phase + 1,
                            "timestamp": timestamp,
                        }
                        with open(f"{phase_dir}/resuming_info.json", "w") as f:
                            json.dump(resume_info, f)
                        
                        # Save all metrics
                        np.savez(
                            os.path.join(phase_dir, f"phased_metrics_{timestamp}.npz"), 
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
                
                if criteria_met:
                    phase_complete = True
                    current_model_path = model_path  # Use this model for the next phase
                    
                    # Save phase transition point
                    all_metrics["phase_transitions"].append(len(all_metrics["rewards"]) - 1)
                    
                    # Save phase-completion model
                    phase_final_path = f"{phase_dir}/rl_aca_phase{current_phase_idx+1}_final.pt"
                    if os.path.exists(model_path):
                        shutil.copy(model_path, phase_final_path)
                        
                    logger.info(f"Phase {current_phase_idx + 1} completed after {episode_in_phase + 1} episodes")
                    logger.info(f"Reason: {reason}")
                    logger.info(f"Phase model saved to {phase_final_path}")
                    
                    # Plot phase results
                    plot_phase_results(phase_metrics, current_phase_idx, phase_name, phase_dir, timestamp)
                                
                # Break the loop if we've reached the maximum episodes
                if episode_in_phase >= max_episodes:
                    phase_complete = True
                    
                # Increment episode counter
                episode_in_phase += 1
        
        # Move to next phase
        current_phase_idx += 1
        start_episode = 0  # Reset episode counter for next phase
    
    # Training complete - plot overall results
    plot_training_results(all_metrics, phases, phase_dir, timestamp)
    
    # Get path to final model
    final_model_path = f"{phase_dir}/rl_aca_phase{len(phases)}_final.pt"
    
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
    
    # Check reward stability
    first_half = recent_rewards[:stability_window//2]
    second_half = recent_rewards[stability_window//2:]
    
    avg_first = np.mean(first_half)
    avg_second = np.mean(second_half)
    
    if avg_first != 0:  # Avoid division by zero
        percent_change = abs((avg_second - avg_first) / avg_first * 100.0)
    else:
        percent_change = 100.0 if avg_second != 0 else 0.0
    
    if percent_change > stability_threshold:
        return False, f"Reward not stable: {percent_change:.2f}% change > {stability_threshold:.2f}% threshold"
    
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
    """Plot results for a single training phase."""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics["rewards"])
    plt.title(f"Phase {phase_idx + 1}: {phase_name} - Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    # Plot delays
    plt.subplot(2, 2, 2)
    plt.plot(metrics["delays"])
    plt.title(f"Phase {phase_idx + 1}: {phase_name} - Delays")
    plt.xlabel("Episode")
    plt.ylabel("Total Delay (minutes)")
    
    # Plot on-time rates
    plt.subplot(2, 2, 3)
    plt.plot(metrics["on_time_rates"])
    plt.title(f"Phase {phase_idx + 1}: {phase_name} - On-Time Rate")
    plt.xlabel("Episode")
    plt.ylabel("On-Time Rate (%)")
    plt.ylim(0, 100)
    
    # Plot postponement rates
    plt.subplot(2, 2, 4)
    plt.plot(metrics["postponement_rates"])
    plt.title(f"Phase {phase_idx + 1}: {phase_name} - Postponement Rate")
    plt.xlabel("Episode")
    plt.ylabel("Postponement Rate (%)")
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"phase{phase_idx+1}_results_{timestamp}.png"), dpi=300)
    plt.close()


def plot_training_results(metrics, phases, output_dir, timestamp):
    """Plot overall training results across all phases."""
    plt.figure(figsize=(15, 12))
    
    # Get phase transition points
    phase_transitions = metrics["phase_transitions"]
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics["rewards"])
    plt.title("Total Reward Across All Phases")
    plt.xlabel("Total Episodes")
    plt.ylabel("Reward")
    
    # Add vertical lines for phase transitions
    for i, transition in enumerate(phase_transitions):
        plt.axvline(x=transition, color='r', linestyle='--')
        plt.text(transition, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.1, 
                 f"P{i+1}→P{i+2}", 
                 rotation=90, verticalalignment='bottom')
    
    # Plot delays
    plt.subplot(2, 2, 2)
    plt.plot(metrics["delays"])
    plt.title("Total Delay Across All Phases")
    plt.xlabel("Total Episodes")
    plt.ylabel("Delay (minutes)")
    
    # Add vertical lines for phase transitions
    for i, transition in enumerate(phase_transitions):
        plt.axvline(x=transition, color='r', linestyle='--')
        plt.text(transition, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.1, 
                 f"P{i+1}→P{i+2}", 
                 rotation=90, verticalalignment='bottom')
    
    # Plot on-time rates
    plt.subplot(2, 2, 3)
    plt.plot(metrics["on_time_rates"])
    plt.title("On-Time Delivery Rate Across All Phases")
    plt.xlabel("Total Episodes")
    plt.ylabel("On-Time Rate (%)")
    plt.ylim(0, 100)
    
    # Add vertical lines for phase transitions
    for i, transition in enumerate(phase_transitions):
        plt.axvline(x=transition, color='r', linestyle='--')
        plt.text(transition, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.1, 
                 f"P{i+1}→P{i+2}", 
                 rotation=90, verticalalignment='bottom')
    
    # Plot postponement rates
    plt.subplot(2, 2, 4)
    plt.plot(metrics["postponement_rates"])
    plt.title("Postponement Rate Across All Phases")
    plt.xlabel("Total Episodes")
    plt.ylabel("Postponement Rate (%)")
    plt.ylim(0, 100)
    
    # Add vertical lines for phase transitions
    for i, transition in enumerate(phase_transitions):
        plt.axvline(x=transition, color='r', linestyle='--')
        plt.text(transition, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.1, 
                 f"P{i+1}→P{i+2}", 
                 rotation=90, verticalalignment='bottom')
    
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
    
    # Find the most recent
    phased_dirs.sort(reverse=True)
    latest_dir = os.path.join(model_dir, phased_dirs[0])
    
    # Check for resume info
    if os.path.exists(os.path.join(latest_dir, "resuming_info.json")):
        with open(os.path.join(latest_dir, "resuming_info.json"), "r") as f:
            resume_info = json.load(f)
            
        phase = resume_info.get("phase", 0)
        episode = resume_info.get("episode", 0)
        timestamp = resume_info.get("timestamp")
        
        # Find the latest model
        model_path = os.path.join(latest_dir, f"rl_aca_phase{phase+1}_{timestamp}_latest.pt")
        if os.path.exists(model_path):
            return model_path, phase, episode
    
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
            seed=seed + episode,  # Different seed each episode 
            reposition_idle_vehicles=False,
            visualize=visualize,
            rl_model_path=model_path,
            **env_params  # Pass environment parameters
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
    
    Args:
        heuristic_episodes: Number of episodes to run with heuristic ACA
        rl_episodes: Number of episodes to run with RL-based ACA
        rl_model_path: Path to the trained RL model
        district_day: Tuple of (district_id, day) for Meituan data
        seed: Random seed for reproducibility
        env_config: Optional environment configuration to use for comparison
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
            **env_params  # Pass environment parameters
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
            **env_params  # Pass environment parameters
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


# Define sample phases for training
def define_training_phases():
    """
    Define the progressive training phases with increasing complexity.
    
    Returns:
        List of phase configurations
    """
    phases = [
        # Phase 1: Simple Environment
        {
            "name": "Simple Environment",
            "env_config": {
                "num_vehicles": 2,  # Start with just 2 vehicles
                "num_restaurants": 10,  # Limited restaurants
                "service_area_dimensions": (3.0, 3.0),  # Small area
                "mean_interarrival_time": 4.0,  # Low order density
            },
            "performance_criteria": {
                "min_on_time_rate": 80.0,  # High on-time rate required
                "max_avg_delay": 5.0,      # Low delay requirement
                "min_avg_reward": -10.0    # Reasonable reward target
            },
            "min_episodes": 20,   # Minimum episodes before advancement
            "max_episodes": 50    # Maximum episodes in this phase
        },
        
        # Phase 2: Intermediate Environment
        {
            "name": "Intermediate Environment",
            "env_config": {
                "num_vehicles": 5,  # Increase to 5 vehicles
                "num_restaurants": 30,  # More restaurants
                "service_area_dimensions": (5.0, 5.0),  # Larger area
                "mean_interarrival_time": 2.0,  # Medium order density
            },
            "performance_criteria": {
                "min_on_time_rate": 75.0,  # Slightly relaxed on-time requirement
                "max_avg_delay": 8.0,      # Allow slightly more delay
                "min_avg_reward": -15.0    # Adjusted reward target
            },
            "min_episodes": 30,
            "max_episodes": 70
        },
        
        # Phase 3: Full Environment
        {
            "name": "Full Environment",
            "env_config": {
                "num_vehicles": 15,  # Full fleet
                "num_restaurants": 110,  # All restaurants
                "service_area_dimensions": (10.0, 10.0),  # Complete service area
                "mean_interarrival_time": 1.5,  # High order density
            },
            "performance_criteria": {
                "min_on_time_rate": 70.0,  # Further relaxed requirements for complex scenario
                "max_avg_delay": 10.0,
                "min_avg_reward": -25.0
            },
            "min_episodes": 50,
            "max_episodes": 100
        }
    ]
    
    return phases


if __name__ == "__main__":
    # Check for command-line arguments for custom configuration
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL-ACA with phased curriculum learning")
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--stability_window', type=int, default=10, help='Window size for stability check')
    parser.add_argument('--stability_threshold', type=float, default=3.0, help='Percentage threshold for stability')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compare', action='store_true', help='Compare with heuristic ACA after training')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    seed = args.seed
    
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
    
    # Define phases and run phased training
    phases = define_training_phases()
    
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
        start_episode=episode if resume else 0
    )
    
    # Compare with heuristic ACA if requested
    if args.compare:
        compare_models(
            heuristic_episodes=5,
            rl_episodes=5,
            rl_model_path=final_model_path,
            seed=seed + 100,  # Different seed for comparison
            visualize=args.visualize,
            env_config=phases[-1]["env_config"]  # Use final phase environment config
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