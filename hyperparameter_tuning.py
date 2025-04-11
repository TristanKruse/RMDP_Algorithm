import random
import train_rl


def run_hyperparameter_tuning(num_experiments=20, episodes_per_experiment=500):
    hyperparameter_space = {
        "learning_rate": [0.0003, 0.0004, 0.0005],
        "batch_size": [32, 64, 96],
        "target_update_frequency": [25, 50, 75],
        "discount_factor": [0.9, 0.95],
        "exploration_decay": [0.99999, 0.999995],
        "min_exploration_rate": [0.2, 0.3],
        "bundling_reward": [0.05],
        "postponement_penalty": [-0.005],
        "on_time_reward": [0.2]
    }

    results = []
    for i in range(num_experiments):
        # Randomly sample hyperparameters
        params = {key: random.choice(values) for key, values in hyperparameter_space.items()}
        print(f"Experiment {i+1}/{num_experiments}: {params}")

        # Define training phases
        phases = [
            {
                "name": "Simple Environment",
                "env_config": {
                    "num_vehicles": 5,
                    "num_restaurants": 5,
                    "service_area_dimensions": (4.0, 4.0),
                    "mean_interarrival_time": 20,
                },
                "performance_criteria": {},
                "min_episodes": 20,
                "max_episodes": episodes_per_experiment
            }
        ]

        # Run training
        try:
            final_model_path = train_rl.train_rl_aca(
                phases=phases,
                save_interval=20,
                stability_window=10,
                stability_threshold=3.0,
                seed=42,
                visualize=False,
                reposition_idle_vehicles=True,
                model_dir=f"data/models/tuning_exp_{i+1}",
                resume_from_model=None,
                start_phase=0,
                start_episode=0,
                exploration_start=0.9,
                exploration_end=params["min_exploration_rate"],
                decay_method="exponential",
                decay_rate=params["exploration_decay"],
                rl_learning_rate=params["learning_rate"],
                rl_batch_size=params["batch_size"],
                rl_target_update_frequency=params["target_update_frequency"],
                rl_discount_factor=params["discount_factor"],
                rl_replay_buffer_capacity=50000,  # Fixed for now
                rl_bundling_reward=params["bundling_reward"],
                rl_postponement_penalty=params["postponement_penalty"],
                rl_on_time_reward=params["on_time_reward"]
            )

            # Evaluate the final model
            eval_stats = train_rl.evaluate_model(
                model_path=final_model_path,
                num_episodes=10,
                seed=100,
                visualize=False,
                env_config=phases[-1]["env_config"]
            )

            # Record the results
            result = {
                "params": params,
                "avg_reward": eval_stats["total_rewards"][-1],
                "avg_delay": eval_stats["total_delays"][-1],
                "avg_on_time_rate": eval_stats["on_time_rates"][-1]
            }
            results.append(result)
            print(f"Experiment {i+1} Results: {result}")

        except Exception as e:
            print(f"Experiment {i+1} failed: {e}")
            continue

    # Find the best combination
    if results:
        best_result = max(results, key=lambda x: x["avg_reward"])
        print(f"\nBest Hyperparameters: {best_result['params']}")
        print(f"Best Average Reward: {best_result['avg_reward']}")
        print(f"Best Average Delay: {best_result['avg_delay']}")
        print(f"Best On-Time Rate: {best_result['avg_on_time_rate']}")
    else:
        print("No successful experiments completed.")

if __name__ == "__main__":
    run_hyperparameter_tuning(num_experiments=20, episodes_per_experiment=500)