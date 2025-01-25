from models.aca_policy.main import ACA
from models.fastest_bundling.main import FastestBundler
from models.fastest_vehicle.main import FastestVehicleSolver
from environment.main import RestaurantMealDeliveryEnv
from typing import Optional, Dict
from datetime import datetime
import os, json, csv, logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


SOLVERS = {
    "aca": lambda s: ACA(
        movement_per_step=s,
        # Core algorithm parameters
        buffer=5.0,
        max_postponements=3,
        max_postpone_time=15.0,
        # Time & Vehicle parameters
        vehicle_capacity=3,
        service_time=2.0,
        mean_prep_time=10.0,
        prep_time_var=2.0,
        delay_normalization_factor=10.0,  # sensitivity for delays
    ),
    "bundle": lambda s: FastestBundler(
        movement_per_step=s,
        max_bundle_size=3,
        max_restaurant_distance=2.0,
    ),
    "fastest": lambda s: FastestVehicleSolver(movement_per_step=s),
}


def get_env_config(movement_per_step):
    """Environment configuration with explanatory documentation"""
    return {
        # System size parameters
        "num_restaurants": 1,  # Production: 110 restaurants in system
        "num_vehicles": 1,  # Production: 15 delivery vehicles
        # Time parameters
        "mean_prep_time": 10.0,  # Gamma distributed preparation time (minutes)
        "prep_time_var": 2.0,  # Preparation time variance (COV: 0.0-0.6)
        "delivery_window": 40.0,  # Delivery time window (minutes)
        "simulation_duration": 80,  # Total simulation time (minutes)
        "cooldown_duration": 0,  # No new orders in final period (minutes)
        # Workload parameters
        "mean_interarrival_time": 60,  # Order frequency:
        # Light: 1.5 orders/hr/vehicle (180 total)
        # Normal: 2.0 orders/hr/vehicle (240 total)
        # Heavy: 2.5 orders/hr/vehicle (300 total)
        # Here: 60/(2.5 orders/hr/vehicle * 15 vehicles)
        # Area parameters
        "service_area_dimensions": (10.0, 10.0),  # 10km x 10km area
        "downtown_concentration": 0.7,  # Restaurant concentration downtown
        # Service parameters
        "service_time": 2.0,  # Time at pickup/delivery locations
        "movement_per_step": movement_per_step,
        # Visualization
        "visualize": True,
        "update_interval": 0.01,  # Update frequency (0.01 or 1)
        # Optional behavior flags (set by run_test_episode)
        "reposition_idle_vehicles": False,  # Whether vehicles reposition when idle
        "bundling_orders": False,  # Whether to allow order bundling
        "seed": None,  # Random seed for reproducibility
    }


def run_test_episode(
    solver_name: str = "fastest",
    seed: Optional[int] = None,
    reposition_idle_vehicles: bool = False,
    bundling_orders: bool = False,
):
    simulation_duration = 80  # 420
    speed = 40.0  # km/h as per paper
    street_network_factor = 1.4  # as per paper
    movement_per_step = (speed / 60) / street_network_factor  # km per minute adjusted for street network

    # Initialize environment and solver
    env_params = get_env_config(movement_per_step)
    env_params.update(
        {"seed": seed, "reposition_idle_vehicles": reposition_idle_vehicles, "bundling_orders": bundling_orders}
    )
    env = RestaurantMealDeliveryEnv(**env_params)
    solver = SOLVERS[solver_name](movement_per_step)

    # Initialize statistics
    episode_stats = {
        "total_orders": 0,
        "orders_delivered": 0,
        "total_delay": 0,
        "late_orders": set(),  # Use set to avoid duplicates
        "max_delay": 0,
        "delay_values": [],
        "total_distance": 0,
        "postponed_orders": set(),  # Change to set instead of counter
    }

    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done and step < simulation_duration:
        route_plan, postponed_orders = solver.solve(state)
        next_state, reward, done, info = env.step((route_plan, postponed_orders))

        # Update statistics
        total_reward += reward
        episode_stats["total_distance"] += info["distance"]
        episode_stats["orders_delivered"] += info["deliveries"]
        episode_stats["total_orders"] = info["total_orders"]

        if info["delays"]:
            episode_stats["delay_values"].extend(info["delays"])
            episode_stats["late_orders"].update(info["late_orders"])
            episode_stats["max_delay"] = max(episode_stats["max_delay"], max(info["delays"]))

        episode_stats["postponed_orders"].update(info.get("postponed_order_ids", postponed_orders))

        state = next_state
        step += 1

    # Calculate final metrics
    total_orders = max(1, episode_stats["total_orders"])
    delivered_orders = episode_stats["orders_delivered"]
    late_orders = len(episode_stats["late_orders"])

    logger.info("\nFinal Performance Metrics:")
    logger.info(f"Total simulation steps: {step}")
    logger.info(f"Total Orders: {total_orders}")
    logger.info(f"Orders Delivered: {delivered_orders}")
    logger.info(
        f"On-Time Delivery Rate: {((delivered_orders - late_orders) / delivered_orders * 100):.1f}%"
        if delivered_orders
        else "0%"
    )

    logger.info("\nDelay Metrics:")
    logger.info(f"Number of Late Orders: {late_orders}")
    logger.info(
        f"Percentage of Late Orders: {(late_orders / delivered_orders * 100):.1f}%" if delivered_orders else "0%"
    )

    if episode_stats["delay_values"]:
        avg_delay = sum(episode_stats["delay_values"]) / len(episode_stats["delay_values"])
        logger.info(f"Average Delay of Late Orders: {avg_delay:.1f} minutes")
        logger.info(f"Maximum Delay: {episode_stats['max_delay']:.1f} minutes")

    logger.info("\nOperational Metrics:")
    logger.info(f"Total Distance Traveled: {episode_stats['total_distance']:.1f}")
    logger.info(
        f"Average Distance per Order: {(episode_stats['total_distance'] / delivered_orders):.1f}"
        if delivered_orders
        else "0.0"
    )
    # Print final metrics
    logger.info(f"Total Orders Postponed: {len(episode_stats['postponed_orders'])}")
    logger.info(
        f"Postponement Rate: {(len(episode_stats['postponed_orders']) / max(1, episode_stats['total_orders']) * 100):.1f}%"
    )
    logger.info(f"Total accumulated reward: {total_reward:.2f}")

    # Save results
    episode_stats["total_reward"] = total_reward
    save_results(episode_stats, solver_name, seed)

    return episode_stats


def save_results(stats: Dict, solver_name: str, seed: Optional[int] = None):
    results_dir = "data/simulation_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_data = {
        **stats,
        "solver": solver_name,
        "seed": seed,
        "timestamp": timestamp,
        "late_orders": list(stats["late_orders"]),
        "postponed_orders": list(stats["postponed_orders"]),
    }

    json_path = os.path.join(results_dir, f"simulation_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=4)

    csv_path = os.path.join(results_dir, "simulation_summary.csv")
    csv_exists = os.path.exists(csv_path)

    summary_data = {
        "timestamp": timestamp,
        "solver": solver_name,
        "seed": seed,
        "total_orders": stats["total_orders"],
        "orders_delivered": stats["orders_delivered"],
        "late_orders": len(stats["late_orders"]),
        "max_delay": stats["max_delay"],
        "avg_delay": sum(stats["delay_values"]) / len(stats["delay_values"]) if stats["delay_values"] else 0,
        "total_distance": stats["total_distance"],
        "postponed_orders": len(stats["postponed_orders"]),
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())
        if not csv_exists:
            writer.writeheader()
        writer.writerow(summary_data)

    logger.info(f"\nResults saved to:")
    logger.info(f"Detailed results: {json_path}")
    logger.info(f"Summary results: {csv_path}")


if __name__ == "__main__":
    logger.info("Starting test episode...")
    stats = run_test_episode(solver_name="fastest", seed=1, reposition_idle_vehicles=False, bundling_orders=False)
    logger.info("\nTest completed!")
