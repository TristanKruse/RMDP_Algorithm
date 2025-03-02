from models.aca_policy.aca_policy import ACA
from models.fastest_bundling.fastest_bundler import FastestBundler
from models.fastest_vehicle.fastest_vehicle import FastestVehicleSolver
from environment.environment import RestaurantMealDeliveryEnv
from datatypes import State
from typing import Optional, Dict
from datetime import datetime
import os, json, csv, logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


SOLVERS = {
    "aca": lambda movement_per_step, location_manager: ACA(
        location_manager=location_manager,  # Add this parameter
        # Core algorithm parameters
        buffer=0,
        max_postponements=3,
        max_postpone_time=15,
        # Time & Vehicle parameters
        vehicle_capacity=3,
        service_time=2.0,
        mean_prep_time=10.0,
        delivery_window=40.0,  ### assumed to be the same for all orders, would potentially have to be adjusted.
    ),
    "bundler": lambda s, loc_manager: FastestBundler(
        movement_per_step=s,
        location_manager=loc_manager,
        max_bundle_size=3,
    ),
    "fastest": lambda s, loc_manager: FastestVehicleSolver(movement_per_step=s, location_manager=loc_manager),
}

hourly_pattern = {
    'type': 'hourly',
    'hourly_rates': [
        6, 3, 1, 0.5, 0.5, 0.5,      # 0-5 AM 
        2, 5, 10, 15, 18, 90,        # 6-11 AM
        40, 22, 19, 19, 30, 40,      # 12-5 PM
        62, 62, 40, 25, 18, 10       # 6-11 PM
    ]
}

lunch_dinner_pattern = {
    'type': 'custom_periods',
    'custom_periods': [
        (0.0, 0.2, 0.2),    # Early morning: 20% of base rate
        (0.2, 0.3, 1.0),    # Late morning: 100% of base rate
        (0.3, 0.45, 3.0),   # Lunch peak: 300% of base rate
        (0.45, 0.6, 0.7),   # Afternoon: 70% of base rate
        (0.6, 0.8, 2.5),    # Dinner peak: 250% of base rate
        (0.8, 1.0, 0.5),    # Late night: 50% of base rate
    ]
}

def bimodal_demand(time_percent):
    """Generate a bimodal distribution with peaks at lunch and dinner."""
    import numpy as np
    # Create two normal distributions centered at lunch and dinner times
    lunch_peak = np.exp(-((time_percent - 0.4) ** 2) / 0.005)  # Peak around 40% of the day
    dinner_peak = np.exp(-((time_percent - 0.7) ** 2) / 0.005) # Peak around 70% of the day
    # Combine the distributions and scale
    return 0.2 + 2.8 * (lunch_peak + dinner_peak)  # Base 0.2, peaks at 3.0


function_pattern = {
    'type': 'function',
    'function': bimodal_demand
}

def get_env_config(movement_per_step):
    """Environment configuration with explanatory documentation"""
    return {
        # System size parameters
        "num_restaurants": 110,  # Production: 110 restaurants in system
        "num_vehicles": 15,  # Production: 15 delivery vehicles
        # Time parameters
        "mean_prep_time": 10.0,  # Gamma distributed preparation time (minutes)
        "prep_time_var": 2.0,  # Preparation time variance (COV: 0.0-0.6)
        "delivery_window": 40.0,  # Delivery time window (minutes)
        "simulation_duration": 420,  # 420 # Total simulation time (minutes)
        "cooldown_duration": 60,  # No new orders in final period (minutes)
        # Workload parameters
        "mean_interarrival_time": 2,  # Order frequency:
        # Andersrum??, kleinere interarrival time = mehr Orders ...
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
        "reposition_idle_vehicles": True,  # Whether vehicles reposition when idle
        "seed": None,  # Random seed for reproducibility
        "demand_pattern": None,# e.g., lunch_dinner_pattern,  # Pass your demand pattern here
    }


def prepare_solver_input(state: State) -> dict:
    """Extracts decision-relevant information from full state (following Ulmer et al.).

    Returns dict containing:
    - tk: current time
    - Dk: orders with their properties (tD, RD, VD, LD)
    - Θk: current route plan
    - and objects for nearest neighbour, nodes + vehcile positions
    """
    nodes = {node.id: node for node in state.nodes.values()}

    # Get vehicle assignments from current routes
    vehicle_assignments = {}
    for vehicle_id, route in state.route_plan.items():
        # Access the sequence attribute of Route object
        for node_id, pickups, deliveries in route.sequence:
            # Combine pickups and deliveries to get all orders at this node
            for order_id in pickups | deliveries:
                vehicle_assignments[order_id] = vehicle_id

    # Extract necessary order information (tD, RD, VD, LD)
    orders_info = {}
    for order in state.orders:
        if order.id in state.unassigned_orders:  # Only include unassigned orders
            orders_info[order.id] = {
                "request_time": order.request_time,
                "pickup_node_id": order.pickup_node_id,
                "delivery_node_id": order.delivery_node_id,
            }

    # Add vehicle positions, needed for fastest vehicle sovler
    vehicle_positions = {}
    for vehicle in state.vehicles:  # Iterate directly over list
        vehicle_positions[vehicle.id] = vehicle.current_location

    return {
        "time": state.time,  # tk: point of time
        "unassigned_orders": orders_info,  # Dk: set of orders with their properties
        "route_plan": state.route_plan,  # Θk: current route plan
        "vehicle_positions": vehicle_positions,  # for fastest vehicle, nearest neighbour
        "nodes": nodes,  # Add nodes to the state dictionary
        "orders": state.orders
    }


def run_test_episode(
    solver_name: str = "fastest",
    seed: Optional[int] = None,
    reposition_idle_vehicles: bool = False,
    visualize: bool = False,
    warmup_duration: int = 60,
):
    # is_paused = False
    simulation_duration = simulation_duration = get_env_config(None)["simulation_duration"]  # 420
    speed = 40.0  # km/h as per paper
    street_network_factor = 1.4  # as per paper
    movement_per_step = (speed / 60) / street_network_factor  # km per minute adjusted for street network

    # Initialize environment and solver
    env_params = get_env_config(movement_per_step)
    cooldown_duration = env_params["cooldown_duration"]
    env_params.update(
        {   
            "seed": seed,
            "reposition_idle_vehicles": reposition_idle_vehicles,
            "visualize": visualize,
        }
    )
    env = RestaurantMealDeliveryEnv(**env_params)
    # solver = SOLVERS[solver_name](movement_per_step)
    solver = SOLVERS[solver_name](movement_per_step, env.location_manager)

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
    }

    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    # Pausing functionality
    while not done and step < simulation_duration:
        # Check for pause state
        if env.viz_manager and env.viz_manager.is_paused():
            plt.pause(0.1)  # Keep window responsive while paused
            continue

        route_plan, postponed_orders = solver.solve(prepare_solver_input(state))
        next_state, reward, done, info = env.step((route_plan, postponed_orders))

        # Add idle time tracking
        if step >= warmup_duration and step < (simulation_duration - cooldown_duration):
            episode_stats["active_period_steps"] += 1
            # We'll only track per-vehicle rates during active period
            for vid, rate in info["vehicle_idle_rates"].items():
                if vid not in episode_stats["active_period_idle_rates_by_vehicle"]:
                    episode_stats["active_period_idle_rates_by_vehicle"][vid] = []
                episode_stats["active_period_idle_rates_by_vehicle"][vid].append(rate)

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
    # Before saving, calculate average rates
    episode_stats["active_period_idle_rates_by_vehicle"] = {
        vid: sum(rates) / len(rates) for vid, rates in episode_stats["active_period_idle_rates_by_vehicle"].items()
    }

    # Calculate overall active period idle rate as average of vehicle rates
    if episode_stats["active_period_idle_rates_by_vehicle"]:
        vehicle_rates = list(episode_stats["active_period_idle_rates_by_vehicle"].values())
        episode_stats["active_period_idle_rate"] = sum(vehicle_rates) / len(vehicle_rates)
    else:
        episode_stats["active_period_idle_rate"] = 0.0

    episode_stats = calculate_capacity_metrics(episode_stats, simulation_duration, cooldown_duration, warmup_duration)
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

    logger.info("\nSystem Capacity Metrics:")
    logger.info(f"Orders Per Hour: {episode_stats['orders_per_hour']:.1f}")
    logger.info(f"Active Period Orders Per Hour: {episode_stats['active_period_orders_per_hour']:.1f}")
    logger.info(f"Theoretical Daily Capacity: {episode_stats['system_capacity']:.1f} orders")
    logger.info(f"Active Period Daily Capacity: {episode_stats['active_period_capacity']:.1f} orders")

    logger.info("\nActive Idle Time Metrics:")
    logger.info(f"Active Period Idle Rate: {(episode_stats['active_period_idle_rate'] * 100):.1f}%")
    logger.info("Active Period Idle Rates by Vehicle:")
    for vid, rate in episode_stats["active_period_idle_rates_by_vehicle"].items():
        logger.info(f"  Vehicle {vid}: {(rate * 100):.1f}%")

    logger.info(f"Total Orders Postponed: {len(episode_stats['postponed_orders'])}")
    logger.info(
        f"Postponement Rate: {(len(episode_stats['postponed_orders']) / max(1, episode_stats['total_orders']) * 100):.1f}%"
    )
    logger.info(f"Total accumulated reward: {total_reward:.2f}")

    # Save results
    episode_stats["total_reward"] = total_reward
    save_results(episode_stats, solver_name, seed)

    return episode_stats


def calculate_capacity_metrics(stats, simulation_duration, cooldown_duration, warmup_duration):
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


def save_results(stats: Dict, solver_name: str, seed: Optional[int] = None):
    results_dir = "data/simulation_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate delivery metrics
    delivered_orders = stats["orders_delivered"]
    late_orders = len(stats["late_orders"])
    on_time_rate = ((delivered_orders - late_orders) / delivered_orders * 100) if delivered_orders else 0
    late_rate = (late_orders / delivered_orders * 100) if delivered_orders else 0
    avg_delay = sum(stats["delay_values"]) / len(stats["delay_values"]) if stats["delay_values"] else 0
    avg_distance = (stats["total_distance"] / delivered_orders) if delivered_orders else 0

    save_data = {
        **stats,
        "solver": solver_name,
        "seed": seed,
        "timestamp": timestamp,
        "late_orders": list(stats["late_orders"]),
        "postponed_orders": list(stats["postponed_orders"]),
        "idle_rates_by_vehicle": stats["idle_rates_by_vehicle"],
        "active_period_idle_rates_by_vehicle": stats["active_period_idle_rates_by_vehicle"],
        "active_period_idle_rate": stats["active_period_idle_rate"],
        "active_period_idle_time": stats["active_period_idle_time"],
        "active_period_capacity": stats["active_period_capacity"],
        "active_period_orders_per_hour": stats["active_period_orders_per_hour"],
        "on_time_delivery_rate": on_time_rate,
        "percentage_late_orders": late_rate,
        "avg_delay_late_orders": avg_delay,
        "avg_distance_per_order": avg_distance,
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
        "total_delay": stats["total_delay"],
        "total_reward": stats["total_reward"],
        "late_orders": len(stats["late_orders"]),
        "max_delay": stats["max_delay"],
        "avg_delay": sum(stats["delay_values"]) / len(stats["delay_values"]) if stats["delay_values"] else 0,
        "total_distance": stats["total_distance"],
        "postponed_orders": len(stats["postponed_orders"]),
        "avg_idle_rate": stats["average_idle_rate"],
        "max_vehicle_idle_rate": max(stats["idle_rates_by_vehicle"].values()) if stats["idle_rates_by_vehicle"] else 0,
        "min_vehicle_idle_rate": min(stats["idle_rates_by_vehicle"].values()) if stats["idle_rates_by_vehicle"] else 0,
        "active_period_idle_rate": stats["active_period_idle_rate"],
        "active_period_max_idle": (
            max(stats["active_period_idle_rates_by_vehicle"].values())
            if stats["active_period_idle_rates_by_vehicle"]
            else 0
        ),
        "active_period_min_idle": (
            min(stats["active_period_idle_rates_by_vehicle"].values())
            if stats["active_period_idle_rates_by_vehicle"]
            else 0
        ),
        "orders_per_hour": stats["orders_per_hour"],
        "active_period_orders_per_hour": stats["active_period_orders_per_hour"],
        "system_capacity": stats["system_capacity"],
        "active_period_capacity": stats["active_period_capacity"],
        "on_time_delivery_rate": on_time_rate,
        "percentage_late_orders": late_rate,
        "avg_delay_late_orders": avg_delay,
        "avg_distance_per_order": avg_distance,
        "Note":"To compare results at NORMAL utilization",
        "":"aca with buffer=0, max_postponements=3, max_postpone_time=15",
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
    stats = run_test_episode(
        solver_name="aca",
        seed=1,
        reposition_idle_vehicles=False,
        visualize=False,
        warmup_duration=0,
    )
    logger.info("\nTest completed!")
