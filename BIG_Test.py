# datatpyes.py
# Order rate, num_restaurants, num_vehicles -> driving up computing time
from models.aca_policy.main import RMDPSolver
from models.fastest_bundling.main import FastestBundler
from models.fastest_vehicle.main import FastestVehicleSolver
from environment.main import RestaurantMealDeliveryEnv
from typing import Optional, Dict
import os
import json
import csv
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger instance
logger = logging.getLogger(__name__)


# Code aufräumen, delete, delete, delete
# Claude fragen, wofür brauche ich das, hinterfragen und dann rausschmeißen.

# speed issues, zurück zu schnelle, manchmal teleportation?

# ggf. wird genau in dem Moment eine Order assigned. oder es wird nicht richtig geresettet (bspw. current location/destination)


# hin zu langsam, stimmt irgendwie nicht mit expected speed überein.

# moving to the nearest “unoccupied” restaurant
# when there are no orders to assign to a vehicle


# decision point logik einführen, entweder bundlen oder delivern, oder zum nächsten Restaurant???


# mit dynamischer zUWEISUNG MACHENß


# Bundling.


# Further, sending the
# closest driver may limit the opportunities to assign orders to drivers that are already en route to a
# nearby restaurant.
# -> also muss es doch möglich sein, dass auch Vehicles die schon unterwegs sind eine Order zugeteilt kriegen.


# If the following conditions
# hold, a decision is feasible:
# 1. The arrival time of the first entry of each route
# θ ∈ Θk remains the same.
# 2. The value VD stays unaltered for all orders D
# with VD > 0 (assignments are permanent).
# 3. The sequencing in Θxk
# reflects the driver’s routing
# routine.


# expand bundling to also include how close customers are to each other??? Also als generelle Erweiterung


# implement logging everywhere, instead of print statements

# Bundling funktioniert nicht - Logik überlegen, wie man in route_processor.py
# integrieren kann und wo, auf jeden Fall als separate Methode (route_processor.py, auch jetzt schon zu lang und groß)
# Test Bundling und dann RMDPSolver step by step,


# ggf. umprogrammieren, dass vehicle standby gegeben werden kann, von Algorithmus
# vehicles are skipped that already have an order 54 route_processor.py
# programmieren, dass vehiclees immer wieder richtung mitte gehen, wenn idle.

# wird hier immer richtig inserted?


def get_solver(solver_name: str, movement_per_step: float):
    solvers = {
        "rmdp": lambda: RMDPSolver(
            movement_per_step=movement_per_step,
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
        # Fasteset vehicles + bundling
        "bundle": lambda: FastestBundler(
            movement_per_step=movement_per_step,
            max_bundle_size=3,
            max_restaurant_distance=2.0,
        ),
        # Fastest Vehicle Solver
        "fastest": lambda: FastestVehicleSolver(movement_per_step=movement_per_step),
    }
    return solvers[solver_name]()


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
    env = RestaurantMealDeliveryEnv(
        num_restaurants=1,  # 110,  # Number of restaurants in the system
        num_vehicles=1,  # 15,  # Number of delivery vehicles available
        mean_prep_time=10.0,  # Mean preparation time in minutes (Gamma distributed)
        prep_time_var=2.0,  # Preparation time variance (COV between 0.0-0.6)
        delivery_window=40.0,  # Time window for delivery in minutes
        mean_interarrival_time=60,  # 60/(2.5 orders/hr/vehicle * 15 vehicles) for heavy workload
        # Light workload: 1.5 orders/hr/vehicle (180 total)
        # Normal workload: 2.0 orders/hr/vehicle (240 total)
        # Heavy workload: 2.5 orders/hr/vehicle (300 total)
        simulation_duration=simulation_duration,  # Total simulation duration in minutes
        cooldown_duration=0,  # 60.0,  # the last X minutes no more orders
        seed=seed,  # Random seed for reproducibility
        service_area_dimensions=(10.0, 10.0),
        # street_network_factor=1.4,
        downtown_concentration=0.7,
        # speed=40.0,
        service_time=2.0,
        reposition_idle_vehicles=reposition_idle_vehicles,
        bundling_orders=bundling_orders,
        movement_per_step=movement_per_step,
        # visualization
        visualize=True,
        update_interval=0.01,  # 0.01 or 1
    )

    solver = get_solver(solver_name, movement_per_step)

    # Initialize statistics
    # Initialize statistics with a set for postponed orders
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

        # Block to update postponed orders
        if "postponed_order_ids" in info:
            episode_stats["postponed_orders"].update(info["postponed_order_ids"])
        elif isinstance(postponed_orders, set):
            episode_stats["postponed_orders"].update(postponed_orders)
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
    # Save simulation results to CSV and JSON files.
    # Create results directory if it doesn't exist
    results_dir = "data/simulation_results"
    os.makedirs(results_dir, exist_ok=True)

    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare data for saving
    save_data = {"solver": solver_name, "seed": seed, "timestamp": timestamp, **stats}

    # Convert sets to lists for JSON serialization
    save_data["late_orders"] = list(save_data["late_orders"])
    save_data["postponed_orders"] = list(save_data["postponed_orders"])

    # Save detailed results as JSON
    json_path = os.path.join(results_dir, f"simulation_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=4)

    # Save summary results as CSV
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
