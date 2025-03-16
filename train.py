from environment.meituan_data.meituan_data_config import MeituanDataConfig
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

def get_solver_params(solver):
    """Extract parameters from the solver object in a more comprehensive way."""
    params = {}
    
    # Common attributes to check across solvers
    attribute_list = [
        'buffer', 'max_postponements', 'max_postpone_time', 
        'vehicle_capacity', 'service_time', 'mean_prep_time', 
        'delivery_window', 'max_bundle_size', 'postponement_method',
        'rl_training_mode', 'rl_state_size'  # Add RL attributes
    ]
    
    # Extract directly accessible parameters
    for attr in attribute_list:
        if hasattr(solver, attr):
            params[attr] = getattr(solver, attr)
    
    # RL specific parameters
    if hasattr(solver, 'postponement_method') and solver.postponement_method == "rl":
        # Add RL-specific metrics
        if hasattr(solver.postponement, 'total_training_steps'):
            params['rl_training_steps'] = solver.postponement.total_training_steps
        if hasattr(solver.postponement, 'exploration_rate'):
            params['rl_exploration_rate'] = solver.postponement.exploration_rate
        if hasattr(solver.postponement, 'batch_losses') and solver.postponement.batch_losses:
            params['rl_avg_loss'] = sum(solver.postponement.batch_losses[-100:]) / min(100, len(solver.postponement.batch_losses))
    
    # For other components that might have parameters
    for component_name in ['vehicle_ops', 'postponement', 'route_utils']:
        if hasattr(solver, component_name):
            component = getattr(solver, component_name)
            for attr in attribute_list:
                if hasattr(component, attr) and attr not in params:
                    params[attr] = getattr(component, attr)
    
    return params

def run_test_episode(
    solver_name: str = "fastest",
    meituan_config=None,
    seed: Optional[int] = None,
    reposition_idle_vehicles: bool = False,
    visualize: bool = False,
    warmup_duration: int = 60,
    save_rl_model: bool = False,
    rl_model_path: str = None,
    save_results_to_disk=True,
    env_config: Optional[Dict] = None
):
    # is_paused = False
    simulation_duration = simulation_duration = get_env_config(None)["simulation_duration"]  # 420
    speed = 20   # 40.0 km/h in paper, 10kmh Durchschnitt in Meituan Daten
    street_network_factor = 1.0  # 1.4 in paper, we calculated the average speed over the euclidic distance, so no adjustment necessary
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
    
    # Apply custom environment config if provided
    if env_config:
        env_params.update(env_config)

    # Apply Meituan data configuration to environment parameters
    if meituan_config is not None:
        env_params = meituan_config.apply_to_env_params(env_params)
    
    env = RestaurantMealDeliveryEnv(**env_params)

    # Apply Meituan data configuration to environment
    if meituan_config is not None:
        meituan_config.apply_to_environment(env)

    # Reset the environment to properly initialize
    state = env.reset()

    logging.info(f"Starting simulation with solver: {solver_name}")
    # solver = SOLVERS[solver_name](movement_per_step)
    solver = SOLVERS[solver_name](movement_per_step, env.location_manager)

    # Load RL model if path provided
    if rl_model_path and hasattr(solver, 'postponement_method') and solver.postponement_method == "rl":
        solver.postponement.load_model(rl_model_path)

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
        "orders_per_restaurant": {},  # Track orders by restaurant ID
        # New meal prep time metrics
        "true_prep_times": [],  # Actual meal preparation times (ready_time - request_time)
        "avg_true_prep_time": 0.0,  # Average true preparation time
        "max_true_prep_time": 0.0,  # Maximum true preparation time
        "order_wait_times": [],  # Time orders waited after being ready (pickup_time - ready_time)
        "avg_order_wait_time": 0.0,  # Average time orders waited after being ready
        "max_order_wait_time": 0.0,  # Maximum time orders waited after being ready
        "total_pickup_times": [],  # Total time from order to pickup (pickup_time - request_time)
        "avg_total_pickup_time": 0.0,  # Average total time to pickup
        "max_total_pickup_time": 0.0,  # Maximum total time to pickup
        "total_driver_wait_time": 0.0,  # Total time drivers waited for food
        "driver_wait_orders": 0,  # Number of orders where driver had to wait
        "true_prep_by_restaurant": {},  # Track true prep times by restaurant {restaurant_id: [true_prep_times]}
        "driver_wait_by_restaurant": {},  # Keep this as is
        "order_wait_by_restaurant": {},  # Track order wait times by restaurant {restaurant_id: [order_wait_times]}
        "total_pickup_by_restaurant": {}  # Track total pickup times by restaurant {restaurant_id: [total_pickup_times]}
    }

    done = False
    total_reward = 0
    step = 0
    order_restaurant_map = {}  # Map of order IDs to restaurant IDs

    # Pausing functionality
    while not done and step < simulation_duration:
        # Check for pause state
        if env.viz_manager and env.viz_manager.is_paused():
            plt.pause(0.1)  # Keep window responsive while paused
            continue

        route_plan, postponed_orders = solver.solve(prepare_solver_input(state))
        next_state, reward, done, info = env.step((route_plan, postponed_orders))

        # Add restaurant tracking for all new orders in state
        for order in next_state.orders:
            if order.id not in order_restaurant_map and hasattr(order, 'pickup_node_id'):
                order_restaurant_map[order.id] = order.pickup_node_id.id

        # Orders per restaurant
        if info["deliveries"] > 0:           
            for order_id in info.get("delivered_orders", set()):
                if order_id in order_restaurant_map:
                    restaurant_id = order_restaurant_map[order_id]
                    if restaurant_id not in episode_stats["orders_per_restaurant"]:
                        episode_stats["orders_per_restaurant"][restaurant_id] = 0
                    episode_stats["orders_per_restaurant"][restaurant_id] += 1
                    # Also track prep time statistics for delivered orders
            # ----- KPI Tracking -----
            # Find order in previously tracked orders
            delivered_order = next((o for o in state.orders if o.id == order_id), None)
            if not delivered_order:
                delivered_order = next((o for o in next_state.orders if o.id == order_id), None)
                        
            if delivered_order:
                restaurant_id = order_restaurant_map.get(order_id)
                # Track true meal prep time (if available)
                if hasattr(delivered_order, 'true_prep_time'):
                    true_prep = delivered_order.true_prep_time
                    episode_stats["true_prep_times"].append(true_prep)
                    episode_stats["max_true_prep_time"] = max(episode_stats["max_true_prep_time"], true_prep)
                    # Track by restaurant
                    if restaurant_id:
                        if restaurant_id not in episode_stats["true_prep_by_restaurant"]:
                            episode_stats["true_prep_by_restaurant"][restaurant_id] = []
                        episode_stats["true_prep_by_restaurant"][restaurant_id].append(true_prep)
                # Track order wait time after being ready (if available)
                if hasattr(delivered_order, 'order_wait_time'):
                    order_wait = delivered_order.order_wait_time
                    episode_stats["order_wait_times"].append(order_wait)
                    episode_stats["max_order_wait_time"] = max(episode_stats["max_order_wait_time"], order_wait)
                    # Track by restaurant
                    if restaurant_id:
                        if restaurant_id not in episode_stats["order_wait_by_restaurant"]:
                            episode_stats["order_wait_by_restaurant"][restaurant_id] = []
                        episode_stats["order_wait_by_restaurant"][restaurant_id].append(order_wait)
                # Track total pickup time (if available)
                if hasattr(delivered_order, 'total_time_to_pickup'):
                    total_pickup = delivered_order.total_time_to_pickup
                    episode_stats["total_pickup_times"].append(total_pickup)
                    episode_stats["max_total_pickup_time"] = max(episode_stats["max_total_pickup_time"], total_pickup)                
                    # Track by restaurant
                    if restaurant_id:
                        if restaurant_id not in episode_stats["total_pickup_by_restaurant"]:
                            episode_stats["total_pickup_by_restaurant"][restaurant_id] = []
                        episode_stats["total_pickup_by_restaurant"][restaurant_id].append(total_pickup)            
                # Track driver wait time (keep this part as is)
                if hasattr(delivered_order, 'driver_wait_time') and delivered_order.driver_wait_time > 0:
                    wait_time = delivered_order.driver_wait_time
                    episode_stats["total_driver_wait_time"] += wait_time
                    episode_stats["driver_wait_orders"] += 1                
                    # Track by restaurant
                    if restaurant_id:
                        if restaurant_id not in episode_stats["driver_wait_by_restaurant"]:
                            episode_stats["driver_wait_by_restaurant"][restaurant_id] = []
                        episode_stats["driver_wait_by_restaurant"][restaurant_id].append(wait_time)

                # ----- KPI Tracking -----


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
    # Save results
    episode_stats["total_reward"] = total_reward

    if save_results_to_disk:

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

        # ----- KPI Tracking -----
        # Calculate averages for all time metrics
        if episode_stats["true_prep_times"]:
            episode_stats["avg_true_prep_time"] = sum(episode_stats["true_prep_times"]) / len(episode_stats["true_prep_times"])

        if episode_stats["order_wait_times"]:
            episode_stats["avg_order_wait_time"] = sum(episode_stats["order_wait_times"]) / len(episode_stats["order_wait_times"])

        if episode_stats["total_pickup_times"]:
            episode_stats["avg_total_pickup_time"] = sum(episode_stats["total_pickup_times"]) / len(episode_stats["total_pickup_times"])

        # Calculate average times by restaurant
        episode_stats["avg_true_prep_by_restaurant"] = {
            r_id: sum(times) / len(times) 
            for r_id, times in episode_stats["true_prep_by_restaurant"].items()
        }

        episode_stats["avg_order_wait_by_restaurant"] = {
            r_id: sum(times) / len(times) 
            for r_id, times in episode_stats["order_wait_by_restaurant"].items()
        }

        episode_stats["avg_total_pickup_by_restaurant"] = {
            r_id: sum(times) / len(times) 
            for r_id, times in episode_stats["total_pickup_by_restaurant"].items()
        }

        episode_stats["avg_driver_wait_by_restaurant"] = {
            r_id: sum(times) / len(times) 
            for r_id, times in episode_stats["driver_wait_by_restaurant"].items()
        }

        # Calculate percentage of orders where drivers had to wait
        if delivered_orders > 0:
            episode_stats["driver_wait_percentage"] = (episode_stats["driver_wait_orders"] / delivered_orders) * 100
        else:
            episode_stats["driver_wait_percentage"] = 0.0
        
        # ----- KPI Tracking -----
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



        # Orders per Restaurants
        logger.info("\nRestaurant Distribution Metrics:")
        if episode_stats["orders_per_restaurant"]:
            # Sort restaurants by order count (descending)
            sorted_restaurants = sorted(
                episode_stats["orders_per_restaurant"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            logger.info(f"Total restaurants served: {len(episode_stats['orders_per_restaurant'])}")
            logger.info(f"Restaurant with most orders: ID {sorted_restaurants[0][0]} with {sorted_restaurants[0][1]} orders")
            
            # Calculate distribution statistics
            order_counts = list(episode_stats["orders_per_restaurant"].values())
            avg_orders = sum(order_counts) / len(order_counts)
            logger.info(f"Average orders per restaurant: {avg_orders:.1f}")
            logger.info(f"Order count distribution: Min: {min(order_counts)}, Max: {max(order_counts)}")
            
            # Output top 5 restaurants
            logger.info("Top 5 restaurants by order count:")
            for i, (r_id, count) in enumerate(sorted_restaurants[:5], 1):
                logger.info(f"  #{i}: Restaurant {r_id} - {count} orders ({count/delivered_orders*100:.1f}% of all deliveries)")

        # ----- KPI Tracking -----
        # Add to your existing logging
        logger.info("\nTime Metrics:")

        # True meal preparation time
        if episode_stats["true_prep_times"]:
            logger.info("\nTrue Meal Preparation Time (cooking time):")
            logger.info(f"  Average: {episode_stats['avg_true_prep_time']:.1f} minutes")
            logger.info(f"  Maximum: {episode_stats['max_true_prep_time']:.1f} minutes")
            
            # Log top 5 restaurants by true prep time
            if episode_stats["avg_true_prep_by_restaurant"]:
                logger.info("\n  Top 5 Restaurants by True Preparation Time:")
                top_prep_restaurants = sorted(
                    episode_stats["avg_true_prep_by_restaurant"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                
                for i, (r_id, avg_time) in enumerate(top_prep_restaurants, 1):
                    order_count = len(episode_stats["true_prep_by_restaurant"][r_id])
                    logger.info(f"    #{i}: Restaurant {r_id} - {avg_time:.1f} minutes average ({order_count} orders)")

        # Order wait time after being ready
        if episode_stats["order_wait_times"]:
            logger.info("\nOrder Wait Time (after food was ready):")
            logger.info(f"  Average: {episode_stats['avg_order_wait_time']:.1f} minutes")
            logger.info(f"  Maximum: {episode_stats['max_order_wait_time']:.1f} minutes")
            
            # Log top 5 restaurants by order wait time
            if episode_stats["avg_order_wait_by_restaurant"]:
                logger.info("\n  Top 5 Restaurants by Order Wait Time:")
                top_wait_restaurants = sorted(
                    episode_stats["avg_order_wait_by_restaurant"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                
                for i, (r_id, avg_time) in enumerate(top_wait_restaurants, 1):
                    order_count = len(episode_stats["order_wait_by_restaurant"][r_id])
                    logger.info(f"    #{i}: Restaurant {r_id} - {avg_time:.1f} minutes average ({order_count} orders)")

        # Total pickup time
        if episode_stats["total_pickup_times"]:
            logger.info("\nTotal Time to Pickup (from order creation to pickup):")
            logger.info(f"  Average: {episode_stats['avg_total_pickup_time']:.1f} minutes")
            logger.info(f"  Maximum: {episode_stats['max_total_pickup_time']:.1f} minutes")

        # Driver wait time
        logger.info("\nDriver Wait Time (drivers waiting for food):")
        logger.info(f"  Total Driver Wait Time: {episode_stats['total_driver_wait_time']:.1f} minutes")
        logger.info(f"  Drivers Waited for {episode_stats['driver_wait_orders']} orders ({episode_stats['driver_wait_percentage']:.1f}%)")
        # ----- KPI Tracking -----
        # Extract solver parameters
        solver_params = get_solver_params(solver)
        # Save results to disk
        save_results(episode_stats, solver_name, seed, meituan_config, solver_params, env_params)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save RL model if requested
    if save_rl_model and hasattr(solver, 'save_rl_model'):
        model_save_path = f"data/models/rl_aca_{timestamp}.pt" if rl_model_path is None else rl_model_path
        solver.save_rl_model(model_save_path)
        logger.info(f"RL model saved to {model_save_path}")

    return episode_stats

# Helper functions to calculate and save results
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

def save_results(stats: Dict, solver_name: str, seed: Optional[int] = None, meituan_config=None, solver_params=None, env_params=None):
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

    # Restaurant metrics
    total_restaurants_served = len(stats["orders_per_restaurant"])
    restaurant_order_counts = list(stats["orders_per_restaurant"].values()) if stats["orders_per_restaurant"] else []
    avg_orders_per_restaurant = sum(restaurant_order_counts) / len(restaurant_order_counts) if restaurant_order_counts else 0
    min_restaurant_orders = min(restaurant_order_counts) if restaurant_order_counts else 0
    max_restaurant_orders = max(restaurant_order_counts) if restaurant_order_counts else 0
    
    # Top restaurants
    top_restaurants = []
    if stats["orders_per_restaurant"]:
        sorted_restaurants = sorted(
            stats["orders_per_restaurant"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_restaurants = sorted_restaurants[:5]
        top_restaurant_id = sorted_restaurants[0][0] if sorted_restaurants else None
        top_restaurant_orders = sorted_restaurants[0][1] if sorted_restaurants else 0
    else:
        top_restaurant_id = None
        top_restaurant_orders = 0
    
    # Time metrics
    avg_true_prep_time = stats["avg_true_prep_time"] if stats["true_prep_times"] else 0
    max_true_prep_time = stats["max_true_prep_time"]
    avg_order_wait_time = stats["avg_order_wait_time"] if stats["order_wait_times"] else 0
    max_order_wait_time = stats["max_order_wait_time"]
    avg_total_pickup_time = stats["avg_total_pickup_time"] if stats["total_pickup_times"] else 0
    max_total_pickup_time = stats["max_total_pickup_time"]
    total_driver_wait_time = stats["total_driver_wait_time"]
    driver_wait_orders = stats["driver_wait_orders"]
    driver_wait_percentage = stats["driver_wait_percentage"] if delivered_orders > 0 else 0

    # Top restaurants by prep time
    top_prep_restaurants = []
    if stats["avg_true_prep_by_restaurant"]:
        top_prep_restaurants = sorted(
            stats["avg_true_prep_by_restaurant"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
    
    # Top restaurants by order wait time
    top_wait_restaurants = []
    if stats["avg_order_wait_by_restaurant"]:
        top_wait_restaurants = sorted(
            stats["avg_order_wait_by_restaurant"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
    
    # Extract configuration parameters if available
    config_info = {}
    if meituan_config is not None:
        config_info = {
            "district_id": getattr(meituan_config, "district_id", None),
            "day": getattr(meituan_config, "day", None),
            "use_restaurant_positions": getattr(meituan_config, "use_restaurant_positions", None),
            "use_vehicle_count": getattr(meituan_config, "use_vehicle_count", None),
            "use_vehicle_positions": getattr(meituan_config, "use_vehicle_positions", None),
            "use_service_area": getattr(meituan_config, "use_service_area", None),
            "use_deadlines": getattr(meituan_config, "use_deadlines", None),
            "order_generation_mode": getattr(meituan_config, "order_generation_mode", None),
            "temporal_pattern_type": (
                getattr(meituan_config, "temporal_pattern", {}).get("type", None) 
                if hasattr(meituan_config, "temporal_pattern") and meituan_config.temporal_pattern is not None 
                else None
            ),
            "simulation_start_hour": getattr(meituan_config, "simulation_start_hour", None),
            "simulation_duration_hours": getattr(meituan_config, "simulation_duration_hours", None),
        }

    # Add all these metrics to the save_data for JSON
    save_data = {
        **stats,
        "solver": solver_name,
        "seed": seed,
        "timestamp": timestamp,
        "config": config_info,  # Add the config information
        "solver_params": solver_params or {},
        "env_params": env_params or {},  # Add environment parameters
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
        "orders_per_restaurant": stats["orders_per_restaurant"],
        
        # New metrics
        "total_restaurants_served": total_restaurants_served,
        "avg_orders_per_restaurant": avg_orders_per_restaurant,
        "min_restaurant_orders": min_restaurant_orders,
        "max_restaurant_orders": max_restaurant_orders,
        "top_restaurant_id": top_restaurant_id,
        "top_restaurant_orders": top_restaurant_orders,
        "top_restaurants": top_restaurants,
        
        # Time metrics
        "avg_true_prep_time": avg_true_prep_time,
        "max_true_prep_time": max_true_prep_time,
        "avg_order_wait_time": avg_order_wait_time,
        "max_order_wait_time": max_order_wait_time,
        "avg_total_pickup_time": avg_total_pickup_time,
        "max_total_pickup_time": max_total_pickup_time,
        "total_driver_wait_time": total_driver_wait_time,
        "driver_wait_orders": driver_wait_orders,
        "driver_wait_percentage": driver_wait_percentage,
        
        # Top restaurants by metrics
        "top_prep_restaurants": top_prep_restaurants,
        "top_wait_restaurants": top_wait_restaurants,
    }

    json_path = os.path.join(results_dir, f"simulation_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=4)

    csv_path = os.path.join(results_dir, "simulation_summary.csv")
    csv_exists = os.path.exists(csv_path)


    # Add key solver parameters to CSV summary
    solver_param_entries = {}
    if solver_params:
        # Add solver parameters with prefix to avoid column name collisions
        for param_name, param_value in solver_params.items():
            solver_param_entries[f"solver_{param_name}"] = param_value

    # Create a fixed set of fieldnames to ensure consistent column order
    # This should include ALL possible fields from ALL solvers
    fixed_fieldnames = [
        "timestamp", "solver", "seed",
        # Config parameters
        "district_id", "day", "order_generation_mode", "simulation_hours", "simulation_start_hour",
        "use_restaurant_positions", "use_vehicle_count", "use_vehicle_positions", 
        "use_service_area", "use_deadlines",

        # Environment parameters
        "env_num_restaurants", "env_num_vehicles", "env_mean_prep_time", "env_prep_time_var",
        "env_delivery_window", "env_simulation_duration", "env_cooldown_duration",
        "env_mean_interarrival_time", "env_service_area_dimensions", "env_downtown_concentration",
        "env_service_time", "env_movement_per_step", "env_reposition_idle_vehicles",
        "env_seed", "env_update_interval", "env_demand_pattern", "env_visualize",

        
        # Solver parameters - include all possible solver parameters here
        "solver_buffer", "solver_max_postponements", "solver_max_postpone_time",
        "solver_vehicle_capacity", "solver_service_time", "solver_mean_prep_time",
        "solver_delivery_window", "solver_max_bundle_size",
        # Add any other potential solver parameters
        
        # Result metrics
        "total_orders", "orders_delivered", "total_delay", "total_reward",
        "late_orders", "max_delay", "avg_delay", "total_distance",
        "postponed_orders", "active_period_idle_rate", "active_period_max_idle",
        "active_period_min_idle", "orders_per_hour", "active_period_orders_per_hour",
        "system_capacity", "active_period_capacity", "on_time_delivery_rate",
        "percentage_late_orders", "avg_delay_late_orders", "avg_distance_per_order",
        
        # Restaurant metrics
        "total_restaurants_served", "avg_orders_per_restaurant",
        "min_restaurant_orders", "max_restaurant_orders", "top_restaurant_id",
        
        # Time metrics
        "avg_true_prep_time", "max_true_prep_time", "avg_order_wait_time",
        "max_order_wait_time", "avg_total_pickup_time", "max_total_pickup_time",
        "total_driver_wait_time", "driver_wait_orders", "driver_wait_percentage",
        
        "Note", ""
    ]

    # Initialize summary_data with empty strings for all fields
    summary_data = {field: "" for field in fixed_fieldnames}
    # Add environment parameters to summary_data
    env_param_entries = {}
    if env_params:
        for param_name, param_value in env_params.items():
            # Convert tuple parameters to string for CSV
            if isinstance(param_value, tuple):
                param_value = str(param_value)
            env_param_entries[f"env_{param_name}"] = param_value

    summary_data = {
        "timestamp": timestamp,
        "solver": solver_name,
        "seed": seed,
        # Config Paremeters
        "district_id": config_info.get("district_id", ""),
        "day": config_info.get("day", ""),
        "order_generation_mode": config_info.get("order_generation_mode", ""),
        "simulation_hours": config_info.get("simulation_duration_hours", ""),
        "simulation_start_hour": config_info.get("simulation_start_hour", ""),  # Add this line
        "use_restaurant_positions": config_info.get("use_restaurant_positions", ""),  # And any others you want
        "use_vehicle_count": config_info.get("use_vehicle_count", ""),
        "use_vehicle_positions": config_info.get("use_vehicle_positions", ""),
        "use_service_area": config_info.get("use_service_area", ""),
        "use_deadlines": config_info.get("use_deadlines", ""),

        # Solver parameters
        **solver_param_entries,

        "total_orders": stats["total_orders"],
        "orders_delivered": stats["orders_delivered"],
        "total_delay": stats["total_delay"],
        "total_reward": stats["total_reward"],
        "late_orders": len(stats["late_orders"]),
        "max_delay": stats["max_delay"],
        "avg_delay": sum(stats["delay_values"]) / len(stats["delay_values"]) if stats["delay_values"] else 0,
        "total_distance": stats["total_distance"],
        "postponed_orders": len(stats["postponed_orders"]),
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
        
        # New restaurant metrics
        "total_restaurants_served": total_restaurants_served,
        "avg_orders_per_restaurant": avg_orders_per_restaurant,
        "min_restaurant_orders": min_restaurant_orders,
        "max_restaurant_orders": max_restaurant_orders,
        "top_restaurant_id": top_restaurant_id,
        
        # New time metrics
        "avg_true_prep_time": avg_true_prep_time,
        "max_true_prep_time": max_true_prep_time,
        "avg_order_wait_time": avg_order_wait_time,
        "max_order_wait_time": max_order_wait_time,
        "avg_total_pickup_time": avg_total_pickup_time,
        "max_total_pickup_time": max_total_pickup_time,
        "total_driver_wait_time": total_driver_wait_time,
        "driver_wait_orders": driver_wait_orders,
        "driver_wait_percentage": driver_wait_percentage,
        
        "Note": "",
        "": "",
    }
    # Update summary_data with environment parameters
    summary_data.update(env_param_entries)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fixed_fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(summary_data)

    logger.info(f"\nResults saved to:")
    logger.info(f"Detailed results: {json_path}")
    logger.info(f"Summary results: {csv_path}")
    # Call this in save_results function:
    visualize_restaurant_distribution(stats, solver_name, timestamp)

def visualize_restaurant_distribution(stats, solver_name, timestamp):
    """Create a visualization of orders per restaurant"""
    print(f"Attempting to visualize restaurant distribution. Data available: {bool(stats['orders_per_restaurant'])}")
    if not stats["orders_per_restaurant"]:
        return  # No data to visualize
        
    # Create output directory
    viz_dir = os.path.join("data/simulation_results/visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get restaurant data
    restaurant_data = list(stats["orders_per_restaurant"].items())
    restaurant_data.sort(key=lambda x: x[1], reverse=True)
    
    # Limit to top 30 restaurants for readability
    if len(restaurant_data) > 30:
        restaurant_data = restaurant_data[:30]
    
    restaurant_ids = [str(r[0]) for r in restaurant_data]
    order_counts = [r[1] for r in restaurant_data]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(restaurant_ids, order_counts)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.0f}',
                ha='center', va='bottom')
    
    plt.title(f'Orders Per Restaurant - {solver_name}')
    plt.xlabel('Restaurant ID')
    plt.ylabel('Number of Orders')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(viz_dir, f"restaurant_distribution_{timestamp}.png"), dpi=300)
    plt.close()


def get_env_config(movement_per_step):
    """Environment configuration with explanatory documentation"""
    return {
        # System size parameters
        "num_restaurants": 15,  # Production: 110 restaurants in system
        "num_vehicles": 5,  # Production: 15 delivery vehicles
        # Time parameters
        "mean_prep_time": 13.0,  # Gamma distributed preparation time (minutes) -> maybe should be Standard dist?
        "prep_time_var": 2.0,  # Preparation time variance (COV: 0.0-0.6)
        "delivery_window": 40,  # Delivery time window (minutes)
        "simulation_duration": 420,  # 420 # Total simulation time (minutes)
        "cooldown_duration": 0,  # No new orders in final period (minutes)
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

SOLVERS = {
    "aca": lambda movement_per_step, location_manager: ACA(
        location_manager=location_manager,  # Add this parameter
        # Core algorithm parameters
        buffer=40,
        max_postponements=0,
        max_postpone_time=10,
        # Time & Vehicle parameters
        vehicle_capacity=3, # test 5 
        service_time=2.0,
        mean_prep_time=13,
        delivery_window=40.0,  ### assumed to be the same for all orders, would potentially have to be adjusted.
        # Default to heuristic postponement
        postponement_method="heuristic",
    ),
    # Add RL-based ACA
    "rl_aca": lambda movement_per_step, location_manager: ACA(
        location_manager=location_manager,
        # Core algorithm parameters
        buffer=40,
        max_postponements=3,
        max_postpone_time=10,
        # Vehicle parameters
        vehicle_capacity=3,
        # Time parameters
        service_time=2.0,
        mean_prep_time=13,
        delivery_window=40.0,
        # Use RL-based postponement
        postponement_method="rl", # rl
        rl_training_mode=True,
        rl_state_size=10,
    ),
    "bundler": lambda s, loc_manager: FastestBundler(
        movement_per_step=s,
        location_manager=loc_manager,
        max_bundle_size=3,
    ),
    "fastest": lambda s, loc_manager: FastestVehicleSolver(movement_per_step=s, location_manager=loc_manager),
}

custom_config = MeituanDataConfig(
    district_id=10,                         # Districts 1 to 22
    day="20221018",                         # Specify district 20221017 to 20221024
    use_restaurant_positions=False,          # Use real restaurant positions
    use_vehicle_count=False,                # Use real number of vehicles
    use_vehicle_positions=False,            # Use random vehicle positions
    use_service_area=False,                  # Use real service area dimensions
    use_deadlines=False,                    # Use real order deadlines
    order_generation_mode="pattern",         # default, pattern, replay
    # None,lunch_dinner_pattern, hourly_pattern or function_pattern
    # Take mean arrival time from the env. config
    temporal_pattern=lunch_dinner_pattern,                  # see comment above
    simulation_start_hour=None,               # e.g., Start at 11 AM
    simulation_duration_hours=None             # e.g., Simulate 8 hours
)

if __name__ == "__main__":
    logger.info("Starting test episode...")
    stats = run_test_episode(
        solver_name="aca",
        meituan_config=custom_config,
        seed=1,
        reposition_idle_vehicles=False,
        visualize=True,
        warmup_duration=0,
    )
    logger.info("\nTest completed!")
