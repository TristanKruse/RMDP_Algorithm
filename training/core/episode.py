import os
import logging
from datetime import datetime
from typing import Optional, Dict

from environment.environment import RestaurantMealDeliveryEnv
from environment.order_generator import OrderGenerator

from .stats import initialize_episode_stats, calculate_capacity_metrics
from .bundling import detect_bundles
from ..config.env_config import get_env_config
from ..config.solver_config import SOLVERS
from ..utils.visualization import visualize_restaurant_distribution
from ..utils.file_io import save_results

from models.aca_policy.aca_policy import ACA

logger = logging.getLogger(__name__)


def prepare_solver_input(state):
    """Extracts decision-relevant information from full state (following Ulmer et al.).

    Returns dict containing:
    - tk: current time
    - Dk: orders with their properties (tD, RD, VD, LD)
    - Θk: current route plan
    - and objects for nearest neighbour, nodes + vehicle positions
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
                "deadline": order.deadline,
                "postpone_count": order.postpone_count,  # Add the postpone_count attribute
            }

    # Add vehicle positions, needed for fastest vehicle solver
    vehicle_positions = {}
    for vehicle in state.vehicles:  # Iterate directly over list
        vehicle_positions[vehicle.id] = vehicle.current_location

    vehicle_movement_info = {}
    for vehicle in state.vehicles:
        vehicle_movement_info[vehicle.id] = {
            "movement_progress": vehicle.movement_progress,
            "current_phase": vehicle.current_phase,
        }

    return {
        "time": state.time,  # tk: point of time
        "unassigned_orders": orders_info,  # Dk: set of orders with their properties
        "route_plan": state.route_plan,  # Θk: current route plan
        "vehicle_positions": vehicle_positions,  # for fastest vehicle, nearest neighbour
        "nodes": nodes,  # Add nodes to the state dictionary
        "orders": state.orders,
        "vehicle_movement_info": vehicle_movement_info,
    }


def run_test_episode(
    solver_name: str = "fastest",
    solver=None,  # Add parameter to accept an existing solver
    meituan_config=None,
    seed: Optional[int] = None,
    reposition_idle_vehicles: bool = False,
    visualize: bool = False,
    warmup_duration: int = 60,
    save_rl_model: bool = False,
    rl_model_path: str = None,
    save_results_to_disk=True,
    env_config: Optional[Dict] = None,
    exploration_rate: Optional[float] = None,
    training_mode: bool = True,  # Parameter to control RL training mode
    # RL hyperparameters
    rl_learning_rate: float = 0.0005,
    rl_discount_factor: float = 0.95,
    rl_exploration_rate: float = 0.9,
    rl_min_exploration_rate: float = 0.2,
    rl_batch_size: int = 64,
    rl_target_update_frequency: int = 50,
    rl_replay_buffer_capacity: int = 10000,
    rl_bundling_reward: float = 0.05,
    rl_postponement_penalty: float = -0.005,
    rl_on_time_reward: float = 0.2,
    aca_buffer: int = 15,  # To override ACA buffer size, for buffer tuning
):
    """Main function to run a test episode with the specified solver."""
    simulation_duration = get_env_config(None)["simulation_duration"]
    speed = 8  # 8 km/h for 30-second intervals (was 16 km/h for 1-minute intervals)
    street_network_factor = 1.0  # 1.4 in paper, we calculated the average speed over the euclidic distance
    movement_per_step = (speed / 60) / street_network_factor  # km per minute adjusted for street network

    env_params = get_env_config(movement_per_step)
    cooldown_duration = env_params["cooldown_duration"]
    env_params.update(
        {
            "seed": seed,
            "reposition_idle_vehicles": reposition_idle_vehicles,
            "visualize": visualize,
        }
    )

    # For bundling rate.
    delivered_orders_set = set()

    # Apply custom environment config if provided
    if env_config:
        env_params.update(env_config)

    # Apply Meituan data configuration to environment parameters
    if meituan_config is not None:
        env_params = meituan_config.apply_to_env_params(env_params)

    # Create the environment
    env = RestaurantMealDeliveryEnv(**env_params)

    # Apply Meituan data configuration to environment (if any additional setup is needed)
    if meituan_config is not None:
        meituan_config.apply_to_environment(env)

    # Create and set the order generator
    if (
        meituan_config is None
        or not hasattr(meituan_config, "order_generation_mode")
        or meituan_config.order_generation_mode != "pattern"
    ):
        # Check if a demand pattern is provided in env_params
        if env_params.get("demand_pattern"):
            # Use pattern mode if a demand pattern is specified
            demand_pattern = env_params.get("demand_pattern")
            temporal_pattern = (
                demand_pattern.get("hourly_rates", {}) if isinstance(demand_pattern, dict) else demand_pattern
            )
            order_generator = OrderGenerator(
                mean_interarrival_time=env_params["mean_interarrival_time"],
                service_area_dimensions=env_params["service_area_dimensions"],
                delivery_window=env_params["delivery_window"],
                service_time=env_params["service_time"],
                mean_prep_time=env_params["mean_prep_time"],
                prep_time_var=env_params["prep_time_var"],
                mode="pattern",
                temporal_pattern=temporal_pattern,
            )
        else:
            # Use default mode with parameters from env_params
            order_generator = OrderGenerator(
                mean_interarrival_time=env_params["mean_interarrival_time"],
                service_area_dimensions=env_params["service_area_dimensions"],
                delivery_window=env_params["delivery_window"],
                service_time=env_params["service_time"],
                mean_prep_time=env_params["mean_prep_time"],
                prep_time_var=env_params["prep_time_var"],
                mode="default",
                temporal_pattern=None,
            )
    else:
        # Use pattern-based generator if specified by meituan_config
        order_generator = OrderGenerator(
            mean_interarrival_time=env_params["mean_interarrival_time"],
            service_area_dimensions=env_params["service_area_dimensions"],
            delivery_window=env_params["delivery_window"],
            service_time=env_params["service_time"],
            mean_prep_time=env_params["mean_prep_time"],
            prep_time_var=env_params["prep_time_var"],
            mode="pattern",
            temporal_pattern=meituan_config.temporal_pattern.get("hourly_rates", {}),
        )

    # Set the order generator on the order manager
    env.order_manager.set_order_generator(order_generator)

    # Reset the environment to properly initialize
    state = env.reset()

    logging.info(f"Starting simulation with solver: {solver_name}")

    # Update SOLVERS dictionary to pass RL hyperparameters only if a new solver needs to be created
    if solver is None:
        global SOLVERS
        SOLVERS = {
            "aca": lambda movement_per_step, location_manager: ACA(
                location_manager=location_manager,
                buffer=aca_buffer,
                max_postponements=0,
                max_postpone_time=0,
                vehicle_capacity=5,
                service_time=4.0,
                mean_prep_time=13,
                delivery_window=35.0,
                postponement_method="heuristic",
            ),
            "rl_aca": lambda movement_per_step, location_manager: ACA(
                location_manager=location_manager,
                buffer=15,
                max_postponements=3,
                max_postpone_time=10,
                vehicle_capacity=3,
                service_time=2.0,
                mean_prep_time=13,
                delivery_window=40.0,
                postponement_method="rl-aca",
                rl_training_mode=training_mode,
                rl_state_size=7,
                rl_model_path=rl_model_path,  # Pass rl_model_path to ACA constructor
                rl_learning_rate=rl_learning_rate,
                rl_discount_factor=rl_discount_factor,
                rl_exploration_rate=rl_exploration_rate,
                rl_min_exploration_rate=rl_min_exploration_rate,
                rl_batch_size=rl_batch_size,
                rl_target_update_frequency=rl_target_update_frequency,
                rl_replay_buffer_capacity=rl_replay_buffer_capacity,
                rl_bundling_reward=rl_bundling_reward,
                rl_postponement_penalty=rl_postponement_penalty,
                rl_on_time_reward=rl_on_time_reward,
            ),
            "bundler": lambda s, loc_manager: FastestBundler(
                movement_per_step=s,
                location_manager=loc_manager,
                max_bundle_size=3,
            ),
            "fastest": lambda s, loc_manager: FastestVehicleSolver(movement_per_step=s, location_manager=loc_manager),
        }

        solver = SOLVERS[solver_name](movement_per_step, env.location_manager)

    # Add this block to update the exploration rate when provided
    if solver_name == "rl_aca" and exploration_rate is not None:
        if hasattr(solver, "postponement"):
            solver.postponement.exploration_rate = exploration_rate

    # Initialize statistics
    episode_stats = initialize_episode_stats()

    # Track delay predictions vs. actual for analysis
    predicted_delays = {}  # order_id -> predicted delay
    actual_delays = {}  # order_id -> actual delay

    done = False
    total_reward = 0
    step = 0
    order_restaurant_map = {}  # Map of order IDs to restaurant IDs

    # Main simulation loop
    while not done and step < simulation_duration:
        # Check for pause state
        if env.viz_manager and env.viz_manager.is_paused():
            plt.pause(0.1)  # Keep window responsive while paused
            continue

        if solver_name == "rl_aca" and exploration_rate is not None:
            route_plan, postponed_orders = solver.solve(prepare_solver_input(state), exploration_rate=exploration_rate)
        else:
            route_plan, postponed_orders = solver.solve(prepare_solver_input(state))

        detect_bundles(route_plan, state, episode_stats)
        next_state, reward, done, info = env.step((route_plan, postponed_orders))

        # Process delivered orders and update statistics
        if info["deliveries"] > 0:
            for order_id in info.get("delivered_orders", set()):
                delivered_orders_set.add(order_id)
                if order_id in order_restaurant_map:
                    restaurant_id = order_restaurant_map[order_id]
                    if restaurant_id not in episode_stats["orders_per_restaurant"]:
                        episode_stats["orders_per_restaurant"][restaurant_id] = 0
                    episode_stats["orders_per_restaurant"][restaurant_id] += 1

                # Process order metrics
                delivered_order = next((o for o in state.orders if o.id == order_id), None)
                if not delivered_order:
                    delivered_order = next((o for o in next_state.orders if o.id == order_id), None)

                if delivered_order:
                    # Calculate actual delay
                    actual_delay = max(0, delivered_order.delivery_time - delivered_order.deadline)
                    actual_delays[order_id] = actual_delay

                    # Update restaurant tracking
                    restaurant_id = order_restaurant_map.get(order_id)

                    # Update various metrics
                    if hasattr(delivered_order, "true_prep_time"):
                        true_prep = delivered_order.true_prep_time
                        episode_stats["true_prep_times"].append(true_prep)
                        episode_stats["max_true_prep_time"] = max(episode_stats["max_true_prep_time"], true_prep)
                        if restaurant_id:
                            if restaurant_id not in episode_stats["true_prep_by_restaurant"]:
                                episode_stats["true_prep_by_restaurant"][restaurant_id] = []
                            episode_stats["true_prep_by_restaurant"][restaurant_id].append(true_prep)

                    if hasattr(delivered_order, "order_wait_time"):
                        order_wait = delivered_order.order_wait_time
                        episode_stats["order_wait_times"].append(order_wait)
                        episode_stats["max_order_wait_time"] = max(episode_stats["max_order_wait_time"], order_wait)
                        if restaurant_id:
                            if restaurant_id not in episode_stats["order_wait_by_restaurant"]:
                                episode_stats["order_wait_by_restaurant"][restaurant_id] = []
                            episode_stats["order_wait_by_restaurant"][restaurant_id].append(order_wait)

                    if hasattr(delivered_order, "total_time_to_pickup"):
                        total_pickup = delivered_order.total_time_to_pickup
                        episode_stats["total_pickup_times"].append(total_pickup)
                        episode_stats["max_total_pickup_time"] = max(
                            episode_stats["max_total_pickup_time"], total_pickup
                        )
                        if restaurant_id:
                            if restaurant_id not in episode_stats["total_pickup_by_restaurant"]:
                                episode_stats["total_pickup_by_restaurant"][restaurant_id] = []
                            episode_stats["total_pickup_by_restaurant"][restaurant_id].append(total_pickup)

                    if hasattr(delivered_order, "driver_wait_time") and delivered_order.driver_wait_time > 0:
                        wait_time = delivered_order.driver_wait_time
                        episode_stats["total_driver_wait_time"] += wait_time
                        episode_stats["driver_wait_orders"] += 1
                        if restaurant_id:
                            if restaurant_id not in episode_stats["driver_wait_by_restaurant"]:
                                episode_stats["driver_wait_by_restaurant"][restaurant_id] = []
                            episode_stats["driver_wait_by_restaurant"][restaurant_id].append(wait_time)

        # Add idle time tracking
        if step >= warmup_duration and step < (simulation_duration - cooldown_duration):
            episode_stats["active_period_steps"] += 1
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

    # Handle undelivered orders at the end of the simulation
    undelivered_orders = set()
    for order in state.orders:
        if order.id not in delivered_orders_set:
            undelivered_orders.add(order.id)

    if undelivered_orders:
        logger.info(f"Processing {len(undelivered_orders)} undelivered orders at simulation end (time: {state.time})")

        # Track total delay from undelivered orders
        total_undelivered_delay = 0

        for order_id in undelivered_orders:
            order = next((o for o in state.orders if o.id == order_id), None)
            if order:
                current_delay = max(0, state.time - order.deadline)
                total_undelivered_delay += current_delay

                # Add delay to episode statistics and total reward
                episode_stats["delay_values"].append(current_delay)
                episode_stats["late_orders"].add(order_id)
                episode_stats["max_delay"] = max(episode_stats["max_delay"], current_delay)
                total_reward -= current_delay  # Add undelivered order delay to total reward

                # logger.info(
                #     f"Undelivered order {order_id}: "
                #     f"request_time={order.request_time:.1f}, "
                #     f"deadline={order.deadline:.1f}, "
                #     f"current_time={state.time:.1f}, "
                #     f"delay={current_delay:.1f}"
                # )

                # Update RL model if applicable
                if (
                    solver_name == "rl_aca"
                    and hasattr(solver, "postponement")
                    and hasattr(solver.postponement, "record_order_delivery")
                ):
                    was_bundled = order_id in episode_stats["bundled_orders"]
                    solver.postponement.record_order_delivery(
                        order_id, current_delay, state.time, was_bundled=was_bundled
                    )
                    # logger.info(
                    #     f"RL feedback for order {order_id}: "
                    #     f"delay={current_delay:.1f}, "
                    #     f"was_bundled={was_bundled}"
                    # )

        # Log summary of undelivered orders
        logger.info(
            f"Summary of undelivered orders: "
            f"count={len(undelivered_orders)}, "
            f"total_delay={total_undelivered_delay:.1f}, "
            f"avg_delay={total_undelivered_delay/len(undelivered_orders):.1f}"
        )

        # Log total statistics including undelivered orders
        total_delay = sum(episode_stats["delay_values"])
        logger.info(
            f"Total statistics including undelivered orders: "
            f"total_delay={total_delay:.1f}, "
            f"max_delay={episode_stats['max_delay']:.1f}, "
            f"total_orders={episode_stats['total_orders']}, "
            f"delivered={episode_stats['orders_delivered']}, "
            f"undelivered={len(undelivered_orders)}"
        )

    # Save results
    episode_stats["total_reward"] = total_reward

    if save_results_to_disk:
        # Calculate final metrics
        episode_stats = calculate_capacity_metrics(
            episode_stats, simulation_duration, cooldown_duration, warmup_duration
        )

        # Save results to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(
            episode_stats,
            solver_name,
            seed,
            meituan_config,
            solver_params=get_solver_params(solver),
            env_params=env_params,
        )
        visualize_restaurant_distribution(episode_stats, solver_name, timestamp)

    # Save RL model if requested
    if save_rl_model and hasattr(solver, "save_rl_model"):
        if rl_model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = os.path.join("data", "models", f"rl_aca_{timestamp}.pt")
        else:
            model_save_path = rl_model_path

        dir_path = os.path.dirname(model_save_path)
        os.makedirs(dir_path, exist_ok=True)
        solver.save_rl_model(model_save_path)

    return episode_stats


def get_solver_params(solver):
    """Extract solver parameters for logging purposes."""
    if hasattr(solver, "postponement_method"):
        return {
            "postponement_method": solver.postponement_method,
            "vehicle_capacity": solver.route_utils.vehicle_capacity,
            "service_time": solver.time_calculator.service_time,
            "postponement": solver.postponement.__class__.__name__ if hasattr(solver, "postponement") else None,
        }
    else:
        return {
            "solver_type": solver.__class__.__name__,
            "vehicle_capacity": getattr(solver, "vehicle_capacity", None),
            "service_time": getattr(solver, "service_time", None),
        }
