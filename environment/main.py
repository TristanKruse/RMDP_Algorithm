# environment.py
import random
import numpy as np
from typing import Tuple, List, Optional
from environment.state_handler import StateHandler
from environment.order_manager import OrderManager
from environment.vehicle_manager import VehicleManager
from environment.route_processing.route_processor import RouteProcessor
from environment.location_manager import LocationManager
from environment.visualization import VisualizationManager
from datatypes import State
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Silence matplotlib and PIL debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# Create logger instance
logger = logging.getLogger(__name__)
# from .environment.metrics import MetricsCalculator


class RestaurantMealDeliveryEnv:
    def __init__(
        self,
        movement_per_step: float,
        num_restaurants: int = 110,  # Total number of restaurants in service area
        num_vehicles: int = 15,  # Number of delivery vehicles available
        vehicle_capacity: int = 3,
        # could be made more complex/realistic
        service_area_dimensions: Tuple[float, float] = (10.0, 10.0),  # 10km x 10km area
        mean_prep_time: float = 10.0,  # Average food preparation time in minutes
        prep_time_var: float = 2.0,  # Variance in preparation time
        delivery_window: float = 40.0,  # Time allowed for delivery after order placement
        # could be made stochastic (if makes sense)
        downtown_concentration: float = 0.7,
        service_time: float = 2.0,  # Time spent at pickup/delivery locations
        mean_interarrival_time: float = 2.0,  # Average orders per hour per vehicle
        simulation_duration: float = 480.0,  # New parameter in minutes
        cooldown_duration: float = 60.0,  # Add cooldown parameter
        seed: Optional[int] = None,  # Add seed parameter
        # visualization
        visualize: bool = True,  # New parameter to control visualization
        update_interval: float = 0.01,
        reposition_idle_vehicles: bool = False,
        bundling_orders: bool = False,
    ):
        self.reposition_idle_vehicles = reposition_idle_vehicles
        self.bundling_orders = bundling_orders
        # Set random seeds if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Validate cooldown duration
        if cooldown_duration >= simulation_duration:
            raise ValueError("Cooldown duration must be less than simulation duration")

        # Store simulation parameters
        self.simulation_duration = simulation_duration
        self.cooldown_duration = cooldown_duration
        self.current_time = 0
        self.order_generation_end_time = simulation_duration - cooldown_duration  # New parameter

        # Initialize Location Manager
        self.location_manager = LocationManager(
            service_area_dimensions=service_area_dimensions,
            num_restaurants=num_restaurants,
            movement_per_step=movement_per_step,
            downtown_concentration=downtown_concentration,
        )

        # Initialize Vehicle Manager
        self.vehicle_manager = VehicleManager(
            num_vehicles=num_vehicles,
            service_area_dimensions=service_area_dimensions,
            vehicle_capacity=vehicle_capacity,
        )

        # Initialize Order Manager
        self.order_manager = OrderManager(
            mean_prep_time=mean_prep_time,
            prep_time_var=prep_time_var,
            delivery_window=delivery_window,
            service_time=service_time,
            mean_interarrival_time=mean_interarrival_time,
            service_area_dimensions=service_area_dimensions,
        )

        # Initialize Route Processor
        self.route_processor = RouteProcessor(
            service_time=service_time, location_manager=self.location_manager, movement_per_step=movement_per_step
        )

        # Initialize State Handler
        self.state_handler = StateHandler(num_vehicles=num_vehicles)

        # Initialize Visualization
        self.viz_manager = None
        if visualize:
            self.viz_manager = VisualizationManager(
                service_area_dimensions=service_area_dimensions, enabled=visualize, update_interval=update_interval
            )

        # Initialize Metrics Calculator
        # self.metrics = MetricsCalculator()

        # Initialize empty route plan
        self.route_plan = [[] for _ in range(num_vehicles)]

    def _setup_initial_visualization(self):
        """Setup initial visualization state"""
        if self.viz_manager:
            restaurant_positions = self.location_manager.get_restaurant_positions()
            self.viz_manager.initialize_visualization(
                restaurants=restaurant_positions, vehicles=self.vehicle_manager.get_vehicle_positions()
            )

    def get_current_state(self) -> State:
        """Helper method to get current state"""
        return self.state_handler.create_new_state(
            current_time=self.current_time, order_manager=self.order_manager, vehicle_manager=self.vehicle_manager
        )

    def reset(self) -> State:
        """Reset environment state"""
        self.current_time = 0
        self.state_handler.reset()
        self.order_manager.reset()
        self.vehicle_manager.reset()
        self.route_plan = [[] for _ in range(self.vehicle_manager.num_vehicles)]

        if self.viz_manager:
            self.viz_manager.reset()
            self._setup_initial_visualization()
            # Update positions after reset
            restaurant_positions = self.location_manager.get_restaurant_positions()
            vehicle_positions = self.vehicle_manager.get_vehicle_positions()
            self.viz_manager.viz.set_static_positions(restaurants=restaurant_positions, customers=np.array([]))
            self.viz_manager.viz.update_vehicle_positions(vehicle_positions)

        return self.get_current_state()

    def close(self):
        """Clean up resources"""
        if self.viz_manager:
            self.viz_manager.close()

    def step(self, action: Tuple[List[List[int]], set]) -> Tuple[State, float, bool, dict]:
        """Main step method coordinating all components"""
        new_route_plan, postponed_orders = action

        # Store current routes before processing
        self.route_plans = new_route_plan

        # Process postponed orders
        self.order_manager.handle_postponed_orders(postponed_orders)

        # Process routes and collect metrics
        step_metrics = self.route_processor.process_all_routes(
            self.route_plans, self.vehicle_manager, self.order_manager, self.current_time
        )

        # Reposition vehicles to the nearest restaurant, if True
        if self.reposition_idle_vehicles:
            self.vehicle_manager.reposition_idle_vehicles(self.get_current_state(), self.location_manager)

        # Update route plans by removing completed orders
        for vehicle_id, route in enumerate(self.route_plans):
            if route and route[0] in step_metrics["delivered_orders"]:
                self.route_plans[vehicle_id] = route[1:]  # Remove completed order
                # logger.info(f"[DEBUG] Removed completed order {route[0]} from vehicle {vehicle_id}'s route")

        # Update state handler with current routes
        self.state_handler.update_route_plan(self.route_plans)

        self.order_manager.cleanup_delivered_orders()

        # Update visualization with new vehicle positions
        if self.viz_manager:
            self.viz_manager.update_step_visualization(
                vehicles=self.vehicle_manager.vehicles,
                active_orders=self.order_manager.get_active_orders(),
                restaurants=self.location_manager.restaurants,
                current_time=self.current_time,
            )

        # Generate new orders only if not in cooldown period
        if self.current_time < self.order_generation_end_time:
            self.order_manager.generate_new_orders(
                current_time=self.current_time,
                restaurants=self.location_manager.restaurants,
            )

        # Update state and calculate reward
        new_state = self.state_handler.create_new_state(self.current_time, self.order_manager, self.vehicle_manager)
        reward = -sum(step_metrics["delays"])

        # Add additional metrics
        step_metrics.update(
            {
                "total_orders": self.order_manager.next_order_id,
                "active_orders": len(self.order_manager.active_orders),
                "postponed_count": len(postponed_orders),
                "current_routes": self.route_plans.copy(),  # Add routes to metrics for debugging
                "in_cooldown": self.current_time >= self.order_generation_end_time,
                "time_until_cooldown": max(0, self.order_generation_end_time - self.current_time),
            }
        )

        # Check if simulation is done
        done = self.current_time >= self.simulation_duration

        # Increment time
        self.current_time += 1

        return new_state, reward, done, step_metrics
