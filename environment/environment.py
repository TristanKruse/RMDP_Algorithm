# environment.py
import random
import numpy as np
from typing import Tuple, Optional, Dict
from environment.state_handler import StateHandler
from environment.order_manager import OrderManager
from environment.order_generator import OrderGenerator
from environment.vehicle_manager import VehicleManager
from environment.route_processing.route_processor import RouteProcessor
from environment.location_manager import LocationManager
from environment.visualization import VisualizationManager
from datatypes import State, Route
import logging

# Configure logging format
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Silence matplotlib and PIL debug messages
for logger_name in ["matplotlib", "PIL"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


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
        # Demand pattern configuration
        demand_pattern: Optional[Dict] = None,  # New parameter for demand pattern
        # visualization
        visualize: bool = True,  # New parameter to control visualization
        update_interval: float = 0.01,
        reposition_idle_vehicles: bool = False,
    ):
        self.reposition_idle_vehicles = reposition_idle_vehicles
        # Set random seeds if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Validate cooldown duration
        if cooldown_duration >= simulation_duration:
            raise ValueError("Cooldown duration must be less than simulation duration")

        # Store simulation parameters
        self.simulation_duration, self.cooldown_duration, self.current_time, self.order_generation_end_time = (
            simulation_duration,
            cooldown_duration,
            0,
            simulation_duration - cooldown_duration,
        )

        self.location_manager = LocationManager(
            service_area_dimensions,
            num_restaurants,
            movement_per_step,
            downtown_concentration,
        )

        self.vehicle_manager = VehicleManager(
            num_vehicles,
            service_area_dimensions,
            vehicle_capacity,
        )

        self.order_manager = OrderManager(
            mean_prep_time,
            prep_time_var,
            delivery_window,
            service_time,
            mean_interarrival_time,
            service_area_dimensions,
            demand_pattern=demand_pattern,  # Pass demand pattern to OrderManager
            simulation_duration=simulation_duration,  # Pass simulation duration
        )

        if demand_pattern and isinstance(demand_pattern, dict) and demand_pattern.get('type') == 'hourly':
            # Properly initialize the OrderGenerator for pattern mode
            self.order_generator = OrderGenerator(
                mean_interarrival_time=mean_interarrival_time,
                service_area_dimensions=service_area_dimensions,
                delivery_window=delivery_window,
                service_time=service_time,
                mean_prep_time=mean_prep_time,
                prep_time_var=prep_time_var,
                mode="pattern",
                temporal_pattern=demand_pattern.get('hourly_rates', {})
            )
        # Set reference to order manager
        self.order_manager.set_order_generator(self.order_generator)


        self.route_processor = RouteProcessor(
            service_time=service_time, location_manager=self.location_manager, 
            movement_per_step=movement_per_step, 
            reposition_idle_vehicles=reposition_idle_vehicles ,
            vehicle_manager=self.vehicle_manager
        )
        
        self.state_handler = StateHandler(num_vehicles=num_vehicles)

        self.viz_manager = (
            VisualizationManager(service_area_dimensions, visualize, update_interval) if visualize else None
        )

        self.route_plan = {
            i: Route(
                vehicle_id=i,
                sequence=[],  # Empty list of tuples (node_id, pickups, deliveries)
                total_distance=0.0,
                total_time=0.0,
            )
            for i in range(num_vehicles)
        }

    def step(self, action: Tuple[Dict[int, Route], set]) -> Tuple[State, float, bool, dict]:
        # logger.info(f"self.current_time: {self.current_time}")
        # 1. Unpack the solver's action
        new_route_plan, postponed_orders = action

        self.route_plans = new_route_plan

        # 3. Process vehicle movements and order deliveries
        step_metrics = self.route_processor.process_all_routes(
            self.route_plans, self.vehicle_manager, self.order_manager, self.current_time
        )

        # 4. Update routes: Remove completed deliveries
        for vehicle_id, route in self.route_plans.items():
            if route.sequence and any(
                order_id in step_metrics["delivered_orders"] for order_id in route.sequence[0][1]
            ):  # Check pickups set
                self.route_plans[vehicle_id] = Route(
                    vehicle_id=vehicle_id,
                    sequence=route.sequence[1:],  # Keep remaining sequence
                    total_distance=0.0,
                    total_time=0.0,
                )

        # 5. State synchronization
        self.state_handler.update_route_plan(self.route_plans)
        self.order_manager.cleanup_delivered_orders()

        # 6. Optional: Update visualization
        if self.viz_manager:
            self.viz_manager.update_step_visualization(
                vehicles=self.vehicle_manager.vehicles,
                active_orders=self.order_manager.get_active_orders(),
                restaurants=self.location_manager.restaurants,
                current_time=self.current_time,
            )

        # 7. Generate new orders (if not in cooldown)
        if self.current_time < self.order_generation_end_time:
            self.order_manager.generate_new_orders(
                current_time=self.current_time,
                restaurants=self.location_manager.restaurants,
            )

        # 8. Calculate step results
        new_state = self.state_handler.create_new_state(
            self.current_time, self.order_manager, self.vehicle_manager, self.location_manager
        )
        reward = -sum(step_metrics["delays"])

        # 9. Update metrics
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
        # 10. Time step and return results
        self.current_time += 1
        return new_state, reward, self.current_time >= self.simulation_duration, step_metrics

    def _setup_initial_visualization(self):
        if self.viz_manager:
            restaurant_positions = self.location_manager.get_restaurant_positions()
            self.viz_manager.initialize_visualization(
                restaurants=restaurant_positions, vehicles=self.vehicle_manager.get_vehicle_positions()
            )

    def get_current_state(self) -> State:
        return self.state_handler.create_new_state(
            current_time=self.current_time,
            order_manager=self.order_manager,
            vehicle_manager=self.vehicle_manager,
            location_manager=self.location_manager,
        )

    def reset(self) -> State:
        self.current_time = 0
        self.state_handler.reset()
        self.order_manager.reset()
        self.vehicle_manager.reset()
        self.route_plan = [[] for _ in range(self.vehicle_manager.num_vehicles)]

        if self.viz_manager:
            self.viz_manager.reset()
            self._setup_initial_visualization()
            # Update positions after reset
            self.viz_manager.viz.set_static_positions(
                restaurants=self.location_manager.get_restaurant_positions(), customers=np.array([])
            )
            self.viz_manager.viz.update_vehicle_positions(self.vehicle_manager.get_vehicle_positions())
        return self.get_current_state()
