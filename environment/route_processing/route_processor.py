# environment/route_processor.py
from typing import Dict, List, Any
from datatypes import Location, Vehicle
from environment.route_processing.handlers import Handlers
from environment.route_processing.metrics_methods import MetricsMethods
from environment.route_processing.movement_location import MovementLocation
from environment.route_processing.phase_management import PhaseManagement
from environment.route_processing.service_time import ServiceTime
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger instance
logger = logging.getLogger(__name__)
# process_all_routes
#     └── _process_single_route
#         └── _process_single_order
#             └── Movement & Status Updates

# lag zwischen assignement und movement? Zeile 62


class RouteProcessor:
    def __init__(self, location_manager, service_time: float, movement_per_step: float):
        self.location_manager = location_manager
        self.service_time = service_time
        self.movement_per_step = movement_per_step

        self.movement_location = MovementLocation(location_manager)
        self.phase_management = PhaseManagement(location_manager)
        self.service_time = ServiceTime(service_time)
        self.metrics_methods = MetricsMethods()
        self.handlers = Handlers(location_manager, service_time)

        # Initialize idle time tracking
        self.total_idle_time = 0
        self.total_time_steps = 0
        self.vehicle_idle_times = {}

    # def process_all_routes(self, route_plan, vehicle_manager, order_manager, current_time):
    #     # 1. Initialize empty metrics dictionary
    #     metrics = self.metrics_methods._initialize_metrics()

    #     # Update total time steps
    #     self.total_time_steps += 1
    #     idle_vehicles = 0

    #     # 2. Process each vehicle and its route in the route plan
    #     for vehicle_id, current_route in enumerate(route_plan):
    #         # 3. Get vehicle object
    #         vehicle = vehicle_manager.get_vehicle_by_id(vehicle_id)

    #         # 4. Skip if vehicle has no route
    #         if not current_route:
    #             continue

    #         # Initialize vehicle idle time tracking if not exists
    #         if vehicle_id not in self.vehicle_idle_times:
    #             self.vehicle_idle_times[vehicle_id] = 0

    #         # 5. Get first order in route and check if it exists in active orders
    #         current_order_id = current_route[0]
    #         order = next((o for o in order_manager.active_orders if o.id == current_order_id), None)

    #         # 6. If order exists, process it
    #         if order:
    #             # 7. Process vehicle movement and order status, returns updated position and metrics
    #             new_loc, distance, delay, completed = self._process_single_order(
    #                 order_id=current_order_id,
    #                 vehicle=vehicle,
    #                 current_loc=vehicle.current_location,
    #                 order_manager=order_manager,
    #                 current_time=current_time,
    #             )
    #             # 8. Update metrics based on processing results
    #             metrics["distance"] += distance
    #             if delay > 0:
    #                 metrics["delays"].append(delay)
    #                 metrics["late_orders"].add(current_order_id)
    #             if completed:
    #                 metrics["deliveries"] += 1
    #                 metrics["delivered_orders"].add(current_order_id)

    #             # 9. Update vehicle location
    #             vehicle.current_location = new_loc

    #         # 10. If no order but vehicle is idle, process idle movement
    #         elif not vehicle.current_phase:
    #             self._process_idle_movement(vehicle, metrics)

    #     # 11. Return accumulated metrics
    #     return metrics

    def process_all_routes(self, route_plan, vehicle_manager, order_manager, current_time):
        # 1. Initialize empty metrics dictionary
        metrics = self.metrics_methods._initialize_metrics()

        # Update total time steps and idle tracking
        self.total_time_steps += 1
        idle_vehicles = 0

        # 2. Process each vehicle and its route in the route plan
        for vehicle_id, current_route in enumerate(route_plan):
            # 3. Get vehicle object
            vehicle = vehicle_manager.get_vehicle_by_id(vehicle_id)

            # Initialize vehicle idle time tracking
            if vehicle_id not in self.vehicle_idle_times:
                self.vehicle_idle_times[vehicle_id] = 0

            # Check if vehicle is idle
            is_idle = not current_route and (not hasattr(vehicle, "current_phase") or vehicle.current_phase is None)
            if is_idle:
                idle_vehicles += 1
                self.vehicle_idle_times[vehicle_id] += 1

            # 4. Skip if vehicle has no route
            if not current_route:
                if not vehicle.current_phase:
                    self._process_idle_movement(vehicle, metrics)
                continue

            # 5. Get first order in route and check if it exists in active orders
            current_order_id = current_route[0]
            order = next((o for o in order_manager.active_orders if o.id == current_order_id), None)

            # 6. If order exists, process it
            if order:
                # 7. Process vehicle movement and order status
                new_loc, distance, delay, completed = self._process_single_order(
                    order_id=current_order_id,
                    vehicle=vehicle,
                    current_loc=vehicle.current_location,
                    order_manager=order_manager,
                    current_time=current_time,
                )
                # 8. Update metrics based on processing results
                metrics["distance"] += distance
                if delay > 0:
                    metrics["delays"].append(delay)
                    metrics["late_orders"].add(current_order_id)
                if completed:
                    metrics["deliveries"] += 1
                    metrics["delivered_orders"].add(current_order_id)

                # 9. Update vehicle location
                vehicle.current_location = new_loc

        # Update idle metrics
        self.total_idle_time += idle_vehicles
        total_vehicles = len(vehicle_manager.vehicles)
        metrics.update(
            {
                "current_idle_rate": idle_vehicles / total_vehicles,
                "average_idle_rate": self.total_idle_time / (self.total_time_steps * total_vehicles),
                "vehicle_idle_rates": {
                    vid: idle_time / self.total_time_steps for vid, idle_time in self.vehicle_idle_times.items()
                },
            }
        )

        return metrics

    def _process_single_route(
        self, route: List[int], vehicle: Vehicle, current_loc: Location, order_manager, current_time: float
    ):
        # 1. Initialize metrics dictionary for this route
        metrics = {"distance": 0, "deliveries": 0, "delays": [], "late_orders": set(), "delivered_orders": set()}

        # 2. Return empty metrics if no route exists
        if not route:
            return metrics

        # 3. Get first order ID from route and find corresponding active order
        order_id = route[0]
        order = next((o for o in order_manager.active_orders if o.id == order_id), None)
        if not order:
            return metrics

        # 4. Process order movement and get updated status
        new_loc, distance, delay, completed = self._process_single_order(
            order_id=order_id,
            vehicle=vehicle,
            current_loc=current_loc,
            order_manager=order_manager,
            current_time=current_time,
        )

        # 5. Update metrics based on processing results
        metrics["distance"] += distance
        if delay > 0:
            metrics["delays"].append(delay)
            metrics["late_orders"].add(order_id)
        if completed:
            metrics["deliveries"] += 1
            metrics["delivered_orders"].add(order_id)

        # 6. Update vehicle location and return metrics
        vehicle.current_location = new_loc
        return metrics

    def _process_single_order(self, order_id, vehicle, current_loc, order_manager, current_time):
        # 1. Initialize default return values
        new_loc = current_loc
        step_distance = delay = 0.0
        completed = False

        # 2. Get and validate order
        order = next((o for o in order_manager.active_orders if o.id == order_id), None)
        if not order:
            return current_loc, step_distance, delay, completed

        # 3. Initialize vehicle phase if needed
        if not hasattr(vehicle, "current_phase") or vehicle.current_phase is None:
            vehicle.current_phase = self.phase_management._initialize_vehicle_phase(order_id, order, current_loc)

        # 4. If vehicle is servicing order
        if vehicle.current_phase["is_servicing"]:
            if self.service_time._process_service_time(vehicle.current_phase):
                return self.handlers._handle_service_completion(vehicle, order, current_loc, current_time)
            return current_loc, 0.0, 0.0, False

        # 5. Handle vehicle movement
        progress = self.movement_location._update_phase_progress(vehicle.current_phase)
        new_loc, step_distance = self.movement_location._calculate_movement(
            vehicle.current_phase["start_loc"], vehicle.current_phase["target_loc"], progress
        )

        # 6. Check for arrival
        if progress >= 1.0:
            return self.handlers._handle_arrival(vehicle, order, new_loc, current_time)

        return new_loc, step_distance, delay, completed

    def _process_idle_movement(self, vehicle, metrics):
        """Process movement for idle vehicles with destinations."""
        # 1. Early return if no destination
        if not vehicle.current_destination:
            return
        current_time = metrics.get("current_time", 0)

        # 2. Initialize travel time if starting movement
        if vehicle.movement_progress == 0.0:
            vehicle.total_travel_time = max(
                0.001, self.location_manager.get_travel_time(vehicle.current_location, vehicle.current_destination)
            )
            vehicle.movement_start_time = current_time

        # 3. Update movement progress
        vehicle.movement_progress = min(1.0, vehicle.movement_progress + (1.0 / vehicle.total_travel_time))

        # 4. Calculate and update new position
        new_loc = self.location_manager.interpolate_position(
            vehicle.current_location, vehicle.current_destination, vehicle.movement_progress
        )

        # 5. Update metrics
        step_distance = self.location_manager.get_travel_time(vehicle.current_location, new_loc)
        metrics["distance"] += step_distance
        vehicle.current_location = new_loc

        # 6. Reset if destination reached
        if vehicle.movement_progress >= 1.0:
            vehicle.current_destination = None
            vehicle.movement_progress = 0.0
            vehicle.total_travel_time = 0.0
