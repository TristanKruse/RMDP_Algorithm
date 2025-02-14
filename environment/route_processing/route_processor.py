# environment/route_processor.py
from datatypes import Route
from environment.route_processing.handlers import Handlers
from environment.route_processing.metrics_methods import MetricsMethods
from environment.route_processing.movement_location import MovementLocation
from environment.route_processing.phase_management import PhaseManagement
from environment.route_processing.service_time import ServiceTime
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(module)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger instance
logger = logging.getLogger(__name__)

# lag zwischen assignement und movement? Zeile 62


class RouteProcessor:
    def __init__(self, location_manager, service_time: float, movement_per_step: float, reposition_idle_vehicles: bool = False, vehicle_manager=None ):
        self.location_manager = location_manager
        self.service_time = service_time
        self.movement_per_step = movement_per_step
        self.reposition_idle_vehicles = reposition_idle_vehicles
        self.vehicle_manager = vehicle_manager

        self.movement_location = MovementLocation(location_manager)
        self.phase_management = PhaseManagement(location_manager)
        self.service_time = ServiceTime(service_time)
        self.metrics_methods = MetricsMethods()
        self.handlers = Handlers(location_manager, service_time)

        # Initialize idle time tracking
        self.total_idle_time = 0
        self.total_time_steps = 0
        self.vehicle_idle_times = {}

        self.route_plan = None

    def process_all_routes(self, route_plan, vehicle_manager, order_manager, current_time):
        """Process all vehicle routes."""
        logger = logging.getLogger(__name__)
        self.route_plan = route_plan
        
        # 1. Initialize metrics and counters
        metrics = self.metrics_methods._initialize_metrics()
        self.total_time_steps += 1
        idle_vehicles = 0

        # 2. Process each vehicle's route
        for vehicle_id, route in route_plan.items():
            vehicle = vehicle_manager.get_vehicle_by_id(vehicle_id)
            current_route = route.sequence
            
            # Handle idle time tracking
            if vehicle_id not in self.vehicle_idle_times:
                self.vehicle_idle_times[vehicle_id] = 0

            # Handle idle vehicles
            is_idle = not current_route and (not hasattr(vehicle, "current_phase") or vehicle.current_phase is None)
            if is_idle:
                logger.info(f"Vehicle {vehicle_id} is idle")
                logger.info(f"Current destination: {vehicle.current_destination}")
                logger.info(f"Movement progress: {vehicle.movement_progress}")
                idle_vehicles += 1
                self.vehicle_idle_times[vehicle_id] += 1

                # In route_processor.py
                if self.reposition_idle_vehicles:
                    if not vehicle.current_destination:
                        # Pass route_plan along with other parameters
                        self.vehicle_manager.assign_idle_vehicle_destinations(
                            order_manager.active_orders,
                            self.location_manager,
                            route_plan  # Add this
                        )
                    if vehicle.current_destination:
                        logger.info(f"Moving idle vehicle {vehicle_id} to destination")
                        self._execute_idle_movement(vehicle, metrics)
                continue

            if not current_route:
                continue

            # Process current stop
            first_stop = current_route[0]
            node_id, pickups, deliveries = first_stop
            new_loc = vehicle.current_location  # Initialize new_loc with current location

            # Check vehicle phase first
            if (hasattr(vehicle, "current_phase") and vehicle.current_phase is not None and vehicle.current_phase.get("stage") == "delivery"):
                # Handle as delivery phase
                bundle_orders = vehicle.current_phase.get("bundle_orders", set())
                current_order = vehicle.current_phase.get("order_id")
                
                if bundle_orders:
                    logger.info(f"Processing delivery phase for bundle: {bundle_orders}, current order: {current_order}")
                    new_loc, distance, delay, completed = self._process_bundle(
                        pickup_orders=bundle_orders,
                        vehicle=vehicle,
                        current_loc=vehicle.current_location,
                        order_manager=order_manager,
                        current_time=current_time
                    )
                    
                    metrics["distance"] += distance
                    if delay > 0:
                        metrics["delays"].append(delay)
                        metrics["late_orders"].add(current_order)
                    
                    if completed:
                        logger.info(f"Completed delivery for order {current_order}")
                        metrics["deliveries"] += 1
                        metrics["delivered_orders"].add(current_order)
                        
                        # Update route plan for remaining deliveries
                        remaining_orders = bundle_orders - {current_order}
                        if remaining_orders:
                            logger.info(f"Updating route for remaining orders: {remaining_orders}")
                            new_sequence = [(node_id, set(), remaining_orders)]
                            route_plan[vehicle_id].sequence = new_sequence
                        else:
                            logger.info("No more deliveries in bundle, clearing route")
                            route_plan[vehicle_id].sequence = []

            else:
                # Handle regular pickup phase
                if pickups:
                    new_loc, distance, delay, completed = self._process_bundle(
                        pickup_orders=pickups,
                        vehicle=vehicle,
                        current_loc=vehicle.current_location,
                        order_manager=order_manager,
                        current_time=current_time
                    )
                    metrics["distance"] += distance
                    
                    if completed:
                        route_plan[vehicle_id].sequence = current_route[1:]
                    else:
                        route_plan[vehicle_id].sequence[0] = (node_id, pickups, deliveries)
                
                elif deliveries:
                    order_id = next(iter(deliveries))
                    order = next((o for o in order_manager.active_orders if o.id == order_id), None)
                    
                    if order and order.status == "picked_up":
                        new_loc, distance, delay, completed = self._process_single_order(
                            order_id=order_id,
                            vehicle=vehicle,
                            current_loc=vehicle.current_location,
                            order_manager=order_manager,
                            current_time=current_time
                        )
                        
                        metrics["distance"] += distance
                        if delay > 0:
                            metrics["delays"].append(delay)
                            metrics["late_orders"].add(order_id)
                        if completed:
                            metrics["deliveries"] += 1
                            metrics["delivered_orders"].add(order_id)
                            route_plan[vehicle_id].sequence = current_route[1:]
                    else:
                        logger.info(f"Order {order_id} not found or not picked up")

            # Update vehicle location
            vehicle.current_location = new_loc

        # Update metrics
        self.total_idle_time += idle_vehicles
        total_vehicles = len(vehicle_manager.vehicles)
        metrics.update({
            "current_idle_rate": idle_vehicles / total_vehicles,
            "average_idle_rate": self.total_idle_time / (self.total_time_steps * total_vehicles),
            "vehicle_idle_rates": {
                vid: idle_time / self.total_time_steps 
                for vid, idle_time in self.vehicle_idle_times.items()
            }
        })

        return metrics

    def _process_bundle(self, pickup_orders, vehicle, current_loc, order_manager, current_time):
        """Process a bundle of orders for pickup or delivery."""
        # Initialize default return values
        new_loc = current_loc
        step_distance = delay = 0.0
        completed = False

        # Get and validate all orders in bundle
        bundle_orders = {
            order_id: next((o for o in order_manager.active_orders if o.id == order_id), None)
            for order_id in pickup_orders
        }
        if not all(bundle_orders.values()):
            logger.info("Not all orders in bundle found")
            return current_loc, step_distance, delay, completed

        # Validate route plan
        if self.route_plan and vehicle.id in self.route_plan:
            route_sequence = self.route_plan[vehicle.id].sequence
            if route_sequence:
                first_stop = route_sequence[0]

        # Initialize or check vehicle phase
        if not hasattr(vehicle, "current_phase") or vehicle.current_phase is None:
            first_order = bundle_orders[next(iter(bundle_orders))]  # Get first order
            vehicle.current_phase = self.phase_management._initialize_vehicle_phase(
                order_id=next(iter(bundle_orders)),  # Pass first order id
                order=first_order,
                current_loc=current_loc
            )
            # Add complete bundle information
            vehicle.current_phase.update({
                "is_bundle": True,
                "bundle_orders": set(pickup_orders),
                "order_ids": set(pickup_orders),
                "initial_bundle_size": len(pickup_orders)
            })

        # Handle service state
        if vehicle.current_phase["is_servicing"]:
            logger.info("Vehicle is in service state")
            if self.service_time._process_service_time(vehicle.current_phase):
                logger.info("Service completed, handling completion")
                return self.handlers._handle_service_completion(
                    vehicle=vehicle,
                    orders=bundle_orders,
                    current_loc=current_loc,
                    current_time=current_time,
                    order_manager=order_manager,
                    route_plan=self.route_plan,
                    pickup_orders=pickup_orders
                )
            return current_loc, 0.0, 0.0, False

        # Handle movement
        progress = self.movement_location._update_phase_progress(vehicle.current_phase)
        new_loc, step_distance = self.movement_location._calculate_movement(
            vehicle.current_phase["start_loc"], 
            vehicle.current_phase["target_loc"], 
            progress
        )

        # Check arrival
        if progress >= 1.0:
            # Validate bundle orders are still valid
            valid_orders = all(order_id in pickup_orders for order_id in vehicle.current_phase.get("bundle_orders", set()))
            if not valid_orders:
                vehicle.current_phase = None
                return current_loc, step_distance, delay, completed

            return self.handlers._handle_arrival(
                vehicle=vehicle,
                orders=bundle_orders,
                new_loc=new_loc,
                current_time=current_time,
                order_manager=order_manager,
                route_plan=self.route_plan,
                pickup_orders=pickup_orders
            )

        return new_loc, step_distance, delay, completed

    def _execute_idle_movement(self, vehicle, metrics):
        """Process movement for idle vehicles with destinations."""
        logger.info(f"Executing idle movement for vehicle {vehicle.id}")
        logger.info(f"From: {vehicle.current_location}")
        logger.info(f"To: {vehicle.current_destination}")
        logger.info(f"Progress: {vehicle.movement_progress}")
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

