# environment/route_processor.py
from typing import Dict, List, Tuple, Any
from datatypes import Location, Vehicle
import numpy as np

# process_all_routes
#     └── _process_single_route
#         └── _process_single_order
#             └── Movement & Status Updates

# In route_processor.py


class RouteProcessor:
    def __init__(self, location_manager, service_time: float, movement_per_step: float):
        self.location_manager = location_manager
        self.service_time = service_time
        self.movement_per_step = movement_per_step
        self.vehicle_assignments = {}

    # === Assignment Management Methods ===
    def _cleanup_stale_assignments(self, order_manager):
        """Clean up stale assignments and return active orders"""
        for vehicle_id, order_id in list(self.vehicle_assignments.items()):
            if order_id is not None:
                order = next((o for o in order_manager.active_orders if o.id == order_id), None)
                if not order:
                    print(f"[DEBUG] Cleaning up stale assignment: Vehicle {vehicle_id}, Order {order_id}")
                    self.vehicle_assignments[vehicle_id] = None
        return set(assignment for assignment in self.vehicle_assignments.values() if assignment is not None)

    def _assign_new_order(self, vehicle_id: int, order_id: int, active_orders: set):
        """Assign a new order to a vehicle"""
        if order_id not in active_orders:
            self.vehicle_assignments[vehicle_id] = order_id
            active_orders.add(order_id)
            print(f"[DEBUG] New assignment: Vehicle {vehicle_id} -> Order {order_id}")

    # === Movement and Location Methods ===
    def _calculate_movement(self, start_loc, target_loc, progress):
        """Calculate new position and step distance"""
        new_loc = self.location_manager.interpolate_position(start_loc, target_loc, progress)
        step_distance = self.location_manager.get_travel_time(start_loc, new_loc)
        return new_loc, step_distance

    def _update_phase_progress(self, phase):
        """Update and return movement progress"""
        phase["time_spent"] += 1
        return min(1.0, phase["time_spent"] / phase["total_time"])

    # === Service Time Methods ===
    def _initialize_service(self, phase):
        """Initialize service at location"""
        phase["service_time_remaining"] = self.service_time
        phase["is_servicing"] = True
        return phase

    def _process_service_time(self, phase):
        """Process service time and return if complete"""
        phase["service_time_remaining"] -= 1.0
        return phase["service_time_remaining"] <= 0

    # === Phase Management Methods ===
    def _initialize_vehicle_phase(self, order_id, order, current_loc):
        """Initialize new phase for vehicle"""
        is_pickup = order.status == "pending"
        target_loc = order.pickup_location if is_pickup else order.delivery_location

        phase = {
            "order_id": order_id,
            "stage": "pickup" if is_pickup else "delivery",
            "total_time": 0,
            "time_spent": 0,
            "start_loc": current_loc,
            "target_loc": target_loc,
            "service_time_remaining": None,
            "is_servicing": False,
        }

        # Initialize total time
        phase["total_time"] = max(0.001, self.location_manager.get_travel_time(phase["start_loc"], phase["target_loc"]))

        return phase

    def _initialize_delivery_phase(self, order_id, current_loc, delivery_loc):
        """Initialize delivery phase after pickup"""
        return {
            "order_id": order_id,
            "stage": "delivery",
            "total_time": max(0.001, self.location_manager.get_travel_time(current_loc, delivery_loc)),
            "time_spent": 0,
            "start_loc": current_loc,
            "target_loc": delivery_loc,
            "service_time_remaining": None,
            "is_servicing": False,
        }

    # === Metrics Methods ===
    def _initialize_metrics(self):
        """Initialize metrics dictionary"""
        return {"distance": 0, "deliveries": 0, "delays": [], "late_orders": set(), "delivered_orders": set()}

    def _update_metrics(self, metrics, route_metrics):
        """Update overall metrics with route metrics"""
        for key in metrics:
            if isinstance(metrics[key], (list, set)):
                if isinstance(metrics[key], list):
                    metrics[key].extend(route_metrics[key])
                else:
                    metrics[key].update(route_metrics[key])
            else:
                metrics[key] += route_metrics[key]

    # === Main Processing Methods (Updated) ===
    def process_all_routes(self, route_plan, vehicle_manager, order_manager, current_time):
        metrics = self._initialize_metrics()
        active_orders = self._cleanup_stale_assignments(order_manager)

        for vehicle_id, current_route in enumerate(route_plan):
            vehicle = vehicle_manager.get_vehicle_by_id(vehicle_id)
            if not vehicle:
                continue

            current_assignment = self.vehicle_assignments.get(vehicle_id)

            if current_assignment is not None:
                route_metrics = self._process_single_route(
                    [current_assignment], vehicle, vehicle.current_location, order_manager, current_time
                )

                if current_assignment in route_metrics["delivered_orders"]:
                    self.vehicle_assignments[vehicle_id] = None
                    active_orders.discard(current_assignment)

                self._update_metrics(metrics, route_metrics)

            elif current_route:
                self._assign_new_order(vehicle_id, current_route[0], active_orders)

        return metrics

    def _process_single_order(self, order_id, vehicle, current_loc, order_manager, current_time):
        """Process individual order (kept largely the same but using new helper methods)"""
        # Initialize variables
        new_loc = current_loc
        step_distance = 0.0
        delay = 0.0
        completed = False
        orders_to_keep = []

        # Get order
        order = next((o for o in order_manager.active_orders if o.id == order_id), None)
        if not order:
            return current_loc, step_distance, delay, completed, orders_to_keep

        # Initialize phase if needed
        if not hasattr(vehicle, "current_phase") or vehicle.current_phase is None:
            vehicle.current_phase = self._initialize_vehicle_phase(order_id, order, current_loc)

        # Handle service time
        if vehicle.current_phase["is_servicing"]:
            if self._process_service_time(vehicle.current_phase):
                return self._handle_service_completion(vehicle, order, current_loc, current_time)
            return current_loc, 0.0, 0.0, False, [order_id]

        # Handle movement
        progress = self._update_phase_progress(vehicle.current_phase)
        new_loc, step_distance = self._calculate_movement(
            vehicle.current_phase["start_loc"], vehicle.current_phase["target_loc"], progress
        )

        # Handle arrival
        if progress >= 1.0:
            return self._handle_arrival(vehicle, order, new_loc, current_time, order_id)

        orders_to_keep = [order_id] if not completed else []
        return new_loc, step_distance, delay, completed, orders_to_keep

    # Keep _process_single_route as is since it's mainly coordination
    def _process_single_route(
        self,
        route: List[int],
        vehicle: Vehicle,
        current_loc: Location,
        order_manager,
        current_time: float,
    ) -> Dict[str, Any]:
        """Process single vehicle route and collect metrics."""
        metrics = {"distance": 0, "deliveries": 0, "delays": [], "late_orders": set(), "delivered_orders": set()}

        print(f"\n[DEBUG] ---- Begin _process_single_route ----")
        print(f"[DEBUG] Vehicle {vehicle.id} processing route: {route}")
        print(f"[DEBUG] Vehicle current phase: {vehicle.current_phase}")

        if not route:
            return metrics

        order_id = route[0]
        # For debugging
        order = next((o for o in order_manager.active_orders if o.id == order_id), None)
        print(f"[DEBUG] Processing order {order_id}, current status: {order.status if order else 'None'}")

        result = self._process_single_order(
            order_id=order_id,
            vehicle=vehicle,
            current_loc=current_loc,
            order_manager=order_manager,
            current_time=current_time,
        )
        print(f"[DEBUG] Finished processing order {order_id}, new status: {order.status if order else 'None'}")

        new_loc, distance, delay, completed, orders_to_keep = result

        # Update metrics
        metrics["distance"] += distance
        if delay > 0:
            metrics["delays"].append(delay)
            metrics["late_orders"].add(order_id)
        if completed:
            metrics["deliveries"] += 1
            metrics["delivered_orders"].add(order_id)

        # Update vehicle location
        vehicle.current_location = new_loc

        return metrics

    def _handle_pickup(self, order, order_id, vehicle, current_loc, current_time):
        """Handle pickup stage of order processing."""
        movement = order.travel_progress["pickup"]
        target_loc = order.pickup_location
        orders_to_keep = []

        # Update movement progress
        movement["time_spent"] += 1
        progress = min(1.0, movement["time_spent"] / movement["total_time"])

        # Handle movement to restaurant
        if progress < 1.0:
            new_loc = self.location_manager.interpolate_position(movement["start_loc"], target_loc, progress)
            step_distance = self.location_manager.get_travel_time(current_loc, new_loc)
            orders_to_keep.append(order_id)
            return new_loc, step_distance, 0.0, False, orders_to_keep

        # Handle restaurant arrival
        if not order.ready_time or current_time >= order.ready_time:
            return self._handle_restaurant_arrival(order, order_id, vehicle, target_loc, current_time)

        orders_to_keep.append(order_id)
        return target_loc, 0.0, 0.0, False, orders_to_keep

    def _handle_delivery(self, order, order_id, vehicle, current_loc, current_time):
        """Handle delivery stage of order processing."""
        movement = order.travel_progress["delivery"]
        target_loc = order.delivery_location
        orders_to_keep = []

        # Initialize delivery if needed
        if movement["time_spent"] == 0:
            movement["start_loc"] = current_loc
            movement["total_time"] = self.location_manager.get_travel_time(current_loc, target_loc)

        # Update movement progress
        movement["time_spent"] += 1
        progress = min(1.0, movement["time_spent"] / movement["total_time"])

        # Handle movement to customer
        if progress < 1.0:
            new_loc = self.location_manager.interpolate_position(movement["start_loc"], target_loc, progress)
            step_distance = self.location_manager.get_travel_time(current_loc, new_loc)
            orders_to_keep.append(order_id)
            return new_loc, step_distance, 0.0, False, orders_to_keep

        # Handle customer arrival
        if not hasattr(order, "service_time_remaining"):
            order.service_time_remaining = self.service_time

        if order.service_time_remaining > 0:
            order.service_time_remaining -= 1
            orders_to_keep.append(order_id)
            return target_loc, 0.0, 0.0, False, orders_to_keep

        # Complete delivery
        order.status = "delivered"
        order.delivery_time = current_time
        delay = max(0, current_time - order.deadline)
        return target_loc, 0.0, delay, True, []

    def _handle_restaurant_arrival(self, order, order_id, vehicle, target_loc, current_time):
        """Handle arrival at restaurant."""
        if not hasattr(order, "service_time_remaining"):
            order.service_time_remaining = self.service_time

        if order.service_time_remaining > 0:
            order.service_time_remaining -= 1
            return target_loc, 0.0, 0.0, False, [order_id]

        # Complete pickup
        order.status = "picked_up"
        order.pickup_time = current_time
        order.service_time_remaining = self.service_time
        order.travel_progress["delivery"].update(
            {
                "total_time": self.location_manager.get_travel_time(vehicle.current_location, order.delivery_location),
                "start_loc": vehicle.current_location,
            }
        )
        return target_loc, 0.0, 0.0, False, [order_id]

    def _handle_arrival(self, vehicle, order, new_loc, current_time, order_id):
        """Handle vehicle arrival at destination."""
        if vehicle.current_phase["stage"] == "pickup":
            if not order.ready_time or current_time >= order.ready_time:
                print(f"[Vehicle {vehicle.id}] Starting pickup service at t={current_time}")
                vehicle.current_phase = self._initialize_service(vehicle.current_phase)
            else:
                print(f"[Order {order.id}] Waiting at restaurant - food not ready yet")
                vehicle.current_phase["time_spent"] = vehicle.current_phase["total_time"]
        else:  # arrived for delivery
            print(f"[Vehicle {vehicle.id}] Starting delivery service at t={current_time}")
            vehicle.current_phase = self._initialize_service(vehicle.current_phase)

        return new_loc, self.location_manager.get_travel_time(vehicle.current_location, new_loc), 0.0, False, [order_id]

    def _handle_service_completion(self, vehicle, order, current_loc, current_time):
        """Handle completion of service at pickup or delivery."""
        if vehicle.current_phase["stage"] == "pickup":
            print(f"[Order {order.id}] Pickup service completed at t={current_time}")
            order.status = "picked_up"
            order.pickup_time = current_time

            # Initialize delivery phase
            vehicle.current_phase = self._initialize_delivery_phase(order.id, current_loc, order.delivery_location)
            return current_loc, 0.0, 0.0, False, [order.id]
        else:  # delivery service completed
            print(f"[Order {order.id}] Delivery service completed at t={current_time}")
            order.status = "delivered"
            order.delivery_time = current_time
            delay = max(0, current_time - order.deadline)
            if delay > 0:
                print(f"[Order {order.id}] Delivery delayed by {delay:.1f} minutes")
            vehicle.current_phase = None
            return current_loc, 0.0, delay, True, []
