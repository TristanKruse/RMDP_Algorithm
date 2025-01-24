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
        self.vehicle_assignments = {}  # Track current order assignment per vehicle

    def process_all_routes(
        self, route_plan: List[List[int]], vehicle_manager, order_manager, current_time: float
    ) -> Dict[str, Any]:
        metrics = {"distance": 0, "deliveries": 0, "delays": [], "late_orders": set(), "delivered_orders": set()}

        print("\n[DEBUG] ---- Begin process_all_routes ----")
        print(f"[DEBUG] Current route plan: {route_plan}")
        print(f"[DEBUG] Active vehicle assignments: {self.vehicle_assignments}")

        # Clean up stale assignments - check if orders still exist
        for vehicle_id, order_id in list(self.vehicle_assignments.items()):
            if order_id is not None:
                order = next((o for o in order_manager.active_orders if o.id == order_id), None)
                if not order:
                    print(f"[DEBUG] Cleaning up stale assignment: Vehicle {vehicle_id}, Order {order_id}")
                    self.vehicle_assignments[vehicle_id] = None

        # Initialize active_orders from current active assignments
        active_orders = set(assignment for assignment in self.vehicle_assignments.values() if assignment is not None)
        print(f"[DEBUG] Initial active orders after cleanup: {active_orders}")

        # Process each vehicle's route
        for vehicle_id, current_route in enumerate(route_plan):
            print(f"\n[DEBUG] Processing vehicle {vehicle_id}")
            print(f"[DEBUG] Current route: {current_route}")
            current_assignment = self.vehicle_assignments.get(vehicle_id)
            print(f"[DEBUG] Current assignment: {current_assignment}")

            vehicle = vehicle_manager.get_vehicle_by_id(vehicle_id)
            if not vehicle:
                continue

            # Process current assignment if it exists
            if current_assignment is not None:
                print(f"[DEBUG] Processing assigned order {current_assignment} for vehicle {vehicle_id}")
                route_metrics = self._process_single_route(
                    route=[current_assignment],
                    vehicle=vehicle,
                    current_loc=vehicle.current_location,
                    order_manager=order_manager,
                    current_time=current_time,
                )

                # Clear completed orders
                if current_assignment in route_metrics["delivered_orders"]:
                    self.vehicle_assignments[vehicle_id] = None
                    active_orders.discard(current_assignment)
                    print(f"[Vehicle {vehicle_id}] Completed Order {current_assignment}")

                # Update metrics
                for key in metrics:
                    if isinstance(metrics[key], (list, set)):
                        if isinstance(metrics[key], list):
                            metrics[key].extend(route_metrics[key])
                        else:
                            metrics[key].update(route_metrics[key])
                    else:
                        metrics[key] += route_metrics[key]

            # Try to assign new order if vehicle is free
            elif current_route:
                order_id = current_route[0]
                if order_id not in active_orders:
                    self.vehicle_assignments[vehicle_id] = order_id
                    active_orders.add(order_id)
                    print(f"[DEBUG] New assignment: Vehicle {vehicle_id} -> Order {order_id}")

        print(f"[DEBUG] ---- End process_all_routes ----")
        print(f"[DEBUG] Final vehicle assignments: {self.vehicle_assignments}")
        print(f"[DEBUG] Final active orders: {active_orders}")
        return metrics

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

    def _process_single_order(
        self,
        order_id: int,
        vehicle: Vehicle,
        current_loc: Location,
        order_manager,
        current_time: float,
    ) -> Tuple[Location, float, float, bool, List[int]]:
        """Process individual order pickup/delivery."""
        print(f"\n[DEBUG] ---- Begin _process_single_order ----")
        print(f"[DEBUG] Vehicle {vehicle.id} processing order {order_id}")
        print(f"[DEBUG] Vehicle current phase: {vehicle.current_phase}")

        # Initialize return variables
        new_loc = current_loc
        step_distance = 0.0
        delay = 0.0
        completed = False
        orders_to_keep = []

        # Get order
        order = next((o for o in order_manager.active_orders if o.id == order_id), None)
        if order:
            print(f"[DEBUG] Order {order_id} status: {order.status}")
        else:
            print(f"[DEBUG] Order {order_id} not found in active orders")

        if not order:
            return current_loc, step_distance, delay, completed, orders_to_keep

        # Initialize phase tracking if needed
        if not hasattr(vehicle, "current_phase") or vehicle.current_phase is None:
            vehicle.current_phase = {
                "order_id": order_id,
                "stage": "pickup" if order.status == "pending" else "delivery",
                "total_time": 0,
                "time_spent": 0,
                "start_loc": current_loc,
                "target_loc": order.pickup_location if order.status == "pending" else order.delivery_location,
                "service_time_remaining": None,  # Changed to None for clearer state tracking
                "is_servicing": False,
            }
            # Initialize total time for current phase
            vehicle.current_phase["total_time"] = max(
                0.001,
                self.location_manager.get_travel_time(
                    vehicle.current_phase["start_loc"], vehicle.current_phase["target_loc"]
                ),
            )
            print(f"\n[Vehicle {vehicle.id}] Starting {vehicle.current_phase['stage']} phase for Order {order_id}")

        # Handle service time if we're servicing
        if vehicle.current_phase["is_servicing"]:
            # Decrement remaining service time
            vehicle.current_phase["service_time_remaining"] -= 1.0
            print(
                f"[Vehicle {vehicle.id}] Servicing at {vehicle.current_phase['stage']} location. "
                f"Remaining service time: {vehicle.current_phase['service_time_remaining']}"
            )

            # Check if service is complete
            if vehicle.current_phase["service_time_remaining"] <= 0:
                if vehicle.current_phase["stage"] == "pickup":
                    print(f"[Order {order.id}] Pickup service completed at t={current_time}")
                    order.status = "picked_up"
                    order.pickup_time = current_time

                    # Initialize delivery phase
                    vehicle.current_phase = {
                        "order_id": order_id,
                        "stage": "delivery",
                        "total_time": max(
                            0.001, self.location_manager.get_travel_time(current_loc, order.delivery_location)
                        ),
                        "time_spent": 0,
                        "start_loc": current_loc,
                        "target_loc": order.delivery_location,
                        "service_time_remaining": None,
                        "is_servicing": False,
                    }
                else:  # delivery service completed
                    print(f"[Order {order.id}] Delivery service completed at t={current_time}")
                    order.status = "delivered"
                    order.delivery_time = current_time
                    delay = max(0, current_time - order.deadline)
                    if delay > 0:
                        print(f"[Order {order.id}] Delivery delayed by {delay:.1f} minutes")
                    completed = True
                    vehicle.current_phase = None
                    return current_loc, 0.0, delay, completed, []

            return current_loc, 0.0, 0.0, False, [order_id]

        # Regular movement processing
        vehicle.current_phase["time_spent"] += 1
        progress = min(1.0, vehicle.current_phase["time_spent"] / vehicle.current_phase["total_time"])

        # Calculate new position
        new_loc = self.location_manager.interpolate_position(
            vehicle.current_phase["start_loc"], vehicle.current_phase["target_loc"], progress
        )

        # Calculate step distance
        step_distance = self.location_manager.get_travel_time(current_loc, new_loc)

        print(f"\n[Vehicle {vehicle.id}] At t={current_time:.1f}")
        print(f"[Vehicle {vehicle.id}] Current location: ({current_loc.x:.2f}, {current_loc.y:.2f})")
        print(f"[Vehicle {vehicle.id}] Processing Order {order_id}")
        print(f"[Vehicle {vehicle.id}] Phase: {vehicle.current_phase['stage']}")
        print(f"[Vehicle {vehicle.id}] Progress: {progress:.2f}")

        # Handle arrival at destination
        if progress >= 1.0:
            if vehicle.current_phase["stage"] == "pickup":
                if not order.ready_time or current_time >= order.ready_time:
                    print(f"[Vehicle {vehicle.id}] Starting pickup service at t={current_time}")
                    vehicle.current_phase["service_time_remaining"] = self.service_time
                    vehicle.current_phase["is_servicing"] = True
                else:
                    print(f"[Order {order.id}] Waiting at restaurant - food not ready yet")
                    vehicle.current_phase["time_spent"] = vehicle.current_phase["total_time"]
            else:  # arrived for delivery
                print(f"[Vehicle {vehicle.id}] Starting delivery service at t={current_time}")
                vehicle.current_phase["service_time_remaining"] = self.service_time
                vehicle.current_phase["is_servicing"] = True
            return new_loc, step_distance, 0.0, False, [order_id]

        orders_to_keep = [order_id] if not completed else []
        return new_loc, step_distance, delay, completed, orders_to_keep
