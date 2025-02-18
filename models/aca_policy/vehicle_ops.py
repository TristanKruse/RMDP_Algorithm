from typing import List, Optional
from dataclasses import dataclass
from datatypes import State, Order, Route
from .assign_order import RouteAssigner
from .time_utils import TimeCalculator


@dataclass
class VehicleAssignment:
    """Container for vehicle assignment results"""

    vehicle_id: int
    tentative_route: List[int]
    delay: float


class VehicleOperations:
    def __init__(
        self,
        service_time: float,
        vehicle_capacity: int,
        mean_prep_time: float,
        prep_time_var: float,
        delay_normalization_factor: float,
        movement_per_step: float,
        location_manager,
    ):
        self.route_assigner = RouteAssigner(
            service_time=service_time,
            mean_prep_time=mean_prep_time,
            delay_normalization_factor=delay_normalization_factor,
            movement_per_step=movement_per_step,
            location_manager=location_manager 
        )
        self.time_calculator = TimeCalculator(
            mean_prep_time=mean_prep_time,
            prep_time_var=prep_time_var,
            service_time=service_time,
            delay_normalization_factor=delay_normalization_factor,
            location_manager=location_manager
        )
        self.service_time = service_time
        self.vehicle_capacity = vehicle_capacity
        self.movement_per_step = movement_per_step
        self.location_manager = location_manager 

    def find_vehicle(self, route_plan: dict, order_id: int, buffer: float, state: dict) -> Optional[VehicleAssignment]:
        """Find best vehicle to assign order to based on minimal delay increase"""
        best_assignment = None
        min_delay = float("inf")

        # Get order info
        order_info = state["unassigned_orders"].get(order_id)
        if not order_info:
            print(f"Warning: No info found for order {order_id}")
            return None

        # Try assigning to each vehicle
        for vehicle_id, route in route_plan.items():
            # Count active orders in current sequence
            active_orders = 0
            if route.sequence:
                for _, pickups, deliveries in route.sequence:
                    active_orders += len(pickups)

            # Skip if vehicle already at capacity
            if active_orders >= self.vehicle_capacity:
                continue

            # Create test route
            test_route = route_plan.copy()
            
            try:
                # Create new pickup and delivery stops
                new_sequence = list(route.sequence) if route.sequence else []
                # Add pickup at restaurant node
                new_sequence.append(
                    (order_info["pickup_node_id"].id, {order_id}, set())
                )
                # Add delivery at customer node
                new_sequence.append(
                    (order_info["delivery_node_id"].id, set(), {order_id})
                )
                
                test_route[vehicle_id] = Route(
                    vehicle_id=vehicle_id,
                    sequence=new_sequence,
                    total_distance=0.0,
                    total_time=0.0
                )

                # Calculate delays for this assignment
                delay, _ = self.time_calculator._calculate_delay(
                    state=state,
                    route_plan=test_route,
                    buffer=buffer,
                    vehicle_id=vehicle_id,
                )

                # Update best vehicle if delay is less
                if delay < min_delay:
                    min_delay = delay
                    best_assignment = VehicleAssignment(
                        vehicle_id=vehicle_id,
                        tentative_route=new_sequence,
                        delay=delay
                    )

            except Exception as e:
                print(f"Error calculating delay for vehicle {vehicle_id}: {str(e)}")
                continue

        return best_assignment
