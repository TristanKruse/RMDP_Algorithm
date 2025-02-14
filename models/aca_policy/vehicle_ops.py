from typing import List, Optional
from dataclasses import dataclass
from datatypes import State, Order
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

    def find_vehicle(
        self, route_plan: List[List[int]], order: Order, buffer: float, state: State
    ) -> Optional[VehicleAssignment]:
        """Find best vehicle to assign order to based on minimal delay increase"""
        best_assignment = None
        min_delay = float("inf")

        # Try assigning to each vehicle
        for vehicle_id, route in enumerate(route_plan):
            # Count currently active orders (picked up but not delivered)
            active_orders = 0
            for order_id in route:
                current_order = next((o for o in state.orders if o.id == order_id), None)
                if current_order and current_order.status == "picked_up":
                    active_orders += 1

            # Skip if vehicle already has maximum active orders
            if active_orders >= self.vehicle_capacity:
                continue

            # Create test route with new order
            test_route = route_plan.copy()

            try:
                # Add order to route
                new_route = route.copy()
                new_route.append(order.id)
                test_route[vehicle_id] = new_route

                # Calculate delays using specific vehicle
                delay, _ = self.time_calculator._calculate_delay(
                    state=state,
                    route_plan=test_route,
                    buffer=buffer,
                    vehicle_id=vehicle_id,
                )

                # Update best vehicle if delay is less
                if delay < min_delay:
                    min_delay = delay
                    best_assignment = VehicleAssignment(vehicle_id=vehicle_id, tentative_route=new_route, delay=delay)

            except Exception as e:
                print(f"Error calculating delay for vehicle {vehicle_id}: {str(e)}")
                continue

        if best_assignment is None:
            print("WARNING: Could not find suitable vehicle!")
            return None  # Let the solver handle this case, possibly through postponement

        return best_assignment
