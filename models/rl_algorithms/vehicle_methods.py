# Mischung aus paper von Ulmer et al. und Hilenbrand et al.
from typing import Optional, List
import numpy as np
from datatypes import Order, State, Location


############

# von Claude erstellt, basierend auf beiden Papern


# vehicle_methods.py
class VehicleMethods:
    def __init__(
        self,
        vehicle_speed: float = 40.0,
        mean_prep_time: float = 10.0,
        prep_time_var: float = 2.0,
        service_time: float = 2.0,
        street_network_factor: float = 1.4,
        md_route_convenience_weight: float = 0.4,
        md_overtime_risk_weight: float = 0.4,
        md_courier_willingness_weight: float = 0.2,
        vehicle_capacity: int = 3,
    ):
        self.vehicle_speed = vehicle_speed
        self.mean_prep_time = mean_prep_time
        self.prep_time_var = prep_time_var
        self.service_time = service_time
        self.street_network_factor = street_network_factor
        self.md_weights = (md_route_convenience_weight, md_overtime_risk_weight, md_courier_willingness_weight)
        self.vehicle_capacity = vehicle_capacity

    def _find_best_vehicle(
        self, route_plan: List[List[int]], order: Order, buffer: float, state: State
    ) -> Optional[int]:
        """
        Find vehicle that maximizes the Matching Degree (MD) score for this order.

        The score evaluates:
        1. Route convenience: Based on added travel time/distance for the order
        2. Overtime risk: Based on potential deadline violations with buffer
        3. Courier willingness: Based on workload balance

        These components reflect the paper's emphasis on efficient routes, meeting
        deadlines, and balanced workload distribution.
        """
        best_md_score = float("-inf")
        best_vehicle = None

        for vehicle_id, route in enumerate(route_plan):
            # Basic feasibility check
            if not self._is_feasible_assignment(route, order):
                continue

            # 1. Route convenience
            # Calculate added travel time/distance when inserting order
            current_route_time = self._calculate_total_route_time(route, state)
            tentative_route = self._insert_order_optimally(route, order, state)
            new_route_time = self._calculate_total_route_time(tentative_route, state)

            # Normalize to [0,1] - higher is better
            detour = new_route_time - current_route_time
            route_convenience = 1 / (1 + detour)  # Asymptotic to 1 as detour approaches 0

            # 2. Overtime risk
            # Consider both current orders' deadlines and new order's deadline
            risk = self._calculate_deadline_violation_risk(
                tentative_route, buffer, self.mean_prep_time, self.prep_time_var, state
            )
            overtime_score = 1 - risk  # Higher score = lower risk

            # 3. Courier willingness
            # Balance workload across fleet
            current_workload = len(route)
            avg_workload = sum(len(r) for r in route_plan) / len(route_plan)
            willingness = 1 / (1 + max(0, current_workload - avg_workload))

            # Combine scores - equal weights by default but could be parameterized
            md_score = route_convenience * overtime_score * willingness

            if md_score > best_md_score:
                best_md_score = md_score
                best_vehicle = vehicle_id

        return best_vehicle

    def _calculate_deadline_violation_risk(
        self, route: List[int], buffer: float, mean_prep_time: float, prep_time_var: float, state: State
    ) -> float:
        """Calculate risk of deadline violations"""
        risk = 0
        current_time = 0
        prev_location = None

        for order_id in route:
            order = next(o for o in state.orders if o.id == order_id)

            # Rest of the logic remains the same
            if prev_location:
                current_time += self._calculate_travel_time(prev_location, order.pickup_location)

            prep_time_buffer = mean_prep_time + 2 * np.sqrt(prep_time_var)
            current_time += prep_time_buffer

            current_time += order.service_time
            current_time += self._calculate_travel_time(order.pickup_location, order.delivery_location)
            current_time += order.service_time

            if current_time + buffer > order.deadline:
                risk += (current_time + buffer - order.deadline) / buffer

            prev_location = order.delivery_location

        return min(1, risk / len(route))

    def _insert_order_optimally(self, route: List[int], order: Order, state: State) -> List[int]:
        best_route = None
        best_cost = float("inf")

        for i in range(len(route) + 1):
            new_route = route.copy()
            new_route.insert(i, order.id)
            route_cost = self._calculate_total_route_time(new_route, state)

            if route_cost < best_cost:
                best_cost = route_cost
                best_route = new_route

        return best_route

    def _calculate_route_times(self, route: List[int], state: State) -> dict:
        """Calculate arrival times for each stop in the route."""
        times = {}
        current_time = 0
        current_location = None  # Will be updated based on first location/vehicle position

        for i, order_id in enumerate(route):
            order = next(o for o in state.orders if o.id == order_id)

            # Add travel time to restaurant
            if current_location is None:
                # For first stop, assume starting from restaurant/pickup location
                current_location = order.pickup_location
            else:
                # Calculate travel from previous location
                current_time += self._calculate_travel_time(current_location, order.pickup_location)
                current_location = order.pickup_location

            # Add preparation and service times
            if order.ready_time is None:
                current_time += self.mean_prep_time
            else:
                current_time = max(current_time, order.ready_time)

            current_time += order.service_time

            # Add travel time to delivery
            current_time += self._calculate_travel_time(order.pickup_location, order.delivery_location)
            current_location = order.delivery_location

            times[order_id] = current_time
            current_time += order.service_time

        return times

    def _calculate_total_route_time(self, route: List[int], state: State) -> float:
        times = self._calculate_route_times(route, state)
        return max(times.values()) if times else 0.0

    # vehicle_methods.py
    def _is_feasible_assignment(self, route: List[int], order: Order) -> bool:
        """Check if adding order to route is feasible considering capacity"""
        return len(route) < self.vehicle_capacity

    def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
        """Calculate travel time between two locations."""
        dx = loc2.x - loc1.x
        dy = loc2.y - loc1.y
        distance = self.street_network_factor * np.sqrt(dx * dx + dy * dy)
        return distance / self.vehicle_speed * 60  # Convert to minutes
