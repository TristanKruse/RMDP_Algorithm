from typing import List, Tuple, Set
import numpy as np
from datatypes import State, Location


class FastestVehicleSolver:
    """
    Simple solver that assigns each order to the nearest available vehicle.
    No postponement, no bundling - just straight pickup and delivery.
    """

    def __init__(self, movement_per_step: float):
        """
        Initialize solver with movement speed.

        Args:
            movement_per_step: Distance that can be covered per minute in km
        """
        self.movement_per_step = movement_per_step

    def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate travel time between locations using movement_per_step.

        Returns:
            Travel time in minutes
        """
        dx = loc2.x - loc1.x
        dy = loc2.y - loc1.y
        distance = np.sqrt(dx * dx + dy * dy)

        # Calculate time using movement_per_step (which is in km/minute)
        return distance / self.movement_per_step

    def solve(self, state: State) -> Tuple[List[List[int]], Set[int]]:
        """Create new route plan based on current state."""
        route_plan = [route.copy() for route in state.route_plan]  # Copy current routes

        # Process each unassigned order
        for order in state.unassigned_orders:
            nearest_vehicle_id = self._find_nearest_vehicle(order.pickup_location, state.vehicles, route_plan)

            if nearest_vehicle_id is not None:
                route_plan[nearest_vehicle_id].append(order.id)

        return route_plan, set()

    def _find_nearest_vehicle(self, target_loc: Location, vehicles, route_plan: List[List[int]]) -> int:
        """Find nearest available vehicle."""
        min_travel_time = float("inf")
        best_vehicle_id = None

        for vehicle in vehicles:
            # Skip if vehicle already has an order
            if route_plan[vehicle.id]:
                continue

            travel_time = self._calculate_travel_time(vehicle.current_location, target_loc)
            if travel_time < min_travel_time:
                min_travel_time = travel_time
                best_vehicle_id = vehicle.id

        return best_vehicle_id
