from typing import List, Tuple, Set
import numpy as np
from datatypes import State, Location


class FastestVehicleSolver:
    def __init__(self, movement_per_step: float):
        self.movement_per_step = movement_per_step

    def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
        dx, dy = loc2.x - loc1.x, loc2.y - loc1.y
        return np.sqrt(dx * dx + dy * dy) / self.movement_per_step

    def solve(self, state: State) -> Tuple[List[List[int]], Set[int]]:
        # 1. Create copy of current route plan
        route_plan = [route.copy() for route in state.route_plan]

        # 2. Assign each unassigned order to nearest available vehicle
        for order in state.unassigned_orders:
            nearest_vehicle_id = self._find_nearest_vehicle(order.pickup_location, state.vehicles, route_plan)
            if nearest_vehicle_id is not None:
                route_plan[nearest_vehicle_id].append(order.id)

        # 3. Return updated route plan (no orders postponed)
        return route_plan, set()

    def _find_nearest_vehicle(self, target_loc: Location, vehicles, route_plan: List[List[int]]) -> int:
        # 1. Initialize tracking variables
        min_travel_time = float("inf")
        best_vehicle_id = None

        # 2. Check each vehicle
        for vehicle in vehicles:
            # Skip busy vehicles
            if route_plan[vehicle.id]:
                continue
            # 3. Calculate travel time to target
            travel_time = self._calculate_travel_time(vehicle.current_location, target_loc)
            # 4. Update if this is fastest vehicle so far
            if travel_time < min_travel_time:
                min_travel_time = travel_time
                best_vehicle_id = vehicle.id

        return best_vehicle_id
