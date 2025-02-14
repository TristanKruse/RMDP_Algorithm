from typing import Dict, Tuple, Set, Optional
import numpy as np
from datatypes import Location, Route, Node
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(module)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger instance
logger = logging.getLogger(__name__)


class FastestVehicleSolver:
    def __init__(self, movement_per_step: float, location_manager):
        """Initialize solver with movement speed and location manager."""
        self.movement_per_step = movement_per_step
        self.location_manager = location_manager

    def solve(self, state: dict) -> Tuple[Dict[int, Route], Set[int]]:
        """Assigns only unassigned orders to nearest available vehicle."""
        # 1. Create copy of current route plan
        route_plan = {vehicle_id: route.copy() for vehicle_id, route in state["route_plan"].items()}

        # 2. Assign each unassigned order to nearest available vehicle
        for order_id, order_info in state["unassigned_orders"].items():
            # Find nearest feasible vehicle
            nearest_vehicle_id = self._find_nearest_vehicle(
                order_info["pickup_node_id"], state["vehicle_positions"], route_plan
            )

            if nearest_vehicle_id is not None:
                # Get existing sequence or empty list
                current_sequence = route_plan[nearest_vehicle_id].sequence
                # Add new pickup and delivery to sequence
                new_sequence = current_sequence + [
                    (order_info["pickup_node_id"].id, {order_id}, set()),  # Pickup
                    (order_info["delivery_node_id"].id, set(), {order_id}),  # Delivery
                ]
                # Update route with new sequence
                route_plan[nearest_vehicle_id] = Route(
                    vehicle_id=nearest_vehicle_id,
                    sequence=new_sequence,
                    total_distance=0.0,
                    total_time=0.0,
                )

        return route_plan, set()

    def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
        """Calculate travel time between two locations based on movement speed."""
        dx, dy = loc2.x - loc1.x, loc2.y - loc1.y
        return np.sqrt(dx * dx + dy * dy) / self.movement_per_step

    def _find_nearest_vehicle(
        self, target_node: Node, vehicle_positions: Dict[int, Location], route_plan: Dict[int, Route]
    ) -> Optional[int]:
        # 1. Initialize tracking variables
        min_travel_time = float("inf")
        best_vehicle_id = None

        # 2. Get target location directly from Node
        target_loc = target_node.location

        # 3. Check each vehicle
        for vehicle_id, vehicle_loc in vehicle_positions.items():
            # Skip busy vehicles - doesn't allow for rerouting
            if route_plan[vehicle_id].sequence:  # Check sequence instead of route
                continue
            # 4. Calculate travel time to target
            travel_time = self._calculate_travel_time(vehicle_loc, target_loc)
            # 5. Update if this is fastest vehicle so far
            if travel_time < min_travel_time:
                min_travel_time = travel_time
                best_vehicle_id = vehicle_id

        return best_vehicle_id
