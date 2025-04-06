# bundling, when multiple orders at restaurant,
# bundling, when another order is close to the restaurant
from typing import Tuple, Set, Dict
import numpy as np
from datatypes import Location, Route, Optional, Node


class FastestBundler:
    def __init__(self, movement_per_step: float, location_manager, max_bundle_size: int):
        """Initialize solver with movement speed and location manager."""
        self.movement_per_step = movement_per_step
        self.location_manager = location_manager
        self.max_bundle_size = max_bundle_size

    def solve(self, state: dict) -> Tuple[Dict[int, Route], Set[int]]:
        """Assigns orders to nearest available vehicle with bundling."""
        import logging

        logger = logging.getLogger(__name__)

        logger.info(f"Current route_plan, before solving{state['route_plan']}")

        # 1. Create copy of current route plan
        route_plan = {vehicle_id: route.copy() for vehicle_id, route in state["route_plan"].items()}

        # Group orders by restaurant
        restaurant_orders = {}
        for order_id, order_info in state["unassigned_orders"].items():
            rest_id = order_info["pickup_node_id"].id
            if rest_id not in restaurant_orders:
                restaurant_orders[rest_id] = []
            restaurant_orders[rest_id].append((order_id, order_info))

        # Process each order
        for restaurant_id, orders in restaurant_orders.items():
            for order_id, order_info in orders:
                # Find suitable vehicle (either empty or already going to this restaurant)
                vehicle_id = self._find_nearest_vehicle(
                    order_info["pickup_node_id"], state["vehicle_positions"], route_plan
                )

                if vehicle_id is not None:
                    current_sequence = route_plan[vehicle_id].sequence

                    # Case 1: Empty vehicle - start new route
                    if not current_sequence:
                        route_plan[vehicle_id] = Route(
                            vehicle_id=vehicle_id,
                            sequence=[
                                (restaurant_id, {order_id}, set()),  # Pickup
                                (order_info["delivery_node_id"].id, set(), {order_id}),  # Delivery
                            ],
                            total_distance=0.0,
                            total_time=0.0,
                        )

                    # Case 2: Add to existing route at same restaurant
                    else:
                        first_stop = current_sequence[0]
                        if first_stop[0] == restaurant_id:
                            # Update pickup set
                            pickups = first_stop[1] | {order_id}
                            new_sequence = [(first_stop[0], pickups, set())]

                            # Add all deliveries after pickups
                            all_deliveries = []
                            # Add existing deliveries
                            for node_id, _, deliveries in current_sequence[1:]:
                                if deliveries:
                                    all_deliveries.append((node_id, set(), deliveries))
                            # Add new delivery
                            all_deliveries.append((order_info["delivery_node_id"].id, set(), {order_id}))
                            new_sequence.extend(all_deliveries)

                            route_plan[vehicle_id] = Route(
                                vehicle_id=vehicle_id, sequence=new_sequence, total_distance=0.0, total_time=0.0
                            )

        logger.info(f"Current route_plan, after solving{route_plan}")
        return route_plan, set()

    def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
        """Calculate travel time between two locations based on movement speed."""
        dx, dy = loc2.x - loc1.x, loc2.y - loc1.y
        return np.sqrt(dx * dx + dy * dy) / self.movement_per_step

    def _find_nearest_vehicle(
        self, restaurant_node: Node, vehicle_positions: Dict[int, Location], route_plan: Dict[int, Route]
    ) -> Optional[int]:
        """Find vehicle that can add this order to its bundle, or nearest empty vehicle."""
        min_travel_time = float("inf")
        best_vehicle_id = None

        for vehicle_id, vehicle_loc in vehicle_positions.items():
            current_route = route_plan[vehicle_id].sequence

            # Case 1: Empty vehicle
            if not current_route:
                travel_time = self._calculate_travel_time(vehicle_loc, restaurant_node.location)
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    best_vehicle_id = vehicle_id

            # Case 2: Vehicle with existing route at same restaurant
            elif current_route and len(current_route) >= 1:
                first_stop = current_route[0]
                # If first stop is this restaurant and bundle not full (< 3 orders)
                if first_stop[0] == restaurant_node.id and len(first_stop[1]) < 3:
                    # Prioritize vehicles already going to this restaurant
                    return vehicle_id

        return best_vehicle_id
