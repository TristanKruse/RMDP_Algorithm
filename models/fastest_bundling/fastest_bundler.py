# bundling, when multiple orders at restaurant,
# bundling, when another order is close to the restaurant
from typing import List, Tuple, Set, Dict
import numpy as np
from datatypes import State, Location, Order, Route, Optional, Node


class FastestBundler:
    def __init__(self, movement_per_step: float, location_manager, max_bundle_size: int):
        """Initialize solver with movement speed and location manager."""
        self.movement_per_step = movement_per_step
        self.location_manager = location_manager
        self.max_bundle_size = max_bundle_size

    # def solve(self, state: dict) -> Tuple[Dict[int, Route], Set[int]]:
    #     """Assigns only unassigned orders to nearest available vehicle."""
    #     # 1. Create copy of current route plan
    #     route_plan = {vehicle_id: route.copy() for vehicle_id, route in state["route_plan"].items()}

    #     # 2. Assign each unassigned order to nearest available vehicle
    #     for order_id, order_info in state["unassigned_orders"].items():
    #         # Find nearest feasible vehicle
    #         nearest_vehicle_id = self._find_nearest_vehicle(
    #             order_info["pickup_node_id"], state["vehicle_positions"], route_plan
    #         )

    #         if nearest_vehicle_id is not None:
    #             # Get existing sequence or empty list
    #             current_sequence = route_plan[nearest_vehicle_id].sequence
    #             # Add new pickup and delivery to sequence
    #             new_sequence = current_sequence + [
    #                 (order_info["pickup_node_id"].id, {order_id}, set()),  # Pickup
    #                 (order_info["delivery_node_id"].id, set(), {order_id}),  # Delivery
    #             ]
    #             # Update route with new sequence
    #             route_plan[nearest_vehicle_id] = Route(
    #                 vehicle_id=nearest_vehicle_id,
    #                 sequence=new_sequence,
    #                 total_distance=0.0,
    #                 total_time=0.0,
    #             )

    #     return route_plan, set()

    # def solve(self, state: dict) -> Tuple[Dict[int, Route], Set[int]]:
    #     """Assigns orders to nearest available vehicle, bundling up to capacity before delivery."""
    #     # 1. Create copy of current route plan
    #     route_plan = {vehicle_id: route.copy() for vehicle_id, route in state["route_plan"].items()}

    #     # Track orders assigned to each vehicle
    #     vehicle_orders = {vid: 0 for vid in route_plan.keys()}

    #     # 2. Process each unassigned order
    #     for order_id, order_info in state["unassigned_orders"].items():
    #         # Find nearest feasible vehicle
    #         nearest_vehicle_id = self._find_nearest_vehicle(
    #             order_info["pickup_node_id"], state["vehicle_positions"], route_plan
    #         )

    #         if nearest_vehicle_id is not None:
    #             current_sequence = route_plan[nearest_vehicle_id].sequence

    #             # If vehicle has no orders yet, start new sequence with pickup
    #             if not current_sequence:
    #                 current_sequence = [(order_info["pickup_node_id"].id, {order_id}, set())]
    #                 vehicle_orders[nearest_vehicle_id] = 1

    #             # If vehicle has orders but not full, add to pickups
    #             elif vehicle_orders[nearest_vehicle_id] < self.max_bundle_size:  # Max capacity of 3
    #                 # Add order to existing pickup set if same restaurant
    #                 if current_sequence[0][0] == order_info["pickup_node_id"].id:
    #                     # Update pickup set at first position
    #                     first_node, pickups, deliveries = current_sequence[0]
    #                     pickups.add(order_id)
    #                     current_sequence[0] = (first_node, pickups, deliveries)
    #                 else:
    #                     # Add new pickup
    #                     current_sequence.insert(0, (order_info["pickup_node_id"].id, {order_id}, set()))
    #                 vehicle_orders[nearest_vehicle_id] += 1

    #             # If this fills the vehicle, add all deliveries
    #             if vehicle_orders[nearest_vehicle_id] == 3:
    #                 # Collect all orders that need delivery
    #                 all_pickups = set()
    #                 for _, pickups, _ in current_sequence:
    #                     all_pickups.update(pickups)

    #                 # Add delivery stops for each order
    #                 for pickup_order_id in all_pickups:
    #                     pickup_info = state["unassigned_orders"][pickup_order_id]
    #                     current_sequence.append((pickup_info["delivery_node_id"].id, set(), {pickup_order_id}))

    #             # Update route with new sequence
    #             route_plan[nearest_vehicle_id] = Route(
    #                 vehicle_id=nearest_vehicle_id,
    #                 sequence=current_sequence,
    #                 total_distance=0.0,
    #                 total_time=0.0,
    #             )

    #     # Add deliveries for any partially filled vehicles
    #     for vehicle_id, num_orders in vehicle_orders.items():
    #         if 0 < num_orders < 3:  # Vehicle has orders but isn't full
    #             current_sequence = route_plan[vehicle_id].sequence
    #             # Collect all orders that need delivery
    #             all_pickups = set()
    #             for _, pickups, _ in current_sequence:
    #                 all_pickups.update(pickups)

    #             # Add delivery stops for each order
    #             delivery_sequence = []
    #             for pickup_order_id in all_pickups:
    #                 pickup_info = state["unassigned_orders"][pickup_order_id]
    #                 delivery_sequence.append((pickup_info["delivery_node_id"].id, set(), {pickup_order_id}))

    #             # Update route with deliveries added
    #             route_plan[vehicle_id] = Route(
    #                 vehicle_id=vehicle_id,
    #                 sequence=current_sequence + delivery_sequence,
    #                 total_distance=0.0,
    #                 total_time=0.0,
    #             )

    #     return route_plan, set()

    # def solve(self, state: dict) -> Tuple[Dict[int, Route], Set[int]]:
    #     """Assigns orders to nearest available vehicle, with detailed logging."""
    #     import logging

    #     logger = logging.getLogger(__name__)

    #     # 1. Create copy of current route plan
    #     route_plan = {vehicle_id: route.copy() for vehicle_id, route in state["route_plan"].items()}
    #     logger.info(f"Current route_plan, before solving{route_plan}")

    #     # Log initial state
    #     logger.info(f"\n{'='*50}\nStarting new solve iteration")
    #     logger.info(f"Unassigned orders: {len(state['unassigned_orders'])}")

    #     # Track orders assigned to each vehicle
    #     vehicle_orders = {vid: 0 for vid in route_plan.keys()}

    #     # Group orders by restaurant for bundling
    #     restaurant_orders = {}
    #     for order_id, order_info in state["unassigned_orders"].items():
    #         rest_id = order_info["pickup_node_id"].id
    #         if rest_id not in restaurant_orders:
    #             restaurant_orders[rest_id] = []
    #         restaurant_orders[rest_id].append((order_id, order_info))

    #     logger.info(f"Orders grouped by restaurant: {dict((k, len(v)) for k, v in restaurant_orders.items())}")

    #     # Process each restaurant's orders
    #     for restaurant_id, orders in restaurant_orders.items():
    #         logger.info(f"\nProcessing restaurant {restaurant_id} with {len(orders)} orders")

    #         # Find nearest vehicle to this restaurant
    #         nearest_vehicle_id = self._find_nearest_vehicle(
    #             orders[0][1]["pickup_node_id"], state["vehicle_positions"], route_plan
    #         )

    #         if nearest_vehicle_id is not None:
    #             logger.info(f"Found nearest vehicle: {nearest_vehicle_id}")

    #             # Build bundle sequence
    #             sequence = []
    #             pickup_orders = set()

    #             # Add all pickups for this restaurant (up to 3)
    #             orders_to_bundle = orders[:3]  # Take up to 3 orders
    #             for order_id, order_info in orders_to_bundle:
    #                 pickup_orders.add(order_id)

    #             # Add single pickup stop for all orders
    #             sequence.append((restaurant_id, pickup_orders, set()))
    #             logger.info(f"Added pickup stop for orders: {pickup_orders}")

    #             # Add all deliveries
    #             for order_id, order_info in orders_to_bundle:
    #                 sequence.append((order_info["delivery_node_id"].id, set(), {order_id}))

    #             # Update route plan
    #             route_plan[nearest_vehicle_id] = Route(
    #                 vehicle_id=nearest_vehicle_id,
    #                 sequence=sequence,
    #                 total_distance=0.0,
    #                 total_time=0.0,
    #             )

    #             logger.info(f"Final sequence for vehicle {nearest_vehicle_id}:")
    #             for stop_idx, (node_id, pickups, deliveries) in enumerate(sequence):
    #                 logger.info(f"  Stop {stop_idx}: Node {node_id}")
    #                 if pickups:
    #                     logger.info(f"    Pickups: {pickups}")
    #                 if deliveries:
    #                     logger.info(f"    Deliveries: {deliveries}")

    #             vehicle_orders[nearest_vehicle_id] = len(orders_to_bundle)

    #     logger.info("\nFinal route plan summary:")
    #     for vehicle_id, route in route_plan.items():
    #         if route.sequence:
    #             logger.info(
    #                 f"Vehicle {vehicle_id}: {len(route.sequence)} stops, " f"{vehicle_orders[vehicle_id]} orders"
    #             )

    #     return route_plan, set()

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

        return route_plan, set()

    def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
        """Calculate travel time between two locations based on movement speed."""
        dx, dy = loc2.x - loc1.x, loc2.y - loc1.y
        return np.sqrt(dx * dx + dy * dy) / self.movement_per_step

    # def _find_nearest_vehicle(
    #     self, target_node: Node, vehicle_positions: Dict[int, Location], route_plan: Dict[int, Route]
    # ) -> Optional[int]:
    #     # 1. Initialize tracking variables
    #     min_travel_time = float("inf")
    #     best_vehicle_id = None

    #     # 2. Get target location directly from Node
    #     target_loc = target_node.location

    #     # 3. Check each vehicle
    #     for vehicle_id, vehicle_loc in vehicle_positions.items():
    #         # Skip busy vehicles - doesn't allow for rerouting
    #         if route_plan[vehicle_id].sequence:  # Check sequence instead of route
    #             continue
    #         # 4. Calculate travel time to target
    #         travel_time = self._calculate_travel_time(vehicle_loc, target_loc)
    #         # 5. Update if this is fastest vehicle so far
    #         if travel_time < min_travel_time:
    #             min_travel_time = travel_time
    #             best_vehicle_id = vehicle_id

    #     return best_vehicle_id

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
