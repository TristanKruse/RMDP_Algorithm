from typing import List, Tuple, Set, Dict, Optional
import numpy as np
from datatypes import State, Location, Order


# bundling, when multiple orders at restaurant,
# bundling, when another order is close to the restaurant


class FastestBundler:
    def __init__(
        self, movement_per_step: float, max_bundle_size: int = 3, max_restaurant_distance: float = 2.0
    ):  # 2km max distance between restaurants
        self.movement_per_step = movement_per_step
        self.max_bundle_size = max_bundle_size
        self.max_restaurant_distance = max_restaurant_distance

    def solve(self, state: State) -> Tuple[List[List[int]], Set[int]]:
        # Initialize empty route plan
        route_plan = [[] for _ in range(len(state.vehicles))]

        # Group orders by restaurant
        restaurant_orders = self._group_orders_by_restaurant(state.unassigned_orders)

        # First pass: Bundle orders from same restaurant
        processed_orders = set()
        for restaurant_loc, orders in restaurant_orders.items():
            if len(orders) > 1:  # Multiple orders from same restaurant
                bundle = self._create_restaurant_bundle(orders, state.vehicles, route_plan)
                if bundle:
                    vehicle_id, order_ids = bundle
                    route_plan[vehicle_id].extend(order_ids)
                    processed_orders.update(order_ids)

        # Second pass: Try to bundle remaining orders with nearby restaurants
        remaining_orders = [order for order in state.unassigned_orders if order.id not in processed_orders]

        for order in remaining_orders:
            self._process_single_order(order, state.vehicles, route_plan, state.unassigned_orders)

        return route_plan, set()

    def _group_orders_by_restaurant(self, orders: List[Order]) -> Dict[Tuple[float, float], List[Order]]:
        """Group orders by restaurant location."""
        grouped = {}
        for order in orders:
            loc_key = (order.pickup_location.x, order.pickup_location.y)
            if loc_key not in grouped:
                grouped[loc_key] = []
            grouped[loc_key].append(order)
        return grouped

    def _create_restaurant_bundle(self, orders: List[Order], vehicles, route_plan) -> Tuple[int, List[int]]:
        """Create bundle for orders from same restaurant."""
        if not orders:
            return None

        # Find nearest available vehicle
        nearest_vehicle = self._find_nearest_vehicle(orders[0].pickup_location, vehicles, route_plan, len(orders))

        if nearest_vehicle is None:
            return None

        # Create bundle of orders up to max size
        bundle_size = min(len(orders), self.max_bundle_size)
        selected_orders = orders[:bundle_size]

        # Sort stops to visit all restaurants first, then all customers
        restaurant_stops = [(order.id, "pickup", order.pickup_location) for order in selected_orders]
        customer_stops = [(order.id, "delivery", order.delivery_location) for order in selected_orders]

        # Sort restaurant stops by distance
        restaurant_stops.sort(
            key=lambda x: self._calculate_travel_time(vehicles[nearest_vehicle].current_location, x[2])
        )

        # Create final route
        route_order = []
        for order_id, _, _ in restaurant_stops:
            route_order.append(order_id)

        return nearest_vehicle, route_order

    def _process_single_order(
        self, order: Order, vehicles, route_plan: List[List[int]], all_orders: List[Order]
    ) -> None:
        """Process single order with potential bundling with nearby restaurants."""
        print(f"\n[Order {order.id}] Starting processing")
        print(f"[Order {order.id}] Restaurant location: ({order.pickup_location.x:.2f}, {order.pickup_location.y:.2f})")
        print(
            f"[Order {order.id}] Customer location: ({order.delivery_location.x:.2f}, {order.delivery_location.y:.2f})"
        )

        # Find orders from nearby restaurants
        nearby_orders = self._find_nearby_orders(order, all_orders)

        if nearby_orders:
            print(f"[Order {order.id}] Found {len(nearby_orders)} nearby orders for potential bundling:")
            for nearby in nearby_orders:
                distance = self._calculate_distance(order.pickup_location, nearby.pickup_location)
                print(f"  - Order {nearby.id}: distance {distance:.2f}km")

            # Find nearest vehicle that can handle the bundle
            vehicle_id = self._find_nearest_vehicle(order.pickup_location, vehicles, route_plan, len(nearby_orders) + 1)

            if vehicle_id is not None:
                # Add all orders to vehicle's route
                bundle_orders = [order] + nearby_orders
                route_plan[vehicle_id].extend(o.id for o in bundle_orders)
                print(f"[Order {order.id}] Created bundle of {len(bundle_orders)} orders for vehicle {vehicle_id}")
                print(f"[Order {order.id}] Bundle contains orders: {[o.id for o in bundle_orders]}")
                return
            else:
                print(f"[Order {order.id}] No vehicle available for bundle of size {len(nearby_orders) + 1}")
        else:
            print(f"[Order {order.id}] No nearby orders found for bundling")

        # If no bundling possible, assign to nearest available vehicle
        print(f"[Order {order.id}] Attempting single order assignment")
        vehicle_id = self._find_nearest_vehicle(order.pickup_location, vehicles, route_plan, 1)
        if vehicle_id is not None:
            route_plan[vehicle_id].append(order.id)
            print(f"[Order {order.id}] Assigned to vehicle {vehicle_id} as single order")
        else:
            print(f"[Order {order.id}] Could not find available vehicle for single order")

    def _find_nearby_orders(self, order: Order, all_orders: List[Order]) -> List[Order]:
        """Find orders from restaurants near the given order's restaurant."""
        nearby_orders = []
        for other_order in all_orders:
            if other_order.id == order.id:
                continue

            distance = self._calculate_distance(order.pickup_location, other_order.pickup_location)

            if distance <= self.max_restaurant_distance:
                nearby_orders.append(other_order)

        return nearby_orders[: self.max_bundle_size - 1]  # Respect max bundle size

    def _find_nearest_vehicle(
        self, target_loc: Location, vehicles, route_plan: List[List[int]], required_capacity: int
    ) -> Optional[int]:
        min_travel_time = float("inf")
        best_vehicle_id = None

        for vehicle in vehicles:
            # Check if vehicle has enough remaining capacity
            current_load = len(route_plan[vehicle.id])
            if current_load + required_capacity > self.max_bundle_size:
                continue

            # If vehicle has active route, calculate from its last restaurant stop
            if route_plan[vehicle.id]:
                current_loc = vehicle.current_location
            else:
                current_loc = vehicle.current_location

            travel_time = self._calculate_travel_time(current_loc, target_loc)
            if travel_time < min_travel_time:
                min_travel_time = travel_time
                best_vehicle_id = vehicle.id

        return best_vehicle_id

    def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
        """Calculate travel time between locations."""
        return self._calculate_distance(loc1, loc2) / self.movement_per_step

    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate Euclidean distance between locations."""
        dx = loc2.x - loc1.x
        dy = loc2.y - loc1.y
        return np.sqrt(dx * dx + dy * dy)
