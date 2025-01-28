# bundling, when multiple orders at restaurant,
# bundling, when another order is close to the restaurant
from typing import List, Tuple, Set, Dict
import numpy as np
from datatypes import State, Location, Order


class FastestBundler:
    def __init__(self, movement_per_step: float, max_bundle_size: int):
        self.movement_per_step = movement_per_step
        self.max_bundle_size = max_bundle_size

    def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
        dx, dy = loc2.x - loc1.x, loc2.y - loc1.y
        return np.sqrt(dx * dx + dy * dy) / self.movement_per_step

    def _find_nearest_vehicle(self, target_loc: Location, vehicles, route_plan: List[List[int]]) -> int:
        min_travel_time = float("inf")
        best_vehicle_id = None

        for vehicle in vehicles:
            # Allow vehicles with routes if they haven't reached max bundle size
            if len(route_plan[vehicle.id]) >= self.max_bundle_size:
                continue

            # Calculate travel time to target
            travel_time = self._calculate_travel_time(vehicle.current_location, target_loc)

            # Update if this is fastest vehicle so far
            if travel_time < min_travel_time:
                min_travel_time = travel_time
                best_vehicle_id = vehicle.id

        return best_vehicle_id

    def _group_orders_by_restaurant(self, orders: List[Order]) -> Dict[Location, List[Order]]:
        """Group orders by their restaurant location."""
        restaurant_groups = {}
        for order in orders:
            loc_key = (order.pickup_location.x, order.pickup_location.y)  # Use coordinates as dictionary key
            if loc_key not in restaurant_groups:
                restaurant_groups[loc_key] = []
            restaurant_groups[loc_key].append(order)
        return restaurant_groups

    def _find_nearby_restaurants(
        self, target_loc: Location, restaurant_groups: Dict[Location, List[Order]], max_distance: float = 2.0
    ) -> List[Tuple[Location, List[Order]]]:
        """Find restaurants within max_distance of target_loc."""
        nearby_restaurants = []
        for loc_key, orders in restaurant_groups.items():
            loc = Location(x=loc_key[0], y=loc_key[1])
            if self._calculate_travel_time(target_loc, loc) <= max_distance:
                nearby_restaurants.append((loc, orders))
        return nearby_restaurants

    def solve(self, state: State) -> Tuple[List[List[int]], Set[int]]:
        # 1. Create copy of current route plan
        route_plan = [route.copy() for route in state.route_plan]

        # 2. Group unassigned orders by restaurant
        restaurant_groups = self._group_orders_by_restaurant(state.unassigned_orders)

        # 3. Process each vehicle
        for vehicle in state.vehicles:
            while len(route_plan[vehicle.id]) < self.max_bundle_size and restaurant_groups:
                # If vehicle has no orders yet, find nearest restaurant
                if not route_plan[vehicle.id]:
                    # Find nearest restaurant to vehicle
                    min_time = float("inf")
                    best_restaurant = None

                    for loc_key, orders in restaurant_groups.items():
                        restaurant_loc = Location(x=loc_key[0], y=loc_key[1])
                        time = self._calculate_travel_time(vehicle.current_location, restaurant_loc)
                        if time < min_time:
                            min_time = time
                            best_restaurant = loc_key

                    if best_restaurant:
                        # Add all orders from this restaurant up to capacity
                        orders = restaurant_groups[best_restaurant]
                        orders_to_add = orders[: self.max_bundle_size]
                        route_plan[vehicle.id].extend(order.id for order in orders_to_add)

                        # Remove assigned orders from restaurant_groups
                        remaining_orders = orders[self.max_bundle_size :]
                        if remaining_orders:
                            restaurant_groups[best_restaurant] = remaining_orders
                        else:
                            del restaurant_groups[best_restaurant]

                        # Find nearby restaurants for additional orders
                        nearby = self._find_nearby_restaurants(
                            Location(x=best_restaurant[0], y=best_restaurant[1]), restaurant_groups
                        )

                        # Add orders from nearby restaurants if capacity allows
                        for nearby_loc, nearby_orders in nearby:
                            if len(route_plan[vehicle.id]) >= self.max_bundle_size:
                                break
                            orders_to_add = nearby_orders[: self.max_bundle_size - len(route_plan[vehicle.id])]
                            route_plan[vehicle.id].extend(order.id for order in orders_to_add)

                            # Update restaurant_groups
                            loc_key = (nearby_loc.x, nearby_loc.y)
                            remaining_orders = nearby_orders[len(orders_to_add) :]
                            if remaining_orders:
                                restaurant_groups[loc_key] = remaining_orders
                            else:
                                del restaurant_groups[loc_key]
                else:
                    break  # Skip if vehicle already has orders
        print(f"Current Route Plan {route_plan}")
        return route_plan, set()
