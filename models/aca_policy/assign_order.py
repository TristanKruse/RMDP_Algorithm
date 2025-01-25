from typing import List, Tuple
from datatypes import Order, State, Location
import numpy as np


class RouteAssigner:
    def __init__(
        self,
        service_time: float,
        mean_prep_time: float,
        delay_normalization_factor: float,
        movement_per_step: float,
    ):
        """
        Initialize the route assigner.

        Args:
            service_time: Time spent at each stop (restaurant/customer)
            mean_prep_time: Average food preparation time at restaurants
            delay_normalization_factor: Penalty multiplier for late deliveries
            vehicle_speed: Speed of vehicles in km/h
            street_network_factor: Factor to convert Euclidean to street distance
        """
        self.service_time = service_time
        self.mean_prep_time = mean_prep_time
        self.delay_normalization_factor = delay_normalization_factor
        self.movement_per_step = movement_per_step

    def _assign_order(
        self, route: List[Tuple[str, int]], order: Order, vehicle_id: int, state: State
    ) -> List[Tuple[str, int]]:
        """
        Assigns an order to a vehicle's route using the cheapest insertion method.

        Args:
            route: Current route as list of tuples (type, id)
            order: Order to be inserted
            vehicle_id: Vehicle ID
            state: Current state containing all system information

        Returns:
            Updated route with new order inserted at best positions

        Raises:
            ValueError: If order or vehicle not found in state
        """
        # Input validation
        if not any(v.id == vehicle_id for v in state.vehicles):
            raise ValueError(f"Vehicle {vehicle_id} not found in state")
        if not any(o.id == order.id for o in state.orders):
            raise ValueError(f"Order {order.id} not found in state")

        # Handle empty route case
        if not route:
            return [("R", order.id), ("C", order.id)]

        current_route = route.copy()

        # Phase 1: Check bundling opportunity
        restaurant_position = self._find_restaurant_in_route(current_route, order, state)
        restaurant_in_route = restaurant_position != -1

        # Phase 2: Insert restaurant if needed
        if not restaurant_in_route:
            best_r_pos = self._find_best_restaurant_position(current_route, order, vehicle_id, state)
            current_route.insert(best_r_pos, ("R", order.id))
            restaurant_position = best_r_pos

        # Phase 3: Insert customer after restaurant
        best_c_pos = self._find_best_customer_position(current_route, order, restaurant_position, vehicle_id, state)
        current_route.insert(best_c_pos, ("C", order.id))

        return current_route

    def _find_restaurant_in_route(self, route: List[Tuple[str, int]], order: Order, state: State) -> int:
        """Find if and where restaurant exists in route. Returns -1 if not found."""
        for i, (stop_type, stop_id) in enumerate(route):
            if stop_type == "R" and self._get_restaurant_id(stop_id, state) == self._get_restaurant_id(order.id, state):
                return i
        return -1

    def _find_best_restaurant_position(
        self, route: List[Tuple[str, int]], order: Order, vehicle_id: int, state: State
    ) -> int:
        """Find best position to insert restaurant stop."""
        min_cost = float("inf")
        best_pos = 0

        for i in range(len(route) + 1):
            test_route = route.copy()
            test_route.insert(i, ("R", order.id))
            cost = self._calculate_route_cost(test_route, vehicle_id, state)

            if cost < min_cost:
                min_cost = cost
                best_pos = i

        return best_pos

    def _find_best_customer_position(
        self, route: List[Tuple[str, int]], order: Order, restaurant_position: int, vehicle_id: int, state: State
    ) -> int:
        """Find best position to insert customer stop after their restaurant."""
        min_cost = float("inf")
        best_pos = restaurant_position + 1

        for i in range(restaurant_position + 1, len(route) + 1):
            test_route = route.copy()
            test_route.insert(i, ("C", order.id))
            cost = self._calculate_route_cost(test_route, vehicle_id, state)

            if cost < min_cost:
                min_cost = cost
                best_pos = i

        return best_pos

    def _calculate_route_cost(self, route: List[Tuple[str, int]], vehicle_id: int, state: State) -> float:
        """Calculate total route cost including travel, service, and delay costs."""
        if not route:
            return 0.0

        total_cost = 0.0
        current_time = state.time
        vehicle = next(v for v in state.vehicles if v.id == vehicle_id)
        current_loc = vehicle.current_location

        for stop_type, order_id in route:
            order = next(o for o in state.orders if o.id == order_id)
            next_loc = order.pickup_location if stop_type == "R" else order.delivery_location

            # Add travel costs
            travel_time = self._calculate_travel_time(current_loc, next_loc)
            current_time += travel_time
            total_cost += travel_time

            # Handle restaurant stops
            if stop_type == "R":
                wait_time = self._calculate_wait_time(order, current_time)
                current_time += wait_time
                total_cost += wait_time
                current_time += self.service_time
                total_cost += self.service_time

            # Handle customer stops
            else:
                current_time += self.service_time
                total_cost += self.service_time
                delay = max(0, current_time - order.deadline)
                total_cost += self.delay_normalization_factor * delay

            current_loc = next_loc

        return total_cost

    def _calculate_wait_time(self, order: Order, current_time: float) -> float:
        """Calculate expected waiting time at restaurant."""
        if order.ready_time is None:
            return max(0, self.mean_prep_time - (current_time - order.request_time))
        return max(0, order.ready_time - current_time)

    def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
        """Calculate travel time between locations using movement_per_step."""
        dx = loc2.x - loc1.x
        dy = loc2.y - loc1.y
        distance = np.sqrt(dx * dx + dy * dy)
        return distance / self.movement_per_step

    def _get_restaurant_id(self, order_id: int, state: State) -> int:
        """Get restaurant ID for an order."""
        try:
            order = next(o for o in state.orders if o.id == order_id)
            return hash((order.pickup_location.x, order.pickup_location.y))  # hash location
        except StopIteration:
            raise ValueError(f"Order {order_id} not found in state")
