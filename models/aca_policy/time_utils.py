from typing import List, Dict, Optional, Tuple, Set
from datatypes import State, Location
import numpy as np

# vehicle seems to always be none


class TimeCalculator:
    def __init__(
        self,
        mean_prep_time: float,
        prep_time_var: float,
        service_time: float,
        delay_normalization_factor: float,
        vehicle_speed: float = 40.0,
        street_network_factor: float = 1.4,
    ):
        self.mean_prep_time = mean_prep_time
        self.prep_time_var = prep_time_var
        self.service_time = service_time
        self.delay_normalization_factor = delay_normalization_factor
        self.vehicle_speed = vehicle_speed
        self.street_network_factor = street_network_factor

    def _calculate_slack(self, state: State, route_plan: List[List[int]]) -> float:
        """Calculate slack (flexibility) in route plan."""
        total_slack = 0.0

        for route in route_plan:
            for order_id in route:
                try:
                    order = next(o for o in state.orders if o.id == order_id)
                    planned_arrival = self._calculate_planned_arrival(state, route, order_id)
                    slack = max(0.0, order.deadline - planned_arrival)
                    total_slack += slack
                except Exception as e:
                    print(f"Error calculating slack for order {order_id}: {str(e)}")
                    continue

        return total_slack

    def _calculate_delay(
        self, state: State, route_plan: List[List[int]], buffer: float = 0.0, vehicle_id: Optional[int] = None
    ) -> Tuple[float, Set[int]]:
        """
        Calculate total delay and identify late orders.

        Args:
            state: Current system state
            route_plan: List of vehicle routes
            buffer: Time buffer to add to planned arrivals
            vehicle_id: Optional specific vehicle being evaluated

        Returns:
            Tuple of (total_delay, set_of_late_order_ids)
        """
        if not route_plan:
            return 0.0, set()

        total_delay = 0.0
        late_order_ids = set()

        # Process each route
        for route in route_plan:
            for order_id in route:
                try:
                    order = next(o for o in state.orders if o.id == order_id)

                    # Calculate planned arrival using specified vehicle
                    planned_arrival = self._calculate_planned_arrival(
                        state=state, route=route, target_order_id=order_id, vehicle_id=vehicle_id
                    )

                    if planned_arrival == float("inf"):
                        continue

                    buffered_arrival = planned_arrival + buffer
                    delay = max(0, buffered_arrival - order.deadline)

                    if delay > 0:
                        late_order_ids.add(order_id)
                        total_delay += delay * self.delay_normalization_factor

                except Exception as e:
                    print(f"Error calculating delay for order {order_id}: {str(e)}")
                    continue

        return total_delay, late_order_ids

    def _calculate_planned_arrival(
        self, state: State, route: List[int], target_order_id: int, vehicle_id: Optional[int] = None
    ) -> float:
        """
        Calculate planned arrival time for a specific order.

        Args:
            state: Current state
            route: Route to evaluate
            target_order_id: Order we want arrival time for
            vehicle_id: Optional specific vehicle to use (for evaluating potential assignments)
        """
        try:
            # print(f"\nCalculating planned arrival for order {target_order_id}")
            # print(f"Current route: {route}")
            # print(f"Using vehicle_id: {vehicle_id}")

            if target_order_id not in route:
                print(f"Target order {target_order_id} not in route")
                return float("inf")

            order_index = route.index(target_order_id)
            current_time = state.time

            # Get vehicle's location
            if vehicle_id is not None:
                # Use specified vehicle for evaluation
                vehicle = next((v for v in state.vehicles if v.id == vehicle_id), None)
                if not vehicle:
                    print(f"Specified vehicle {vehicle_id} not found!")
                    return float("inf")
            else:

                # Find vehicle that has this route
                found_vehicle_id = self._find_vehicle_for_route(state, target_order_id)
                if found_vehicle_id is not None:
                    vehicle = next(v for v in state.vehicles if v.id == found_vehicle_id)
                else:
                    # print("No vehicle found for route, using first available")
                    vehicle = state.vehicles[0]

                # # Find vehicle that has this route
                # vehicle = None
                # for v in state.vehicles:
                #     if v.id < len(state.route_plan):
                #         route_orders = state.route_plan[v.id]
                #         if any(
                #             target_order_id in r
                #             for r in ([route_orders] if isinstance(route_orders, int) else route_orders)
                #         ):
                #             vehicle = v
                #             break

                if vehicle is None:
                    # print("No vehicle found for route, using first available")
                    vehicle = state.vehicles[0]

            current_loc = vehicle.current_location
            # print(f"Using vehicle {vehicle.id} at location ({current_loc.x:.2f}, {current_loc.y:.2f})")

            # Process all stops up to target order
            for i in range(order_index + 1):
                try:
                    order_id = route[i]
                    # print(f"\nProcessing order {order_id} in sequence")

                    current_order = next((o for o in state.orders if o.id == order_id), None)
                    if not current_order:
                        print(f"Could not find order {order_id} in state orders")
                        return float("inf")

                    # Travel to restaurant
                    travel_time = self._calculate_travel_time(current_loc, current_order.pickup_location)
                    current_time += travel_time
                    # print(f"Travel to restaurant: +{travel_time:.1f} min")

                    # Handle restaurant service and ready time
                    if not current_order.ready_time:
                        expected_ready = current_order.request_time + self.mean_prep_time
                        wait_time = max(0, expected_ready - current_time)
                        if wait_time > 0:
                            # print(f"Waiting for food preparation: +{wait_time:.1f} min")
                            current_time = expected_ready
                    else:
                        wait_time = max(0, current_order.ready_time - current_time)
                        if wait_time > 0:
                            # print(f"Waiting for ready time: +{wait_time:.1f} min")
                            current_time = current_order.ready_time

                    current_time += self.service_time
                    # print(f"Restaurant service: +{self.service_time} min")

                    # Travel to customer
                    travel_time = self._calculate_travel_time(
                        current_order.pickup_location, current_order.delivery_location
                    )
                    current_time += travel_time
                    # print(f"Travel to customer: +{travel_time:.1f} min")

                    current_time += self.service_time
                    # print(f"Customer service: +{self.service_time} min")

                    current_loc = current_order.delivery_location
                    # print(f"Current time after order {order_id}: {current_time:.1f}")

                except Exception as e:
                    print(f"Error processing order {order_id}: {str(e)}")
                    return float("inf")

            # print(f"\nFinal planned arrival time: {current_time:.1f}")
            return current_time

        except Exception as e:
            print(f"Error calculating planned arrival: {str(e)}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            return float("inf")

    def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
        """Calculate travel time between two locations."""
        dx = loc2.x - loc1.x
        dy = loc2.y - loc1.y
        distance = self.street_network_factor * np.sqrt(dx * dx + dy * dy)
        return distance / self.vehicle_speed * 60  # Convert to minutes

    def _find_vehicle_for_route(self, state: State, target_order_id: int) -> Optional[int]:
        """Helper method to find vehicle associated with a route containing target_order."""
        for v in state.vehicles:
            if v.id >= len(state.route_plan):
                continue
            route_orders = state.route_plan[v.id]
            # Handle both single integers and lists
            if isinstance(route_orders, int):
                if route_orders == target_order_id:
                    return v.id
            elif isinstance(route_orders, list):
                if target_order_id in route_orders:
                    return v.id
        return None
