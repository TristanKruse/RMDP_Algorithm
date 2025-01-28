from typing import List, Tuple, Set, Dict
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    PICKUP = "pickup"
    DELIVERY = "delivery"
    REPOSITION = "reposition"


@dataclass
class RouteAction:
    action_type: ActionType
    order_id: int = None  # None for REPOSITION
    location_id: int = None  # Used for REPOSITION


class RouteValidator:
    def __init__(self, max_bundle_size: int):
        self.max_bundle_size = max_bundle_size

    def validate_route(self, route: List[RouteAction]) -> Tuple[bool, str]:
        """Validates if a route is feasible.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        picked_up_orders = set()  # Track which orders are in vehicle
        current_load = 0  # Track current vehicle load

        for action in route:
            if action.action_type == ActionType.PICKUP:
                # Check vehicle capacity
                if current_load >= self.max_bundle_size:
                    return False, f"Vehicle capacity exceeded at action: {action}"

                # Check if order already picked up
                if action.order_id in picked_up_orders:
                    return False, f"Order {action.order_id} already picked up"

                picked_up_orders.add(action.order_id)
                current_load += 1

            elif action.action_type == ActionType.DELIVERY:
                # Check if order was picked up
                if action.order_id not in picked_up_orders:
                    return False, f"Attempting to deliver order {action.order_id} before pickup"

                picked_up_orders.remove(action.order_id)
                current_load -= 1

            elif action.action_type == ActionType.REPOSITION:
                # Repositioning is always valid
                pass

        # Check if all picked up orders were delivered
        if picked_up_orders:
            return False, f"Orders {picked_up_orders} were picked up but not delivered"

        return True, "Route is valid"

    def create_bundle_route(
        self,
        pickup_sequence: List[Tuple[int, int]],  # List of (order_id, restaurant_id)
        delivery_sequence: List[Tuple[int, int]],  # List of (order_id, customer_id)
    ) -> List[RouteAction]:
        """Creates a validated bundle route from pickup and delivery sequences."""

        # Create route from sequences
        route = []

        # Add pickup actions
        for order_id, restaurant_id in pickup_sequence:
            route.append(RouteAction(ActionType.PICKUP, order_id))

        # Add delivery actions
        for order_id, customer_id in delivery_sequence:
            route.append(RouteAction(ActionType.DELIVERY, order_id))

        # Validate route
        is_valid, error_msg = self.validate_route(route)
        if not is_valid:
            raise ValueError(f"Invalid route created: {error_msg}")

        return route
