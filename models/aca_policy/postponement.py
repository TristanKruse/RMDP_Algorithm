from typing import Set, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PostponementHandler:
    def __init__(self, max_postponements: int, max_postpone_time: float):
        self.max_postponements = max_postponements
        self.max_postpone_time = max_postpone_time

    def _can_postpone(self, postponed: Set[int], current_time: float) -> bool:
        """Check if more orders can be postponed based on max limit."""
        return len(postponed) < self.max_postponements

    def _get_order_info(self, order_id: int, state: dict) -> Optional[dict]:
        """Get order information from state dictionary."""
        return state["unassigned_orders"].get(order_id)

    def _is_next_stop_related(self, order_id: int, route_plan: dict, state: dict) -> bool:
        """Check if order's restaurant is next stop for any vehicle."""
        # Get order info
        order_info = self._get_order_info(order_id, state)
        if not order_info:
            logger.warning(f"No info found for order {order_id}")
            return False

        # Get pickup node id
        pickup_node_id = order_info["pickup_node_id"].id

        # Check each vehicle's next stop
        for route in route_plan.values():
            if route.sequence:  # If route has any stops
                next_stop = route.sequence[0]  # (node_id, pickups, deliveries)
                if next_stop[0] == pickup_node_id:  # If next stop is this restaurant
                    return True

        return False

    def evaluate_postponement(
        self,
        postponed: Set[int],
        route_plan: dict,
        order_id: int,
        current_time: float,
        state: dict,
    ) -> bool:
        """Evaluate whether an order should be postponed."""
        # Basic checks
        if not self._can_postpone(postponed, current_time):
            return False

        # Get order info
        order_info = self._get_order_info(order_id, state)
        if not order_info:
            logger.warning(f"No info found for order {order_id}")
            return False

        # Check if order's restaurant is next stop for any vehicle
        if self._is_next_stop_related(order_id, route_plan, state):
            return False

        # Time window check (if needed)
        if hasattr(self, 'max_postpone_time'):
            order_time = order_info["request_time"]
            if current_time - order_time >= self.max_postpone_time:
                return False

        return True