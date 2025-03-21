from typing import Set, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PostponementHandler:
    def __init__(self, max_postponements: int, max_postpone_time: float):
        self.max_postponements = max_postponements
        self.max_postpone_time = max_postpone_time

    def evaluate_postponement(
        self,
        postponed: Set[int],
        route_plan: dict,
        order_id: int,
        current_time: float,
        state: dict,
    ) -> bool:
        """Evaluate whether an order should be postponed.
        
        Following Ulmer et al. criteria:
        1. Don't exceed maximum number of postponements
        2. Don't postpone orders related to next stops
        3. Don't postpone orders that have been waiting too long
        """
        # 1. Check if we already have too many postponed orders
        if len(postponed) >= self.max_postponements:
            return False

        # 2. Get order info
        order_info = state["unassigned_orders"].get(order_id)
        if not order_info:
            logger.warning(f"No info found for order {order_id}")
            return False

        # 3. Check if order's restaurant is next stop for any vehicle
        pickup_node_id = order_info["pickup_node_id"].id
        for route in route_plan.values():
            if route.sequence:  # If route has any stops
                next_stop = route.sequence[0]  # (node_id, pickups, deliveries)
                if next_stop[0] == pickup_node_id:  # If next stop is this restaurant
                    return False

        # 4. Check how long order has been in the system
        order_time = order_info["request_time"]
        if current_time - order_time >= self.max_postpone_time:
            return False

        # 5. Default is to postpone if none of the above conditions are met
        return True