# from typing import Set, List, Optional
# from datatypes import Order


# class PostponementHandler:
#     def __init__(self, max_postponements: int, max_postpone_time: float, min_service_time: float = 20.0):
#         self.max_postponements = max_postponements
#         self.max_postpone_time = max_postpone_time
#         self.min_service_time = min_service_time

#     def _get_order_from_id(self, order_id: int, state) -> Optional[Order]:
#         """Helper method to get Order object from ID"""
#         try:
#             return next((o for o in state.orders if o.id == order_id), None)
#         except Exception as e:
#             print(f"Error getting order from ID {order_id}: {str(e)}")
#             return None

#     def _is_next_stop_related(self, order: Order, route_plan: List[List[int]], state) -> bool:
#         """
#         Check if order's restaurant would be the next stop for any vehicle's route.
#         According to paper: don't postpone if vehicle would drive to order's restaurant next.
#         """
#         try:
#             # Get order and its restaurant location
#             order_to_check = next((o for o in state.orders if o.id == order.id), None)
#             if not order_to_check:
#                 print(f"Warning: Order {order.id} not found in state")
#                 return False

#             restaurant_to_check = order_to_check.pickup_location

#             # Check each vehicle's route
#             for route in route_plan:
#                 if not route:
#                     continue

#                 # Get next planned stop's order
#                 if route:  # if route is not empty
#                     next_order = next((o for o in state.orders if o.id == route[0]), None)
#                     if next_order:
#                         # If next stop is a restaurant pickup and it's the same restaurant
#                         if (
#                             next_order.status == "pending"
#                             and next_order.pickup_location.x == restaurant_to_check.x
#                             and next_order.pickup_location.y == restaurant_to_check.y
#                         ):
#                             return True

#             return False

#         except Exception as e:
#             print(f"Error checking next stop relation: {str(e)}")
#             return False

#     def _can_postpone(self, postponed: Set[Order], order: Order, current_time: float) -> bool:
#         """Determine if an order can be postponed based on criteria."""
#         # Check 1: Maximum postponements limit
#         if len(postponed) >= self.max_postponements:
#             return False

#         # Check 2: Order hasn't already been postponed too long
#         if order.first_postpone_time is not None:
#             if current_time - order.first_postpone_time >= self.max_postpone_time:
#                 return False
#         # Add logging
#         # print(f"Order {order.id} eligible for postponement")
#         return True
#         # Claude suggestions
#         # # Check 4: Sufficient time margin before deadline
#         # time_to_deadline = order.deadline - current_time
#         # if time_to_deadline < self.min_service_time:
#         #     return False

#         # # Check 6: Enough buffer for postponement and service
#         # return time_to_deadline > self.max_postpone_time + self.min_service_time

#     def evaluate_postponement(
#         self,
#         postponed: Set[Order],
#         route_plan: List[List[int]],
#         order,  # Can be Order or int
#         current_time: float,
#         state=None,  # Added state parameter
#     ) -> bool:
#         """Evaluate whether an order should be postponed."""
#         # Convert order ID to Order object if necessary
#         if isinstance(order, int) and state is not None:
#             order = self._get_order_from_id(order, state)
#             if not order:
#                 print(f"Warning: Could not find order for ID {order}")
#                 return False

#         if not isinstance(order, Order):
#             print(f"Warning: Invalid order type {type(order)} and no state provided to look up order")
#             return False

#         # Check1: check if postponement is possible
#         if self._can_postpone(postponed, order, current_time):
#             # Check2: check if order is related to next stops
#             if not self._is_next_stop_related(order, route_plan, state):
#                 # Only set first_postpone_time when actually postponing
#                 if order.first_postpone_time is None:
#                     order.first_postpone_time = current_time
#                 # Update order's postponement tracking
#                 order.postpone_count += 1
#                 # print(f"Postponing order {order.id}")
#                 return True

#         return False



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