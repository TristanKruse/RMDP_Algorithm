from typing import List, Set
from itertools import permutations
from datatypes import Order
from copy import deepcopy


class RouteUtils:
    def __init__(self, vehicle_capacity: int):
        self.vehicle_capacity = vehicle_capacity

    def _generate_order_sequences(self, orders: List[Order]) -> List[List[Order]]:
        """Generate all possible sequences of unassigned orders. -> n! sequences"""
        return list(permutations(orders))

    def _remove_postponed_orders(self, route_plan: List[List[int]], postponed: Set[Order]) -> List[List[int]]:
        """Remove postponed orders from the route plan."""
        new_route_plan = deepcopy(route_plan)
        postponed_ids = {order.id for order in postponed}
        return [[order_id for order_id in route if order_id not in postponed_ids] for route in new_route_plan]

    def _is_feasible_assignment(self, route: List[int], order: Order) -> bool:
        """Check if adding order to route is feasible considering capacity"""
        return len(route) < self.vehicle_capacity
