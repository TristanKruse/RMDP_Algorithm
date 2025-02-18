# from typing import List, Set
# from itertools import permutations
# from datatypes import Order
# from copy import deepcopy


# class RouteUtils:
#     def __init__(self, vehicle_capacity: int):
#         self.vehicle_capacity = vehicle_capacity

#     def _generate_order_sequences(self, orders: List[Order]) -> List[List[Order]]:
#         """Generate all possible sequences of unassigned orders. -> n! sequences"""
#         return list(permutations(orders))

#     def _remove_postponed_orders(self, route_plan: List[List[int]], postponed: Set[Order]) -> List[List[int]]:
#         """Remove postponed orders from the route plan."""
#         new_route_plan = deepcopy(route_plan)
#         postponed_ids = {order.id for order in postponed}
#         return [[order_id for order_id in route if order_id not in postponed_ids] for route in new_route_plan]

#     def _is_feasible_assignment(self, route: List[int], order: Order) -> bool:
#         """Check if adding order to route is feasible considering capacity"""
#         return len(route) < self.vehicle_capacity


from typing import List, Set, Dict, Tuple
from itertools import permutations
from datatypes import Route
import logging

logger = logging.getLogger(__name__)

class RouteUtils:
    def __init__(self, vehicle_capacity: int):
        self.vehicle_capacity = vehicle_capacity
    
    def _generate_order_sequences(self, order_items) -> List[List[Tuple[int, dict]]]:
            """Generate all possible sequences of unassigned orders.
            
            Args:
                order_items: Either list of orders or dictionary of orders
            Returns:
                List of possible order sequences
            """
            # If we get a dictionary, convert to items list
            if isinstance(order_items, dict):
                order_items = list(order_items.items())
                
            # Generate all possible permutations
            return list(permutations(order_items))

    def _count_route_load(self, route: Route) -> int:
        """Count number of orders currently being carried in route."""
        if not route.sequence:
            return 0
            
        current_load = 0
        carried_orders = set()
        
        for _, pickups, deliveries in route.sequence:
            carried_orders.update(pickups)
            carried_orders.difference_update(deliveries)
            current_load = max(current_load, len(carried_orders))
            
        return current_load

    def _is_valid_insertion(self, route: Route, order_id: int, pickup_node: int, delivery_node: int) -> bool:
        """Check if inserting order at current position maintains route validity."""
        # Check vehicle capacity
        if self._count_route_load(route) >= self.vehicle_capacity:
            return False
            
        # Ensure pickup comes before delivery
        pickup_found = False
        for _, pickups, deliveries in route.sequence:
            if order_id in pickups:
                pickup_found = True
            if order_id in deliveries and not pickup_found:
                return False
                
        return True

    def _remove_postponed_orders(self, route_plan: Dict[int, Route], postponed: Set[int]) -> Dict[int, Route]:
        """Remove postponed orders from route plan."""
        updated_plan = {}
        
        for vehicle_id, route in route_plan.items():
            new_sequence = []
            for node_id, pickups, deliveries in route.sequence:
                # Remove postponed orders from pickups and deliveries
                new_pickups = pickups - postponed
                new_deliveries = deliveries - postponed
                
                # Only keep stop if it still has orders
                if new_pickups or new_deliveries:
                    new_sequence.append((node_id, new_pickups, new_deliveries))
            
            updated_plan[vehicle_id] = Route(
                vehicle_id=vehicle_id,
                sequence=new_sequence,
                total_distance=0.0,
                total_time=0.0
            )
            
        return updated_plan

    def _create_pickup_delivery_pair(
        self, order_id: int, order_info: dict
    ) -> Tuple[Tuple[int, Set[int], Set[int]], Tuple[int, Set[int], Set[int]]]:
        """Create pickup-delivery pair of stops for an order."""
        pickup_node_id = order_info["pickup_node_id"].id
        delivery_node_id = order_info["delivery_node_id"].id
        
        pickup_stop = (pickup_node_id, {order_id}, set())
        delivery_stop = (delivery_node_id, set(), {order_id})
        
        return pickup_stop, delivery_stop

    def _is_feasible_bundle(self, route: Route, new_order_id: int) -> bool:
        """Check if adding order to route would exceed vehicle capacity."""
        max_load = self._count_route_load(route)
        return max_load < self.vehicle_capacity