# environment/state_handler.py
from typing import Dict, List, Set
from datatypes import State, Order, Node, Route


class StateHandler:
    def __init__(self, num_vehicles: int):
        self.num_vehicles = num_vehicles
        self.route_plan = {
            i: Route(vehicle_id=i, sequence=[], total_distance=0.0, total_time=0.0) for i in range(num_vehicles)
        }

    def reset(self) -> None:
        self.route_plan = {
            i: Route(vehicle_id=i, sequence=[], total_distance=0.0, total_time=0.0) for i in range(self.num_vehicles)
        }

    def create_new_state(self, current_time: float, order_manager, vehicle_manager, location_manager) -> State:
        active_orders = order_manager.active_orders

        # Create a dictionary of nodes from restaurants
        nodes = {i: Node(id=i, location=restaurant) for i, restaurant in enumerate(location_manager.restaurants)}
        return State(
            time=current_time,
            nodes=nodes,  # Add nodes to State creation
            orders=active_orders,
            route_plan=self.route_plan,
            unassigned_orders=self._get_unassigned_orders(active_orders),
            vehicles=vehicle_manager.vehicles,
        )

    def update_route_plan(self, new_route_plan: Dict[int, Route]) -> None:
        for vehicle_id, route in new_route_plan.items():
            self.route_plan[vehicle_id] = route.copy()

    def _get_unassigned_orders(self, orders: List[Order]) -> Set[int]:
        # Get all order IDs from routes
        assigned_orders = {order_id for route in self.route_plan.values() for order_id in route.get_all_orders()}

        # Return order IDs not in assigned_orders
        return {order.id for order in orders if order.id not in assigned_orders}
