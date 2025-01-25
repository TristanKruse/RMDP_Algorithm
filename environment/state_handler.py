# environment/state_handler.py
from typing import List
from datatypes import State, Order


class StateHandler:
    def __init__(self, num_vehicles: int):
        self.num_vehicles = num_vehicles
        self.route_plan = [[] for _ in range(num_vehicles)]

    def reset(self) -> None:
        self.route_plan = [[] for _ in range(self.num_vehicles)]

    def create_new_state(self, current_time: float, order_manager, vehicle_manager) -> State:
        active_orders = order_manager.active_orders
        return State(
            time=current_time,
            orders=active_orders,
            route_plan=self.route_plan,
            unassigned_orders=self._get_unassigned_orders(active_orders),
            vehicles=vehicle_manager.vehicles,
        )

    def update_route_plan(self, new_route_plan: List[List[int]]) -> None:
        self.route_plan = new_route_plan

    def _get_unassigned_orders(self, orders: List[Order]) -> List[Order]:
        return [order for order in orders if not any(order.id in route for route in self.route_plan)]
