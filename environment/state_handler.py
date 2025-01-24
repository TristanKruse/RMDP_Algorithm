# environment/state_handler.py
from typing import List
from datatypes import State, Order


class StateHandler:
    """Handles state management for the restaurant delivery environment."""

    def __init__(self, num_vehicles: int):
        """
        Initialize the state handler.

        Args:
            num_vehicles: Number of vehicles in the system
        """
        self.num_vehicles = num_vehicles
        self.route_plan = [[] for _ in range(num_vehicles)]

    def reset(self) -> None:
        """Reset the state handler to initial conditions."""
        self.route_plan = [[] for _ in range(self.num_vehicles)]

    def create_new_state(
        self, current_time: float, order_manager, vehicle_manager  # OrderManager instance  # VehicleManager instance
    ) -> State:
        """
        Create a new state based on current system conditions.

        Args:
            current_time: Current simulation time
            order_manager: OrderManager instance containing order information
            vehicle_manager: VehicleManager instance containing vehicle information

        Returns:
            State: New state object with current system information
        """

        active_orders = order_manager.active_orders
        vehicles = vehicle_manager.vehicles

        return State(
            time=current_time,
            orders=active_orders,
            route_plan=self.route_plan,
            unassigned_orders=self._get_unassigned_orders(active_orders),
            vehicles=vehicles,
        )

    def update_route_plan(self, new_route_plan: List[List[int]]) -> None:
        """
        Update the current route plan.

        Args:
            new_route_plan: New route plan to replace current one
        """
        self.route_plan = new_route_plan

    def _get_unassigned_orders(self, orders: List[Order]) -> List[Order]:
        """
        Get list of orders that haven't been assigned to any route.

        Args:
            orders: List of orders to check

        Returns:
            List of unassigned orders
        """
        return [order for order in orders if not self._is_order_assigned(order)]

    def _is_order_assigned(self, order: Order) -> bool:
        """
        Check if an order is assigned to any vehicle route.

        Args:
            order: Order to check

        Returns:
            True if order is assigned to a route, False otherwise
        """
        return any(order.id in route for route in self.route_plan)
