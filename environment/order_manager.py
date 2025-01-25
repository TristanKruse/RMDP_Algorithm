# environment/order_manager.py
from typing import List, Set, Tuple
import numpy as np
import random
from datatypes import Order, Location


class OrderManager:
    """Manages order generation and tracking for delivery environment."""

    def __init__(
        self,
        mean_prep_time: float,
        prep_time_var: float,
        delivery_window: float,
        service_time: float,
        mean_interarrival_time: float,
        service_area_dimensions: Tuple[float, float],
    ):
        # Order generation parameters
        self.mean_prep_time = mean_prep_time
        self.prep_time_var = prep_time_var
        self.delivery_window = delivery_window
        self.service_time = service_time
        self.mean_interarrival_time = mean_interarrival_time
        self.service_area = service_area_dimensions

        # Order tracking
        self.next_order_id = 0
        self.next_order_time = 0
        self.active_orders = []
        self.postponed_order_ids = set()

    def reset(self) -> None:
        self.next_order_id = self.next_order_time = 0
        self.active_orders = []
        self.postponed_order_ids = set()

    def generate_new_orders(self, current_time: float, restaurants: List[Location]) -> None:
        if current_time >= self.next_order_time:
            self.active_orders.append(self.generate_order(current_time, restaurants))
            self.next_order_time = current_time + np.random.exponential(self.mean_interarrival_time)

    def generate_order(self, current_time: float, restaurants: List[Location]) -> Order:
        """
        Generate a new order with random customer location and restaurant.

        Args:
            current_time: Current simulation time
            restaurants: List of available restaurant locations

        Returns:
            New Order object
        """
        restaurant = random.choice(restaurants)
        area_width, area_height = self.service_area

        # Generate random customer location
        customer_loc = Location(x=random.uniform(0, area_width), y=random.uniform(0, area_height))

        # Generate preparation time using gamma distribution
        prep_time = np.random.gamma(
            shape=(self.mean_prep_time**2) / self.prep_time_var, scale=self.prep_time_var / self.mean_prep_time
        )

        # Create order object
        order = Order(
            id=self.next_order_id,
            request_time=current_time,
            pickup_location=restaurant,
            delivery_location=customer_loc,
            deadline=current_time + self.delivery_window,
            ready_time=current_time + prep_time,
            service_time=self.service_time,
            pickup_service_time=self.service_time,
        )

        # Add tracking print
        print(f"\n[Order {order.id}] Created at t={current_time:.1f}")
        print(f"[Order {order.id}] Restaurant location: ({restaurant.x:.2f}, {restaurant.y:.2f})")
        print(f"[Order {order.id}] Customer location: ({customer_loc.x:.2f}, {customer_loc.y:.2f})")
        print(f"[Order {order.id}] Expected ready time: {order.ready_time:.1f}")
        print(f"[Order {order.id}] Deadline: {order.deadline:.1f}")

        self.next_order_id += 1
        return order

    def generate_order(self, current_time: float, restaurants: List[Location]) -> Order:
        restaurant = random.choice(restaurants)
        area_width, area_height = self.service_area
        # Generate random customer location
        customer_loc = Location(x=random.uniform(0, area_width), y=random.uniform(0, area_height))
        # Generate preparation time using gamma distribution
        prep_time = np.random.gamma(
            shape=(self.mean_prep_time**2) / self.prep_time_var, scale=self.prep_time_var / self.mean_prep_time
        )

        order = Order(
            id=self.next_order_id,
            request_time=current_time,
            pickup_location=restaurant,
            delivery_location=customer_loc,
            deadline=current_time + self.delivery_window,
            ready_time=current_time + prep_time,
            service_time=self.service_time,
            pickup_service_time=self.service_time,
        )

        self.next_order_id += 1
        return order

    def handle_postponed_orders(self, postponed_orders: Set) -> None:
        """
        Update tracking of postponed orders.

        Args:
            postponed_orders: Set of orders or order IDs to be postponed
        """
        if not postponed_orders:
            return

        # Handle both Order objects and order IDs
        postponed_ids = (
            {order.id for order in postponed_orders}
            if isinstance(next(iter(postponed_orders)), Order)
            else set(postponed_orders)
        )
        self.postponed_order_ids.update(postponed_ids)

    def get_active_orders(self) -> List[Order]:
        """
        Get list of currently active orders.

        Returns:
            List of active orders
        """
        return self.active_orders

    def get_postponed_orders(self) -> Set[int]:
        """
        Get set of postponed order IDs.

        Returns:
            Set of postponed order IDs
        """
        return self.postponed_order_ids

    def cleanup_delivered_orders(self) -> None:
        """Remove delivered orders from active orders list."""
        self.active_orders = [order for order in self.active_orders if order.status != "delivered"]
