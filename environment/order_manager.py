# environment/order_manager.py
from typing import List, Set, Tuple
import numpy as np
import random
from datatypes import Order, Location, Node
import logging


# Configure logging format
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Silence matplotlib and PIL debug messages
for logger_name in ["matplotlib", "PIL"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


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

    def generate_new_orders(self, current_time: float, restaurants: List[Node]) -> None:
        if current_time >= self.next_order_time:
            self.active_orders.append(self.generate_order(current_time, restaurants))
            self.next_order_time = current_time + np.random.exponential(self.mean_interarrival_time)

    def generate_order(self, current_time: float, restaurants: List[Node]) -> Order:
        # Generate customer location
        area_width, area_height = self.service_area
        customer_loc = Location(x=random.uniform(0, area_width), y=random.uniform(0, area_height))
        customer_node = Node(id=self.next_order_id + len(restaurants), location=customer_loc)

        restaurant_node = random.choice(restaurants)

        # Generate preparation time using gamma distribution
        prep_time = np.random.gamma(
            shape=(self.mean_prep_time**2) / self.prep_time_var, scale=self.prep_time_var / self.mean_prep_time
        )

        order = Order(
            id=self.next_order_id,
            request_time=current_time,
            pickup_node_id=restaurant_node,
            delivery_node_id=customer_node,  # Use offset for customer nodes
            deadline=current_time + self.delivery_window,
            ready_time=current_time + prep_time,
            service_time=self.service_time,
        )

        self.next_order_id += 1
        return order

    def handle_postponed_orders(self, postponed_orders: Set) -> None:
        if postponed_orders:
            self.postponed_order_ids.update(
                order.id if isinstance(order, Order) else order for order in postponed_orders
            )

    def get_active_orders(self) -> List[Order]:
        return self.active_orders

    def get_postponed_orders(self) -> Set[int]:
        return self.postponed_order_ids

    def cleanup_delivered_orders(self):
        self.active_orders = [order for order in self.active_orders if order.status != "delivered"]