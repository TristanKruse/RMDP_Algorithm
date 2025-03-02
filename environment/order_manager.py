# environment/order_manager.py
from typing import List, Set, Tuple
import numpy as np
import random
from datatypes import Order, Location, Node
import logging
from environment.demand_pattern import DemandPattern



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
        simulation_duration: float,  # Total simulation duration 
        demand_pattern=None,
    ):
        # Order generation parameters
        self.mean_prep_time = mean_prep_time
        self.prep_time_var = prep_time_var
        self.delivery_window = delivery_window
        self.service_time = service_time
        self.mean_interarrival_time = mean_interarrival_time
        self.service_area = service_area_dimensions
        self.base_demand_rate = mean_interarrival_time  
        self.service_area = service_area_dimensions
        self.simulation_duration = simulation_duration

        self.demand_pattern = DemandPattern(demand_pattern)


        # Order tracking
        self.next_order_id = 0
        self.next_order_time = 0
        self.active_orders = []

    def reset(self) -> None:
        self.next_order_id = self.next_order_time = 0
        self.active_orders = []
        self.postponed_order_ids = set()

    # def generate_new_orders(self, current_time: float, restaurants: List[Node]) -> None:
    #     if current_time >= self.next_order_time:
    #         self.active_orders.append(self.generate_order(current_time, restaurants))
    #         self.next_order_time = current_time + np.random.exponential(self.mean_interarrival_time)

    def generate_new_orders(self, current_time: float, restaurants: List[Node]) -> None:
        if current_time >= self.next_order_time:
            self.active_orders.append(self.generate_order(current_time, restaurants))
            
            # Get current demand rate factor based on time of day (1.0 is the baseline)
            rate_factor = self.demand_pattern.get_rate(current_time, self.simulation_duration)
            
            # Calculate actual interarrival time by scaling the base interarrival time
            # Higher rate factor = shorter interarrival time (more frequent orders)
            if rate_factor > 0:
                # Divide base interarrival time by rate factor to get actual interarrival time
                actual_interarrival_time = self.base_demand_rate / rate_factor
            else:
                actual_interarrival_time = float('inf')  # No arrivals
                
            # Schedule next order with exponential distribution
            self.next_order_time = current_time + np.random.exponential(actual_interarrival_time)

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

    def get_active_orders(self) -> List[Order]:
        return self.active_orders

    def get_postponed_orders(self) -> Set[int]:
        return self.postponed_order_ids

    def cleanup_delivered_orders(self):
        self.active_orders = [order for order in self.active_orders if order.status != "delivered"]

    def get_arrival_rate(current_time):
        # Time in hours (assuming simulation time is in minutes)
        hour = (current_time % 1440) / 60  # 1440 minutes = 24 hours
        
        # Define arrival rates for each hour (this matches your graph)
        hourly_rates = [
            6, 3, 1, 0.5, 0.5, 0.5,          # 0-5 AM: Low activity
            2, 5, 10, 15, 18, 90,            # 6-11 AM: Morning rise & lunch peak
            40, 22, 19, 19, 30, 40,          # 12-5 PM: Afternoon
            62, 62, 40, 25, 18, 10           # 6-11 PM: Evening peak and decline
        ]
        
        # Get rate for current hour
        hour_index = int(hour)
        if hour_index >= len(hourly_rates):
            hour_index = len(hourly_rates) - 1
        return hourly_rates[hour_index]