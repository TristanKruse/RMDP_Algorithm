# environment/order_manager.py
from typing import List, Tuple
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

    def cleanup_delivered_orders(self):
        self.active_orders = [order for order in self.active_orders if order.status != "delivered"]


#### For individual orders, from the Meituan dataset.

    def set_real_orders(self, orders_df):
        """
        Set real orders from dataset to be used instead of generating random orders
        
        Parameters:
        -----------
        orders_df : pandas.DataFrame
            DataFrame containing real order data with at least:
            - order_push_time: time when order was placed
            - sender_lat/lng: restaurant coordinates
            - recipient_lat/lng: customer coordinates 
            - poi_id: restaurant ID
            - order_id: order ID
        """
        # Normalize and sort the orders by push time
        self.real_orders = orders_df.sort_values('order_push_time').reset_index(drop=True)
        
        # Initialize the index to track which order to process next
        self.current_order_index = 0
        
        # Flag to indicate we're using real orders
        self.using_real_orders = True
        
        # Calculate simulation start time (use the earliest order push time)
        self.simulation_start_time = self.real_orders['order_push_time'].min()
        
        logger.info(f"Loaded {len(self.real_orders)} real orders for simulation")
        logger.info(f"Order timespan: {self.real_orders['order_push_time'].min()} to {self.real_orders['order_push_time'].max()}")

    def set_order_generator(self, generator):
        """Set custom order generator"""
        self.order_generator = generator
        self.using_order_generator = True
        # Reset any existing order generation state
        self.next_order_time = 0
        if hasattr(self, 'current_order_index'):
            self.current_order_index = 0

    def generate_new_orders(self, current_time: float, restaurants: List[Node]) -> None:
        """
        Generate new orders based on simulation time
        
        Uses one of three approaches:
        1. Custom OrderGenerator (if set with set_order_generator)
        2. Real orders replay (if set with set_real_orders)
        3. Default Poisson process
        """
        # Priority 1: Use custom OrderGenerator if available
        if hasattr(self, 'using_order_generator') and self.using_order_generator:
            # Get geo_bounds if available for coordinate conversion
            geo_bounds = getattr(self.location_manager, 'geo_bounds', None) if hasattr(self, 'location_manager') else None
            
            # Generate new orders using the OrderGenerator
            new_orders = self.order_generator.generate_orders(current_time, restaurants, geo_bounds)
            self.active_orders.extend(new_orders)
            
        # Priority 2: Use existing real orders functionality
        elif hasattr(self, 'using_real_orders') and self.using_real_orders:
            # Skip if we've processed all orders
            if self.current_order_index >= len(self.real_orders):
                return
                
            # Calculate elapsed time in simulation
            elapsed_minutes = current_time
            
            # Get the next order's timestamp
            next_order_time = self.real_orders.iloc[self.current_order_index]['order_push_time']
            
            # Convert real timestamp to simulation minutes
            next_order_minutes = (next_order_time - self.simulation_start_time).total_seconds() / 60
            
            # Check if it's time to add this order
            if elapsed_minutes >= next_order_minutes:
                # Get order details
                order_row = self.real_orders.iloc[self.current_order_index]
                
                # Create order with real data
                order = self.create_order_from_real_data(order_row, current_time)
                self.active_orders.append(order)
                
                # Move to next order
                self.current_order_index += 1
                
                # Log progress occasionally
                if self.current_order_index % 10 == 0:
                    logger.info(f"Processed {self.current_order_index}/{len(self.real_orders)} orders")
        
        # Priority 3: Default Poisson process
        else:
            if current_time >= self.next_order_time:
                self.active_orders.append(self.generate_order(current_time, restaurants))
                self.next_order_time = current_time + np.random.exponential(self.mean_interarrival_time)

    def create_order_from_real_data(self, order_row, current_time):
        """
        Create an Order object using real order data
        """
        # Find the restaurant node in the manager
        restaurant_node = None
        for node in self.location_manager.restaurants:
            if node.id == order_row['poi_id']:
                restaurant_node = node
                break
        
        # If not found, create a new node
        if restaurant_node is None:
            restaurant_node = Node(
                id=int(order_row['poi_id']),
                location=Location(
                    x=float(order_row['sender_lat']/1000000), 
                    y=float(order_row['sender_lng']/1000000)
                )
            )
        
        # Create customer node
        customer_node = Node(
            id=int(order_row['order_id']) + 1000000,  # Add offset to avoid ID collisions
            location=Location(
                x=float(order_row['recipient_lat']/1000000), 
                y=float(order_row['recipient_lng']/1000000)
            )
        )
        
        # Calculate prep time from real data if available
        if pd.notna(order_row['estimate_meal_prepare_time']) and pd.notna(order_row['order_push_time']):
            prep_time = (order_row['estimate_meal_prepare_time'] - order_row['order_push_time']).total_seconds() / 60
        else:
            # Use default if no real data
            prep_time = np.random.gamma(
                shape=(self.mean_prep_time**2) / self.prep_time_var, 
                scale=self.prep_time_var / self.mean_prep_time
            )
        
        # Enforce reasonable prep time (between 1 and 60 minutes)
        prep_time = max(1.0, min(60.0, prep_time))
        
        # Create order with real data
        order = Order(
            id=int(order_row['order_id']),
            request_time=current_time,
            pickup_node_id=restaurant_node,
            delivery_node_id=customer_node,
            deadline=current_time + self.delivery_window,
            ready_time=current_time + prep_time,
            service_time=self.service_time,
        )
        
        return order