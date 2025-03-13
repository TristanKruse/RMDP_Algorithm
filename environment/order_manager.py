# environment/order_manager.py
from typing import List, Tuple
import numpy as np
from datatypes import Order, Node
import logging
from environment.demand_pattern import DemandPattern
from environment.order_generator import OrderGenerator


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

    def generate_new_orders(self, current_time: float, restaurants: List[Node]) -> None:
        """
        Generate new orders based on simulation time
        """
        # Check if we have an OrderGenerator
        if hasattr(self, 'order_generator'):
            # Get geo_bounds if available for coordinate conversion
            geo_bounds = getattr(self.location_manager, 'geo_bounds', None) if hasattr(self, 'location_manager') else None
            
            # Generate new orders using the OrderGenerator
            new_orders = self.order_generator.generate_orders(current_time, restaurants, geo_bounds)
            self.active_orders.extend(new_orders)
            
            # Sync the next_order_id
            self.next_order_id = self.order_generator.next_order_id
        
        # Otherwise use the basic Poisson process
        else:
            if current_time >= self.next_order_time:
                # Use OrderGenerator for even the basic Poisson process
                if not hasattr(self, '_basic_generator'):
                    # Create basic OrderGenerator for default generation
                    self._basic_generator = OrderGenerator(
                        mean_interarrival_time=self.mean_interarrival_time,
                        service_area_dimensions=self.service_area,
                        delivery_window=self.delivery_window,
                        service_time=self.service_time,
                        mean_prep_time=self.mean_prep_time,
                        prep_time_var=self.prep_time_var
                    )
                
                # Generate orders and extend active_orders list with any new orders
                new_orders = self._basic_generator.generate_orders(current_time, restaurants)
                if new_orders:  # Check that orders were actually generated
                    self.active_orders.extend(new_orders)
                    
                    # Sync the next_order_id
                    self.next_order_id = self._basic_generator.next_order_id
                    
                # Set next order time regardless of whether an order was generated
                self.next_order_time = current_time + np.random.exponential(self.mean_interarrival_time)

    def reset(self) -> None:
        """Reset the order manager state"""
        self.active_orders = []
        self.next_order_id = self.next_order_time = 0
        
        # Reset OrderGenerator if available
        if hasattr(self, 'order_generator'):
            self.order_generator.reset()
        if hasattr(self, '_basic_generator'):
            self._basic_generator.reset()

    def get_active_orders(self) -> List[Order]:
        return self.active_orders

    def cleanup_delivered_orders(self):
        self.active_orders = [order for order in self.active_orders if order.status != "delivered"]

    def set_real_orders(self, orders_df):
        """
        Set real orders from dataset to be used instead of generating random orders
        
        Parameters:
        -----------
        orders_df : pandas.DataFrame
            DataFrame containing real order data
        """
        # Create an OrderGenerator in replay mode
        self.order_generator = OrderGenerator(
            mean_interarrival_time=self.mean_interarrival_time,
            service_area_dimensions=self.service_area,
            delivery_window=self.delivery_window,
            service_time=self.service_time,
            mean_prep_time=self.mean_prep_time,
            prep_time_var=self.prep_time_var,
            mode="replay",
            real_orders_df=orders_df
        )
        
        # Give the generator access to the location manager if available
        if hasattr(self, 'location_manager'):
            self.order_generator.location_manager = self.location_manager
        
        logger.info(f"Set up OrderGenerator in replay mode with {len(orders_df)} real orders")

    def set_order_generator(self, generator):
        """Set custom order generator"""
        self.order_generator = generator
        # Reset existing order generation state
        self.next_order_time = 0
