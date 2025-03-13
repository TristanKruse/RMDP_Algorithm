import random
import numpy as np
import pandas as pd
from environment.meituan_data.utils import geo_to_sim_coords
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Silence matplotlib and PIL debug messages
for logger_name in ["matplotlib", "PIL"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)



class OrderGenerator:
    """
    Generates orders using various strategies
    """
    def __init__(self, mean_interarrival_time=float, service_area_dimensions=tuple,
                 delivery_window=float, service_time=float, mean_prep_time=float, prep_time_var=float,
                 mode="default", temporal_pattern=None, real_orders_df=None, use_real_deadlines=True):
        """
        Initialize the order generator with parameters and generation mode
        
        Args:
            mean_interarrival_time: Average time between orders (for default mode)
            service_area_dimensions: Dimensions of service area (for random locations)
            delivery_window: Time window for delivery
            service_time: Service time at pickup/delivery
            mean_prep_time: Mean order preparation time
            prep_time_var: Variance in preparation time
            mode: "default", "pattern", or "replay"
            temporal_pattern: For pattern mode, defines hourly order rates
            real_orders_df: For replay mode, DataFrame with actual orders
        """
        self.mean_interarrival_time = mean_interarrival_time
        self.service_area = service_area_dimensions
        self.delivery_window = delivery_window
        self.service_time = service_time
        self.mean_prep_time = mean_prep_time
        self.prep_time_var = prep_time_var
        self.use_real_deadlines = use_real_deadlines
        
        # Generation mode
        self.mode = mode
        self.temporal_pattern = temporal_pattern
        self.real_orders_df = real_orders_df
        
        # Validate mode-specific parameters
        if mode == "pattern" and temporal_pattern is None:
            raise ValueError("Pattern mode requires temporal_pattern parameter")
        if mode == "replay" and real_orders_df is None:
            raise ValueError("Replay mode requires real_orders_df parameter")
            
        # Initialize state
        self.next_order_id = 0
        self.next_order_time = 0
        self.simulation_start_time = None
        self.real_order_index = 0
        
        # If using replay mode, preprocess real orders
        if mode == "replay" and real_orders_df is not None:
            self._preprocess_real_orders()

    def _preprocess_real_orders(self):
        """Prepare real order data for replay with actual deadlines"""
        # Sort by order time
        self.real_orders_df = self.real_orders_df.sort_values('order_push_time')
        
        # Store simulation reference time (first order time)
        self.simulation_start_time = self.real_orders_df['order_push_time'].min()
        logger.info(f"Generator simulation start time: {self.simulation_start_time}")
        
        # Check that our delivery window column exists
        if 'delivery_window_minutes' not in self.real_orders_df.columns:
            logger.warning("No 'delivery_window_minutes' column found - will use default delivery window")
        else:
            valid_windows = self.real_orders_df['delivery_window_minutes'].dropna()
            logger.info(f"Found {len(valid_windows)} orders with valid delivery windows")
            logger.info(f"Average delivery window: {valid_windows.mean():.1f} minutes")
        
        # Check that our prep time column exists
        if 'prep_time_minutes' not in self.real_orders_df.columns:
            logger.warning("No 'prep_time_minutes' column found - will use default prep time")
        else:
            valid_prep = self.real_orders_df['prep_time_minutes'].dropna()
            logger.info(f"Found {len(valid_prep)} orders with valid prep times")
            logger.info(f"Average prep time: {valid_prep.mean():.1f} minutes")
        
        # Convert timestamps to simulation time (minutes from start)
        if isinstance(self.simulation_start_time, pd.Timestamp):
            logger.info("Converting datetime timestamps to simulation minutes")
            # Convert order creation time to simulation minutes
            self.real_orders_df['sim_time'] = (self.real_orders_df['order_push_time'] - 
                                            self.simulation_start_time).dt.total_seconds() / 60
        else:
            # Handle numeric timestamps
            logger.info("Converting numeric timestamps to simulation minutes")
            self.real_orders_df['sim_time'] = self.real_orders_df['order_push_time'] - self.simulation_start_time
            
        # Reset index to access by position
        self.real_orders_df = self.real_orders_df.reset_index(drop=True)

        # Log order distribution
        logger.info(f"Loaded {len(self.real_orders_df)} orders spanning {self.real_orders_df['sim_time'].max():.1f} simulation minutes")

    def reset(self):
        """Reset generator state for a new simulation"""
        self.next_order_time = 0
        self.real_order_index = 0
    
    def generate_orders(self, current_time, restaurants, geo_bounds=None):
        """
        Generate new orders based on the current simulation time
        
        Args:
            current_time: Current simulation time (minutes)
            restaurants: List of available restaurants
            geo_bounds: Geographic boundaries for coordinate conversion
            
        Returns:
            List of new Order objects (may be empty)
        """


        if self.mode == "replay" and current_time % 10 == 0:  # Log every 10 minutes
                next_idx = self.real_order_index if self.real_order_index < len(self.real_orders_df) else len(self.real_orders_df)-1
                logger.info(f"Current time: {current_time}, next order at sim_time: {self.real_orders_df['sim_time'].iloc[next_idx] if next_idx >= 0 else 'none'}")
                logger.info(f"Orders processed so far: {self.real_order_index}/{len(self.real_orders_df)}")

        new_orders = []
        
        if self.mode == "default":
            # Default Poisson process
            if current_time >= self.next_order_time:
                new_orders.append(self._generate_random_order(current_time, restaurants))
                self.next_order_time = current_time + np.random.exponential(self.mean_interarrival_time)
                
        elif self.mode == "pattern":
            # Time-varying Poisson process
            if current_time >= self.next_order_time:
                # Calculate hour of day (0-23)
                hour = int((current_time / 60.0) % 24)
                
                # Get rate multiplier for current hour
                hourly_rate = self.temporal_pattern.get(hour, 1.0)
                
                # Adjust interarrival time based on hourly rate
                adjusted_interarrival = self.mean_interarrival_time / max(0.01, hourly_rate)
                
                new_orders.append(self._generate_random_order(current_time, restaurants))
                self.next_order_time = current_time + np.random.exponential(adjusted_interarrival)
                
        elif self.mode == "replay":
            # Replay real orders
            while (self.real_order_index < len(self.real_orders_df) and 
                   self.real_orders_df.loc[self.real_order_index, 'sim_time'] <= current_time):
                
                # Get the real order data
                order_row = self.real_orders_df.iloc[self.real_order_index]
                
                # Create order from real data
                order = self._create_order_from_real_data(order_row, current_time, restaurants, geo_bounds)
                new_orders.append(order)
                
                # Move to next order
                self.real_order_index += 1
        
        return new_orders
    
    def _generate_random_order(self, current_time, restaurants):
        """Generate a random order"""
        from datatypes import Order, Location, Node
        
        # Select random restaurant
        restaurant = random.choice(restaurants)
        
        # Generate random customer location
        area_width, area_height = self.service_area
        customer_loc = Location(x=random.uniform(0, area_width), y=random.uniform(0, area_height))
        customer_node = Node(
            id=self.next_order_id + 1000000,  # Offset to avoid ID conflicts
            location=customer_loc
        )
        
        # Generate preparation time
        prep_time = np.random.gamma(
            shape=(self.mean_prep_time**2) / self.prep_time_var,
            scale=self.prep_time_var / self.mean_prep_time
        )
        
        # Create order
        order = Order(
            id=self.next_order_id,
            request_time=current_time,
            pickup_node_id=restaurant,
            delivery_node_id=customer_node,
            deadline=current_time + self.delivery_window,
            ready_time=current_time + prep_time,
            service_time=self.service_time
        )
        
        self.next_order_id += 1
        return order
        
    def _create_order_from_real_data(self, order_row, current_time, restaurants, geo_bounds):
        """Create an order using real data from the Meituan dataset"""
        from datatypes import Order, Location, Node

        restaurant_lat = order_row['sender_lat']
        restaurant_lng = order_row['sender_lng']
        customer_lat = order_row['recipient_lat']
        customer_lng = order_row['recipient_lng']
        
        logger.debug(f"Restaurant coords: ({restaurant_lat}, {restaurant_lng}), Customer: ({customer_lat}, {customer_lng})")
        
        # Find matching restaurant by ID
        restaurant_id = order_row['poi_id']
        restaurant = next((r for r in restaurants if r.id == restaurant_id), None)
        
        # If no matching restaurant, use closest one
        if restaurant is None:
            # Create simulation coordinates for restaurant
            if geo_bounds:
                # Transform to simulation space
                restaurant_sim_x, restaurant_sim_y = geo_to_sim_coords(restaurant_lat, restaurant_lng, geo_bounds)
                
                # Find closest restaurant in simulation space
                closest_dist = float('inf')
                for r in restaurants:
                    # Calculate distance in simulation space
                    dist = ((r.location.x - restaurant_sim_x)**2 + (r.location.y - restaurant_sim_y)**2)**0.5
                    if dist < closest_dist:
                        closest_dist = dist
                        restaurant = r
                        
                logger.debug(f"Using closest restaurant: {restaurant.id} at ({restaurant.location.x}, {restaurant.location.y})")
            else:
                # If no geo_bounds, just use the first restaurant as fallback
                restaurant = restaurants[0] if restaurants else None
                logger.warning("No geo_bounds, using first restaurant as fallback")
        
        if restaurant is None:
            raise ValueError("No restaurant available to create order")
        
        # Create customer node
        if geo_bounds:
            # Transform geographic customer coordinates to simulation space
            customer_sim_x, customer_sim_y = geo_to_sim_coords(customer_lat, customer_lng, geo_bounds)
            
            # Ensure coordinates are within service area bounds
            width, height = self.service_area
            customer_sim_x = max(0, min(customer_sim_x, width))
            customer_sim_y = max(0, min(customer_sim_y, height))
            
            logger.debug(f"Customer sim coordinates: ({customer_sim_x}, {customer_sim_y})")
            customer_loc = Location(x=customer_sim_x, y=customer_sim_y)
        else:
            # Fallback without geo_bounds - use simple normalization based on service area
            logger.warning("No geo_bounds for coordinate transformation, using basic normalization")
            
            # Create estimated boundaries based on all coordinates seen so far
            if not hasattr(self, '_coord_bounds'):
                self._coord_bounds = {
                    'min_lat': float('inf'), 'max_lat': float('-inf'),
                    'min_lng': float('inf'), 'max_lng': float('-inf')
                }
            
            # Update bounds with current coordinates
            self._coord_bounds['min_lat'] = min(self._coord_bounds['min_lat'], customer_lat, restaurant_lat)
            self._coord_bounds['max_lat'] = max(self._coord_bounds['max_lat'], customer_lat, restaurant_lat)
            self._coord_bounds['min_lng'] = min(self._coord_bounds['min_lng'], customer_lng, restaurant_lng)
            self._coord_bounds['max_lng'] = max(self._coord_bounds['max_lng'], customer_lng, restaurant_lng)
            
            # Use service area dimensions
            width, height = self.service_area
            
            # Normalize coordinates using observed bounds
            norm_x = ((customer_lng - self._coord_bounds['min_lng']) / 
                    (self._coord_bounds['max_lng'] - self._coord_bounds['min_lng']) 
                    if self._coord_bounds['max_lng'] > self._coord_bounds['min_lng'] else 0.5)
            
            norm_y = ((customer_lat - self._coord_bounds['min_lat']) / 
                    (self._coord_bounds['max_lat'] - self._coord_bounds['min_lat'])
                    if self._coord_bounds['max_lat'] > self._coord_bounds['min_lat'] else 0.5)
            
            customer_sim_x = norm_x * width
            customer_sim_y = norm_y * height
            customer_loc = Location(x=customer_sim_x, y=customer_sim_y)
        
        customer_node = Node(
            id=self.next_order_id + 1000000,  # Offset to avoid ID conflicts
            location=customer_loc
        )
        
        # Log final coordinates
        logger.info(f"Order {self.next_order_id} - Restaurant: ({restaurant.location.x}, {restaurant.location.y}), "
                    f"Customer: ({customer_loc.x}, {customer_loc.y}), Service area: {self.service_area}")
        

        # Use real delivery window if available, otherwise fall back to default
        if self.use_real_deadlines and pd.notna(order_row.get('delivery_window_minutes')):
            delivery_window = order_row['delivery_window_minutes']
            # Ensure delivery window is reasonable (between 10 and 120 minutes)
            delivery_window = max(10.0, min(120.0, delivery_window))
        else:
            delivery_window = self.delivery_window
            logging.debug(f"TEST BIG Using default deadlines: {delivery_window}")

        logger.info(f"TEST BIG Using real deadlines: {delivery_window}")


        # Calculate preparation time
        # In your _create_order_from_real_data method
        if pd.notna(order_row.get('prep_time_minutes')):
            # Use the preprocessed prep time
            prep_time = order_row['prep_time_minutes']
        else:
            # Fall back to gamma distribution if prep time is not available
            prep_time = np.random.gamma(
                shape=(self.mean_prep_time**2) / self.prep_time_var,
                scale=self.prep_time_var / self.mean_prep_time
            )

        # Create order
        order = Order(
            id=self.next_order_id,
            request_time=current_time,
            pickup_node_id=restaurant,
            delivery_node_id=customer_node,
            deadline=current_time + delivery_window,
            ready_time=current_time + prep_time,
            service_time=self.service_time
        )
        
        self.next_order_id += 1
        return order