# meituan_data_config.py
import os
import pandas as pd
import numpy as np
import logging
import math
from typing import Dict, Tuple, Optional
from datatypes import Node, Location
from environment.order_generator import OrderGenerator  


logger = logging.getLogger(__name__)

class MeituanDataConfig:
    """Configuration for loading and applying Meituan data to simulations."""
    
    def __init__(
        self, 
        district_id: int, 
        day: str,
        use_restaurant_positions: bool = True,
        use_vehicle_count: bool = True,
        use_vehicle_positions: bool = True,
        use_service_area: bool = True,
        use_deadlines: bool = True,

        # Order generation mode
        order_generation_mode: str = "default",  # "default", "pattern", or "replay"
        temporal_pattern: dict = None,  # For "pattern" mode
        simulation_start_hour: int = None,
        simulation_duration_hours: int = None
    ):
        self.district_id = district_id
        self.day = day
        self.simulation_start_hour = simulation_start_hour
        self.simulation_duration_hours = simulation_duration_hours
        self.use_restaurant_positions = use_restaurant_positions
        self.use_vehicle_count = use_vehicle_count
        self.use_vehicle_positions = use_vehicle_positions
        self.use_service_area = use_service_area
        self.use_deadlines = use_deadlines

        # Data paths
        self.data_dir = os.path.join("data/meituan_data/processed/daily_orders", str(day))
        self.restaurant_file = os.path.join(self.data_dir, f"district_{district_id}_restaurants.csv")
        self.vehicle_file = os.path.join(self.data_dir, f"district_{district_id}_vehicles.csv")
        self.order_file = os.path.join(self.data_dir, f"district_{district_id}_orders.csv")
        
        # Loaded data (initialized as None)
        self.restaurants_df = None
        self.vehicles_df = None
        self.orders_df = None
        
        # Geographic boundaries (initialized as None)
        self.geo_bounds = None
        
        # Validate and load data
        self._validate_data_files()
        self._load_data()
        
        if self.use_service_area:
            self._calculate_geo_bounds()

        # Load and store order data if needed
        self.orders_df = None
        if order_generation_mode == "replay":
            if os.path.exists(self.order_file):
                self.orders_df = pd.read_csv(self.order_file)
                # print(f"PRINTING THE COLUMNS {self.orders_df.columns}")
                # Convert timestamp columns
                timestamp_cols = ['platform_order_time', 'estimate_meal_prepare_time', 
                                 'order_push_time', 'dispatch_time', 'grab_time', 
                                 'fetch_time', 'arrive_time']
                for col in timestamp_cols:
                    if col in self.orders_df.columns:
                        self.orders_df[col] = pd.to_datetime(self.orders_df[col])
            else:
                raise FileNotFoundError(f"Order data file not found: {self.order_file}")
        
        # Store order generation config
        self.order_generation_mode = order_generation_mode
        self.temporal_pattern = temporal_pattern
    
    def _validate_data_files(self) -> None:
        """Validate that required data files exist."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"No data found for day {self.day}")
        
        if self.use_restaurant_positions and not os.path.exists(self.restaurant_file):
            raise FileNotFoundError(f"No restaurant data found for district {self.district_id} on day {self.day}")
        
        # Need vehicle data for either count or positions
        if (self.use_vehicle_count or self.use_vehicle_positions) and not os.path.exists(self.vehicle_file):
            raise FileNotFoundError(f"No vehicle data found for district {self.district_id} on day {self.day}")
    
    def _load_data(self) -> None:
        """Load required data from files."""
        if self.use_restaurant_positions:
            self.restaurants_df = pd.read_csv(self.restaurant_file)
        
        if self.use_vehicle_count or self.use_vehicle_positions:
            self.vehicles_df = pd.read_csv(self.vehicle_file)

    def _calculate_geo_bounds(self) -> None:
        """Calculate geographic boundaries based on restaurant and customer positions."""
        if self.restaurants_df is None:
            raise ValueError("Cannot calculate geo bounds: No restaurant data loaded")
        
        # Start with restaurant boundaries
        min_lat = self.restaurants_df['sender_lat'].min()
        max_lat = self.restaurants_df['sender_lat'].max()
        min_lng = self.restaurants_df['sender_lng'].min()
        max_lng = self.restaurants_df['sender_lng'].max()
        
        # Include customer locations if available
        if hasattr(self, 'orders_df') and self.orders_df is not None:
            min_lat = min(min_lat, self.orders_df['recipient_lat'].min())
            max_lat = max(max_lat, self.orders_df['recipient_lat'].max())
            min_lng = min(min_lng, self.orders_df['recipient_lng'].min())
            max_lng = max(max_lng, self.orders_df['recipient_lng'].max())
                    
        # Add a buffer around the service area (15% instead of 10%)
        lat_range = max_lat - min_lat
        lng_range = max_lng - min_lng
        buffer_lat = lat_range * 0.15
        buffer_lng = lng_range * 0.15
        
        # Update bounds with buffer
        min_lat -= buffer_lat
        max_lat += buffer_lat
        min_lng -= buffer_lng
        max_lng += buffer_lng
        
        # Calculate dimensions in km
        lat_km = (max_lat - min_lat) * 111
        lng_km = (max_lng - min_lng) * 111 * math.cos(math.radians((min_lat + max_lat) / 2))
        
        # Store bounds for coordinate conversion
        self.geo_bounds = {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lng": min_lng,
            "max_lng": max_lng,
            "width_km": lng_km,
            "height_km": lat_km
        }


    def get_service_area_dimensions(self) -> Optional[Tuple[float, float]]:
        """Get dimensions of service area in km."""
        if not self.use_service_area or self.geo_bounds is None:
            return None
        return (self.geo_bounds["width_km"], self.geo_bounds["height_km"])
    
    def get_restaurant_count(self) -> Optional[int]:
        """Get number of restaurants."""
        if not self.use_restaurant_positions or self.restaurants_df is None:
            return None
        return len(self.restaurants_df)
    
    def get_vehicle_count(self) -> Optional[int]:
        """Get number of vehicles."""
        if not self.use_vehicle_positions or self.vehicles_df is None:
            return None
        return len(self.vehicles_df)
    
    def geo_to_sim_coords(self, lat: float, lng: float) -> Tuple[float, float]:
        """Convert geographic coordinates to simulation coordinates."""
        if self.geo_bounds is None:
            raise ValueError("Cannot convert coordinates: No geo bounds calculated")
        
        # Convert from geographic coordinates to normalized simulation coordinates (0-1)
        norm_x = (lng - self.geo_bounds["min_lng"]) / (self.geo_bounds["max_lng"] - self.geo_bounds["min_lng"])
        norm_y = (lat - self.geo_bounds["min_lat"]) / (self.geo_bounds["max_lat"] - self.geo_bounds["min_lat"])
        
        # Scale to simulation dimensions
        sim_x = norm_x * self.geo_bounds["width_km"]
        sim_y = norm_y * self.geo_bounds["height_km"]
        
        return sim_x, sim_y
        
    # In meituan_data_config.py:
    def create_restaurant_nodes(self) -> Optional[list]:
        """Create Node objects for restaurants, preserving original IDs."""
        if not self.use_restaurant_positions or self.restaurants_df is None:
            return None
        
        restaurant_nodes = []
        
        for _, row in self.restaurants_df.iterrows():
            # Use the original poi_id as the restaurant ID
            restaurant_id = row['poi_id']  # This is the key change
            
            sim_x, sim_y = self.geo_to_sim_coords(row['sender_lat'], row['sender_lng'])
            restaurant_nodes.append(
                Node(
                    id=restaurant_id,  # Use original restaurant ID instead of sequential i
                    location=Location(x=sim_x, y=sim_y)
                )
            )
        
        return restaurant_nodes

    def _create_order_from_real_data(self, order_row, current_time, restaurants, geo_bounds):
        restaurant_lat = order_row['sender_lat']
        restaurant_lng = order_row['sender_lng']
        
        # First try exact coordinate match
        coord_key = (restaurant_lat, restaurant_lng)
        restaurant = self.restaurant_coordinate_lookup.get(coord_key)
        
        # If exact match fails, find closest restaurant by coordinates
        if restaurant is None:
            # Find closest restaurant using a small search radius
            closest_dist = float('inf')
            for r in restaurants:
                # Convert restaurant simulation coordinates back to geographic
                # This requires a reverse transformation function
                r_lat, r_lng = sim_to_geo_coords(r.location.x, r.location.y, geo_bounds)
                
                # Calculate distance in geographic space
                dist = ((r_lat - restaurant_lat)**2 + (r_lng - restaurant_lng)**2)**0.5
                if dist < closest_dist:
                    closest_dist = dist
                    restaurant = r

    def create_vehicle_positions(self) -> Optional[Dict[int, Location]]:
        """Create initial positions for vehicles."""
        if not self.use_vehicle_positions or self.vehicles_df is None:
            return None
        
        vehicle_positions = {}
        for i, row in self.vehicles_df.iterrows():
            sim_x, sim_y = self.geo_to_sim_coords(row['grab_lat'], row['grab_lng'])
            vehicle_positions[i] = Location(x=sim_x, y=sim_y)
        
        return vehicle_positions  

    def apply_to_env_params(self, env_params: dict) -> dict:
        """Apply configuration to environment parameters."""
        updated_params = env_params.copy()
        
        # Update service area dimensions
        if self.use_service_area and self.geo_bounds is not None:
            updated_params["service_area_dimensions"] = (
                self.geo_bounds["width_km"], 
                self.geo_bounds["height_km"]
            )

        # Update restaurant count
        if self.use_restaurant_positions and self.restaurants_df is not None:
            updated_params["num_restaurants"] = len(self.restaurants_df)
        
        # Update vehicle count
        if self.use_vehicle_count and self.vehicles_df is not None:
            updated_params["num_vehicles"] = len(self.vehicles_df)
        
        # Override simulation duration if using time window with real orders
        if self.order_generation_mode == "replay" and self.simulation_duration_hours is not None:
            # Convert hours to minutes for simulation duration
            updated_params["simulation_duration"] = self.simulation_duration_hours * 60
            
            # Adjust cooldown duration if necessary to ensure it doesn't exceed simulation duration
            cooldown_ratio = 0.1  # Make cooldown 10% of total simulation
            updated_params["cooldown_duration"] = min(
                updated_params["cooldown_duration"],
                updated_params["simulation_duration"] * cooldown_ratio
            )
        
        return updated_params
    
    def create_order_generator(self, env_params):
        """Create an OrderGenerator based on configuration"""
        if self.order_generation_mode == "default":
            return OrderGenerator(
                mean_interarrival_time=env_params.get("mean_interarrival_time", 2.0),
                service_area_dimensions=env_params.get("service_area_dimensions", (10.0, 10.0)),
                delivery_window=env_params.get("delivery_window", 40.0),
                service_time=env_params.get("service_time", 2.0),
                mean_prep_time=env_params.get("mean_prep_time", 10.0),
                prep_time_var=env_params.get("prep_time_var", 2.0),
                mode="default"
            )
        elif self.order_generation_mode == "pattern":
            return OrderGenerator(
                mean_interarrival_time=env_params.get("mean_interarrival_time", 2.0),
                service_area_dimensions=env_params.get("service_area_dimensions", (10.0, 10.0)),
                delivery_window=env_params.get("delivery_window", 40.0),
                service_time=env_params.get("service_time", 2.0),
                mean_prep_time=env_params.get("mean_prep_time", 10.0),
                prep_time_var=env_params.get("prep_time_var", 2.0),
                mode="pattern",
                temporal_pattern=self.temporal_pattern
            )
        elif self.order_generation_mode == "replay":
            return OrderGenerator(
                mean_interarrival_time=env_params.get("mean_interarrival_time", 2.0),
                service_area_dimensions=env_params.get("service_area_dimensions", (10.0, 10.0)),
                delivery_window=env_params.get("delivery_window", 40.0),
                service_time=env_params.get("service_time", 2.0),
                mean_prep_time=env_params.get("mean_prep_time", 10.0),
                prep_time_var=env_params.get("prep_time_var", 2.0),
                mode="replay",
                real_orders_df=self.orders_df,
                use_real_deadlines=self.use_deadlines
            )
        else:
            raise ValueError(f"Unknown order generation mode: {self.order_generation_mode}")
    
    def apply_to_environment(self, env) -> None:
        """Apply configuration to an initialized environment."""
        # Apply restaurant positions
        if self.use_restaurant_positions:
            restaurant_nodes = self.create_restaurant_nodes()
            if restaurant_nodes:
                env.location_manager.restaurants = restaurant_nodes
        
        # Apply vehicle positions - only if use_vehicle_positions is True
        if self.use_vehicle_positions:
            vehicle_positions = self.create_vehicle_positions()
            if vehicle_positions:
                # Only apply positions for as many vehicles as the environment has
                position_count = 0
                for i, position in vehicle_positions.items():
                    if i < len(env.vehicle_manager.vehicles):
                        # Only set initial location
                        env.vehicle_manager.vehicles[i].initial_location = position
                        position_count += 1
        
        # Store geo bounds and coordinate conversion in environment
        if self.use_service_area and self.geo_bounds is not None:
            env.geo_bounds = self.geo_bounds
            env.geo_to_sim_coords = self.geo_to_sim_coords
    
        # Set up order generator based on mode
        if hasattr(self, 'order_generation_mode') and self.order_generation_mode == "replay":
            if hasattr(self, 'orders_df') and self.orders_df is not None:
                # Create a copy for filtering to avoid modifying the original
                filtered_orders_df = self.orders_df.copy()
                                
                # Apply time window filtering if specified
                if self.simulation_start_hour is not None:
                    try:
                        # Ensure order_push_time is a datetime type
                        if not pd.api.types.is_datetime64_any_dtype(filtered_orders_df['order_push_time']):
                            filtered_orders_df['order_push_time'] = pd.to_datetime(filtered_orders_df['order_push_time'])
                        
                        first_date = filtered_orders_df['order_push_time'].dt.date.min()                        
                        # Create start timestamp for filtering
                        start_time = pd.Timestamp(first_date) + pd.Timedelta(hours=self.simulation_start_hour)                        
                        # Create end timestamp if duration is specified
                        if self.simulation_duration_hours is not None:
                            end_time = start_time + pd.Timedelta(hours=self.simulation_duration_hours)
                        else:
                            end_time = pd.Timestamp.max
                        
                        filtered_orders_df = filtered_orders_df[
                            (filtered_orders_df['order_push_time'] >= start_time) &
                            (filtered_orders_df['order_push_time'] <= end_time)
                        ]
                        
                        # If no orders in time window, warning and fallback
                        if len(filtered_orders_df) == 0:
                            filtered_orders_df = self.orders_df.copy()
                    except Exception as e:
                        logger.warning(f"Error filtering orders by time window: {e}")
                        # Fall back to using all orders
                        filtered_orders_df = self.orders_df.copy()
                
                # Create order generator with filtered data
                env_params = {
                    "mean_interarrival_time": env.order_manager.mean_interarrival_time,
                    "service_area_dimensions": env.location_manager.service_area,
                    "delivery_window": env.order_manager.delivery_window,
                    "service_time": env.order_manager.service_time,
                    "mean_prep_time": env.order_manager.mean_prep_time,
                    "prep_time_var": env.order_manager.prep_time_var
                }
                
                order_generator = OrderGenerator(
                    **env_params,
                    mode="replay",
                    real_orders_df=filtered_orders_df,
                    use_real_deadlines=self.use_deadlines
                )
                
                # Set the generator in OrderManager
                env.order_manager.set_order_generator(order_generator)