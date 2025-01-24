# environment/vehicle_manager.py
from typing import List, Tuple, Optional, Set
import random
import numpy as np
from datatypes import Vehicle, Location, State
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger instance
logger = logging.getLogger(__name__)


class VehicleManager:
    """Manages vehicles and their states for the restaurant delivery environment."""

    def __init__(self, num_vehicles: int, service_area_dimensions: Tuple[float, float], vehicle_capacity: int):
        """
        Initialize the vehicle manager.

        Args:
            num_vehicles: Number of vehicles to manage
            service_area_dimensions: Dimensions of service area (width, height)
            vehicle_capacity: Maximum number of orders a vehicle can carry
        """
        self.num_vehicles = num_vehicles
        self.service_area = service_area_dimensions
        self.vehicle_capacity = vehicle_capacity
        self.vehicles = self.initialize_vehicles()
        self.initial_positions = {v.id: v.initial_location for v in self.vehicles}

    def initialize_vehicles(self) -> List[Vehicle]:
        """
        Initialize vehicles with random starting locations.

        Returns:
            List of initialized vehicles
        """
        vehicles = []
        for i in range(self.num_vehicles):
            location = Location(x=random.uniform(0, self.service_area[0]), y=random.uniform(0, self.service_area[1]))
            # Add tracking logger.info for vehicle initialization
            # logger.info(f"\n[Vehicle {i}] Initialized")
            # logger.info(f"[Vehicle {i}] Starting location: ({location.x:.2f}, {location.y:.2f})")
            vehicles.append(Vehicle(id=i, initial_location=location, current_location=location))
        return vehicles

    def reset(self) -> None:
        """Reset all vehicles to their initial positions."""
        for vehicle in self.vehicles:
            vehicle.current_location = self.initial_positions[vehicle.id]

    def get_vehicle_by_id(self, vehicle_id: int) -> Optional[Vehicle]:
        return next((v for v in self.vehicles if v.id == vehicle_id), None)

    def update_vehicle_location(self, vehicle_id: int, new_location: Location) -> bool:
        vehicle = self.get_vehicle_by_id(vehicle_id)
        if vehicle:
            vehicle.current_location = new_location
            return True
        return False

    def get_vehicle_positions(self) -> np.ndarray:
        """Get current positions of all vehicles."""
        return np.array([[v.current_location.x, v.current_location.y] for v in self.vehicles])

    def get_vehicle_load(self, vehicle_id: int) -> int:
        """Get current load (number of orders) for a vehicle.

        Args:
            vehicle_id: ID of vehicle to check

        Returns:
            Current number of orders assigned to vehicle
        """
        vehicle = self.get_vehicle_by_id(vehicle_id)
        if vehicle:
            return len(vehicle.current_orders) if hasattr(vehicle, "current_orders") else 0
        return 0

    def is_vehicle_available(self, vehicle_id: int) -> bool:
        """Check if vehicle has capacity for more orders."""
        return self.get_vehicle_load(vehicle_id) < self.vehicle_capacity

    # ------------------------------------------------------------------------------------

    def reposition_idle_vehicles(self, state: State, location_manager) -> None:
        """
        Move idle vehicles to nearest unoccupied restaurant.

        Args:
            state: Current system state
            location_manager: LocationManager instance for distance calculations
        """
        # Get all restaurant locations from active orders and available restaurants
        restaurant_locations = {order.pickup_location for order in state.orders}
        restaurant_locations.update(location_manager.restaurants)

        # Track occupied locations
        occupied_locations = set()
        for vehicle in self.vehicles:
            if vehicle.current_destination:
                occupied_locations.add(vehicle.current_destination)

        # Find and assign repositioning for idle vehicles
        for vehicle in self.vehicles:
            # Only assign new destination if vehicle doesn't already have one
            if not self._is_vehicle_busy(vehicle, state.route_plan) and not vehicle.current_destination:
                nearest_restaurant = self._find_nearest_unoccupied_restaurant(
                    vehicle.current_location, restaurant_locations, occupied_locations, location_manager
                )

                if nearest_restaurant:
                    # Update vehicle destination
                    vehicle.current_destination = nearest_restaurant
                    vehicle.movement_progress = 0.0  # Initialize movement tracking
                    vehicle.total_travel_time = location_manager.get_travel_time(
                        vehicle.current_location, vehicle.current_destination
                    )
                    occupied_locations.add(nearest_restaurant)

    def _is_vehicle_busy(self, vehicle: Vehicle, route_plan: List[List[int]]) -> bool:
        """Check if vehicle is currently busy with orders."""
        return route_plan[vehicle.id] or (hasattr(vehicle, "current_phase") and vehicle.current_phase)

    def _find_nearest_unoccupied_restaurant(
        self,
        vehicle_location: Location,
        restaurant_locations: Set[Location],
        occupied_locations: Set[Location],
        location_manager,
    ) -> Optional[Location]:
        """Find nearest available restaurant."""
        available_restaurants = []

        for loc in restaurant_locations:
            if loc not in occupied_locations:
                travel_time = location_manager.get_travel_time(vehicle_location, loc)
                available_restaurants.append((travel_time, loc))

        if available_restaurants:
            return min(available_restaurants, key=lambda x: x[0])[1]
        return None
