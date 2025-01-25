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
    def __init__(self, num_vehicles: int, service_area_dimensions: Tuple[float, float], vehicle_capacity: int):
        self.num_vehicles = num_vehicles
        self.service_area = service_area_dimensions
        self.vehicle_capacity = vehicle_capacity
        self.vehicles = [
            Vehicle(
                id=i,
                initial_location=Location(
                    x=random.uniform(0, service_area_dimensions[0]), y=random.uniform(0, service_area_dimensions[1])
                ),
                current_location=Location(
                    x=random.uniform(0, service_area_dimensions[0]), y=random.uniform(0, service_area_dimensions[1])
                ),
            )
            for i in range(num_vehicles)
        ]
        self.initial_positions = {v.id: v.initial_location for v in self.vehicles}

    def reset(self) -> None:
        for vehicle in self.vehicles:
            vehicle.current_location = self.initial_positions[vehicle.id]

    def get_vehicle_by_id(self, vehicle_id: int) -> Optional[Vehicle]:
        return next((v for v in self.vehicles if v.id == vehicle_id), None)

    def get_vehicle_positions(self) -> np.ndarray:
        return np.array([[v.current_location.x, v.current_location.y] for v in self.vehicles])

    def get_vehicle_load(self, vehicle_id: int) -> int:
        vehicle = self.get_vehicle_by_id(vehicle_id)
        return len(getattr(vehicle, "current_orders", [])) if vehicle else 0

    def is_vehicle_available(self, vehicle_id: int) -> bool:
        return self.get_vehicle_load(vehicle_id) < self.vehicle_capacity

    def reposition_idle_vehicles(self, state: State, location_manager) -> None:
        # Get all possible restaurant locations by combining pickup locations and restaurant locations
        restaurant_locations = {order.pickup_location for order in state.orders} | set(location_manager.restaurants)

        # Track which restaurants are already targeted by other vehicles
        occupied_locations = {v.current_destination for v in self.vehicles if v.current_destination}

        for vehicle in self.vehicles:
            # Only reposition vehicles that are idle and not already heading somewhere
            if not self._is_vehicle_busy(vehicle, state.route_plan) and not vehicle.current_destination:
                # Find closest available restaurant using min() with generator
                nearest_restaurant = min(
                    (
                        (location_manager.get_travel_time(vehicle.current_location, loc), loc)
                        for loc in restaurant_locations
                        if loc not in occupied_locations
                    ),
                    key=lambda x: x[0],
                    default=(None, None),
                )[1]

                if nearest_restaurant:
                    # Update vehicle with new destination and initialize movement tracking
                    vehicle.current_destination = nearest_restaurant
                    vehicle.movement_progress = 0.0
                    vehicle.total_travel_time = location_manager.get_travel_time(
                        vehicle.current_location, nearest_restaurant
                    )
                    occupied_locations.add(nearest_restaurant)

    def _is_vehicle_busy(self, vehicle: Vehicle, route_plan: List[List[int]]) -> bool:
        return route_plan[vehicle.id] or (hasattr(vehicle, "current_phase") and vehicle.current_phase)
