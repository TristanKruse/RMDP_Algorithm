# environment/location_manager.py
from typing import List, Tuple, Optional
import numpy as np
import random
from datatypes import Location


class LocationManager:
    def __init__(
        self,
        service_area_dimensions: Tuple[float, float],
        num_restaurants: int,
        movement_per_step: float,
        downtown_concentration: float,
    ):
        self.service_area = service_area_dimensions
        self.num_restaurants = num_restaurants
        self.movement_per_step = movement_per_step  # km per minute
        self.downtown_concentration = downtown_concentration

        # Calculate downtown parameters once
        self.downtown_center = (self.service_area[0] / 2, self.service_area[1] / 2)
        self.downtown_radius = min(self.service_area) / 4

        # Generate initial restaurant locations
        self.restaurants = self.generate_restaurant_locations()

    def get_travel_time(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate travel time between locations.

        Note: Locations are in grid units (0-1), need to convert to km first
        """
        # Calculate distance in km
        dx = loc2.x - loc1.x
        dy = loc2.y - loc1.y
        distance = np.sqrt(dx * dx + dy * dy)  # Distance in km

        # Return time in minutes
        return distance / self.movement_per_step

    def interpolate_position(self, start: Location, end: Location, progress: float) -> Location:
        """
        Calculate intermediate position between two locations using linear interpolation.
        Takes into account the actual movement speed per minute.
        """
        if not 0 <= progress <= 1:
            raise ValueError("Progress must be between 0 and 1")

        # Calculate direct vector in grid coordinates
        dx = end.x - start.x
        dy = end.y - start.y

        # Linear interpolation along vector
        new_x = start.x + dx * progress
        new_y = start.y + dy * progress

        return Location(x=new_x, y=new_y)

    def generate_restaurant_locations(self) -> List[Location]:
        """
        Generate restaurant locations with concentration in downtown area.

        Returns:
            List of restaurant Location objects
        """
        locations = []

        for _ in range(self.num_restaurants):
            if random.random() < self.downtown_concentration:
                location = self._generate_downtown_location()
            else:
                location = self._generate_uniform_location()
            locations.append(location)

        return locations

    def _generate_downtown_location(self) -> Location:
        """
        Generate a location in the downtown area using polar coordinates.

        Returns:
            Location object in downtown area
        """
        r = random.random() * self.downtown_radius
        theta = random.random() * 2 * np.pi

        x = self.downtown_center[0] + r * np.cos(theta)
        y = self.downtown_center[1] + r * np.sin(theta)

        return Location(x=x, y=y)

    def _generate_uniform_location(self) -> Location:
        """
        Generate a location uniformly in the service area.

        Returns:
            Location object in service area
        """
        return Location(x=random.uniform(0, self.service_area[0]), y=random.uniform(0, self.service_area[1]))

    def get_restaurant_positions(self) -> np.ndarray:
        """
        Get array of restaurant positions for visualization.

        Returns:
            Nx2 array of restaurant coordinates
        """
        return np.array([[r.x, r.y] for r in self.restaurants])

    def is_location_in_service_area(self, location: Location) -> bool:
        """
        Check if location is within service area bounds.

        Args:
            location: Location to check

        Returns:
            True if location is in service area
        """
        return 0 <= location.x <= self.service_area[0] and 0 <= location.y <= self.service_area[1]
