from typing import List, Tuple
import numpy as np
import random
from datatypes import Location, Node


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
        self.movement_per_step = movement_per_step
        self.downtown_concentration = downtown_concentration
        self.downtown_center = (self.service_area[0] / 2, self.service_area[1] / 2)
        self.downtown_radius = min(self.service_area) / 4
        self.restaurants = self.generate_restaurant_locations()

    def get_travel_time(self, loc1: Location, loc2: Location) -> float:
        dx = loc2.x - loc1.x
        dy = loc2.y - loc1.y
        return np.sqrt(dx * dx + dy * dy) / self.movement_per_step

    def interpolate_position(self, start: Location, end: Location, progress: float) -> Location:
        if not 0 <= progress <= 1:
            raise ValueError("Progress must be between 0 and 1")
        return Location(x=start.x + (end.x - start.x) * progress, y=start.y + (end.y - start.y) * progress)

    def generate_restaurant_locations(self) -> List[Node]:
        return [
            Node(
                id=i,
                location=(
                    self._generate_downtown_location()
                    if random.random() < self.downtown_concentration
                    else self._generate_uniform_location()
                ),
            )
            for i in range(self.num_restaurants)
        ]

    def _generate_downtown_location(self) -> Location:
        r = random.random() * self.downtown_radius
        theta = random.random() * 2 * np.pi
        return Location(x=self.downtown_center[0] + r * np.cos(theta), y=self.downtown_center[1] + r * np.sin(theta))

    def _generate_uniform_location(self) -> Location:
        return Location(x=random.uniform(0, self.service_area[0]), y=random.uniform(0, self.service_area[1]))

    def get_restaurant_positions(self) -> np.ndarray:
        return np.array([[r.location.x, r.location.y] for r in self.restaurants])
