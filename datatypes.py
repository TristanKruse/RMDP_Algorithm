# datatypes.py
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Location:
    x: float
    y: float


@dataclass
class Order:
    id: int
    request_time: float
    pickup_location: Location
    delivery_location: Location
    deadline: float
    ready_time: Optional[float]
    service_time: float
    status: str = "pending"
    pickup_time: Optional[float] = None
    delivery_time: Optional[float] = None
    first_postpone_time: Optional[float] = None
    postpone_count: int = 0

    def __str__(self):
        return f"Order {self.id} (requested at {self.request_time}, due by {self.deadline})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Order) and self.id == other.id


@dataclass
class Vehicle:
    id: int
    initial_location: Location
    current_location: Location
    current_destination: Optional[Location] = None
    movement_progress: float = 0.0
    total_travel_time: float = 0.0
    current_phase: Optional[dict] = None

    def __str__(self):
        return f"Vehicle {self.id} at ({self.current_location.x}, {self.current_location.y})"


@dataclass
class State:
    time: float
    orders: List[Order]
    route_plan: List[List[int]]
    unassigned_orders: List[Order]
    vehicles: List[Vehicle]
