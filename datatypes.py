# datatypes.py
from dataclasses import dataclass
from typing import List, Optional


class NodeType(Enum):
    RESTAURANT = "restaurant"  # Static nodes (restaurants)
    CUSTOMER = "customer"  # Dynamic nodes (customer locations)
    WAITING = "waiting"  # Strategic waiting points


@dataclass(frozen=True)
class Location:
    x: float
    y: float


@dataclass
class Node:
    """Represents a location in the network (restaurant, customer, or waiting point)."""

    id: int  # Unique identifier
    type: NodeType  # Type of node
    location: Location  # Physical location
    is_permanent: bool  # True for restaurants/waiting points, False for customers


@dataclass
class Order:
    id: int  # pickup, delivery & reposition
    request_time: float
    pickup_node_id: int  # Restaurant node ID
    delivery_node_id: int  # Customer node ID (dynamically created)
    # pickup_location: Location
    # delivery_location: Location - anstelle dessen jetzt Nodes
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


@dataclass(frozen=True)
class RouteAction:
    type: str  # 'pickup', 'delivery', or 'reposition'
    order_id: int


@dataclass
class State:
    time: float
    orders: List[Order]
    route_plan: List[List[int]]
    unassigned_orders: List[Order]
    vehicles: List[Vehicle]


@dataclass
class State:
    time: float
    orders: List[Order]
    route_plan: List[List[RouteAction]]  # Changed from List[List[int]]
    unassigned_orders: List[Order]
    vehicles: List[Vehicle]
