# datatypes.py
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple


@dataclass(frozen=True)
class Location:
    x: float
    y: float


@dataclass
class Node:
    id: int  # Unique identifier
    location: Location  # Physical location


@dataclass
class Order:
    id: int  # pickup, delivery & reposition
    request_time: float 
    pickup_node_id: Node  # Restaurant node ID
    delivery_node_id: Node  # Customer node ID (dynamically created) -> when order is created!!!
    deadline: float
    ready_time: Optional[float]
    service_time: float
    status: str = "pending"
    pickup_time: Optional[float] = None
    delivery_time: Optional[float] = None
    first_postpone_time: Optional[float] = None
    postpone_count: int = 0
    expected_prep_time: float = None  # Expected preparation time when order created
    actual_prep_time: float = None  # Actual preparation time based on when order was ready
    driver_wait_time: float = 0.0  # How long driver waited for the order
    current_estimated_delay: float = 0.0  # Estimated delay based on current route plan


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
class Route:
    vehicle_id: int
    sequence: List[Tuple[int, Set[int], Set[int]]]  # (node_id, pickups, deliveries)
    # Bundled Route Example: 
    # {0: Route(vehicle_id=0, sequence=[(0, {0, 1, 2}, set()), (1000000, set(), {0}), (1000001, set(), {1}), (1000002, set(), {2})], 
    # total_distance=0.0, total_time=0.0)}
    # Sequential Route example:
    # Route(vehicle_id=0, sequence=[(0, {0}, set()), (1000000, set(), {0}), (0, {1}, set()), (1000001, set(), {1}), (0, {2}, set()), (1000002, set(), {2})])
    total_distance: float = 0.0
    total_time: float = 0.0

    def __iter__(self):
        """Make Route iterable by returning iterator over sequence"""
        return iter(self.sequence)

    def get_all_orders(self) -> Set[int]:
        """Get all order IDs in this route"""
        orders = set()
        for _, pickups, deliveries in self.sequence:
            orders.update(pickups)
            orders.update(deliveries)
        return orders

    def copy(self):
        """Create a deep copy of the Route"""
        return Route(
            vehicle_id=self.vehicle_id,
            sequence=deepcopy(self.sequence),
            total_distance=self.total_distance,
            total_time=self.total_time,
        )


@dataclass
class State:
    """Represents the current state of the system."""

    time: float
    orders: List[Order]  # All active orders
    route_plan: Dict[int, Route]  # Current routes (key: vehicle_id)
    unassigned_orders: Set[int]  # Order IDs not yet assigned
    vehicles: Dict[int, Vehicle]  # All vehicles
    nodes: Dict[int, Node]  # Static nodes (restaurants)


# -> den State den dann die Algorithmen bekommen. (wissen nicht alles)
# Nur wenn sich dieser geändert hat muss eine neue entscheidung getroffen werdne
# tk, Dk,Θk,
# tk = point of time
# Dk = order = tD, RD, VD, LD
# tD, time of request
# RD, restaurant associated with the request, Node ggf. -> man könnte darüber nachdenken, ob man hier
# VD, vehicle assigned to the Order/request
# LD, loading status of the request


# maybe better to take directly from the state, since then
# there are less likely to be any discrepancies.
