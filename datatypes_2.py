from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from enum import Enum


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
    """Represents a delivery order."""

    id: int  # Unique identifier (incrementing)
    request_time: float  # When order was placed
    pickup_node_id: int  # Restaurant node ID
    delivery_node_id: int  # Customer node ID (dynamically created)
    deadline: float  # Delivery deadline
    ready_time: Optional[float]  # When order will be ready at restaurant
    service_time: float  # Time needed for pickup/delivery
    status: str = "pending"  # Order status
    pickup_time: Optional[float] = None
    delivery_time: Optional[float] = None


@dataclass
class NodeVisit:
    """Represents a planned visit to a node."""

    node_id: int
    orders_pickup: Set[int]  # Order IDs to pick up at this node
    orders_delivery: Set[int]  # Order IDs to deliver at this node
    planned_arrival_time: float


@dataclass
class Vehicle:
    """Represents a delivery vehicle."""

    id: int
    initial_location: Location
    current_location: Location
    capacity: int
    current_node_id: Optional[int] = None
    current_orders: Set[int] = None  # Currently carrying these orders
    movement_progress: float = 0.0
    total_travel_time: float = 0.0

    def __post_init__(self):
        if self.current_orders is None:
            self.current_orders = set()


@dataclass
class Route:
    """Represents a planned sequence of node visits."""

    vehicle_id: int
    visits: List[NodeVisit]  # Sequence of nodes to visit
    total_distance: float = 0.0
    total_time: float = 0.0


@dataclass
class State:
    """Represents the current state of the system."""

    time: float
    nodes: Dict[int, Node]  # All nodes in system (static + dynamic)
    orders: Dict[int, Order]  # All active orders
    vehicles: Dict[int, Vehicle]  # All vehicles
    routes: Dict[int, Route]  # Current routes (key: vehicle_id)
    unassigned_orders: Set[int]  # Order IDs not yet assigned

    # Additional tracking
    next_order_id: int = 1001  # For generating new order IDs
    next_dynamic_node_id: int = 5001  # For generating customer node IDs

    def add_order(
        self, pickup_node_id: int, delivery_location: Location, ready_time: float, deadline: float, service_time: float
    ) -> Order:
        """Creates a new order and its associated customer node."""
        # Create new customer node
        customer_node = Node(
            id=self.next_dynamic_node_id, type=NodeType.CUSTOMER, location=delivery_location, is_permanent=False
        )
        self.nodes[customer_node.id] = customer_node
        self.next_dynamic_node_id += 1

        # Create new order
        order = Order(
            id=self.next_order_id,
            request_time=self.time,
            pickup_node_id=pickup_node_id,
            delivery_node_id=customer_node.id,
            deadline=deadline,
            ready_time=ready_time,
            service_time=service_time,
        )
        self.orders[order.id] = order
        self.unassigned_orders.add(order.id)
        self.next_order_id += 1

        return order

    def get_route_statistics(self) -> Dict:
        """Returns statistics about current routes."""
        stats = {
            "total_assigned_orders": sum(len(v.current_orders) for v in self.vehicles.values()),
            "total_unassigned_orders": len(self.unassigned_orders),
            "vehicles_in_use": len([v for v in self.vehicles.values() if v.current_orders]),
            "orders_per_vehicle": {v.id: len(v.current_orders) for v in self.vehicles.values()},
        }
        return stats
