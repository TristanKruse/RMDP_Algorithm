# capacity, deadline, pickup_location added to Order class, needed?
# check with paper, ob alles gebraucht und alles drinne

# Defines data structures and their basic properties/behaviors
# datatypes.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass(frozen=True)
class Location:
    """Represents a point in 2D space for restaurants, customers, and vehicles"""

    x: float  # x-coordinate in service area
    y: float  # y-coordinate in service area


@dataclass  # (frozen=True)
class Order:
    """
    Represents a food delivery order in the system.
    Each order has a unique ID, timing information, locations, and service requirements.
    """

    id: int  # Unique identifier for the order
    request_time: float  # Time when order was placed (t_D in paper)
    pickup_location: Location  # Restaurant location (R_D in paper)
    delivery_location: Location  # Customer location (N_D in paper)
    deadline: float  # Time by which delivery must be completed (t_D + t̄)
    ready_time: Optional[float]  # When food is ready at restaurant (ρ(D)), None until revealed
    service_time: float  # Time required at customer location (t_s)
    pickup_service_time: float  # Time required at restaurant for pickup (t_R)
    status: str = "pending"  # New field: pending, picked_up, delivered
    pickup_time: Optional[float] = None  # New field to track pickup time
    delivery_time: Optional[float] = None  # New field to track delivery time
    first_postpone_time: Optional[float] = None  # When order was first postponed
    postpone_count: int = 0  # Number of times postponed

    def __str__(self):
        """String representation for debugging"""
        return f"Order {self.id} (requested at {self.request_time}, due by {self.deadline})"

    def __repr__(self):
        """Representation for debugging"""
        return self.__str__()

    def __hash__(self):
        """Makes Order hashable based on ID for use in sets/dicts"""
        return hash(self.id)

    def __eq__(self, other):
        """Defines equality based on order ID"""
        if not isinstance(other, Order):
            return False
        return self.id == other.id


@dataclass
class Vehicle:
    """
    Represents a delivery vehicle in the system.
    Tracks vehicle location and movement through service area.
    """

    id: int  # Unique identifier for the vehicle
    initial_location: Location  # Starting position (N_V^init)
    current_location: Location  # Current position in service area
    current_destination: Optional[Location] = None  # Target location vehicle is moving towards
    movement_progress: float = 0.0  # Progress towards destination (0.0 to 1.0)
    total_travel_time: float = 0.0  # Total time needed for current movement
    current_phase: Optional[dict] = None  # Add this line to track current order processing phase

    def __str__(self):
        """String representation for debugging"""
        return f"Vehicle {self.id} at ({self.current_location.x}, {self.current_location.y})"


@dataclass
class RouteStop:
    """
    Represents a single stop on a vehicle's route.
    Could be either a pickup at restaurant or delivery to customer.
    """

    location: Location  # Location of this stop
    planned_arrival_time: float  # When vehicle plans to arrive (a_θ)
    is_pickup: bool  # True if this is a restaurant pickup stop
    is_delivery: bool  # True if this is a customer delivery stop
    order: Optional[Order] = None  # Associated order, if any

    def is_feasible(self) -> bool:
        """
        Checks if stop is feasible given time windows.
        For pickups: can't arrive before ready time
        For deliveries: must arrive before deadline
        """
        pass


@dataclass
class Route:
    """
    Represents a vehicle's complete route with sequence of stops.
    Includes both pickups at restaurants and deliveries to customers.
    """

    vehicle_id: int  # ID of vehicle assigned to this route
    stops: List[RouteStop]  # Sequence of stops with planned arrival times

    def get_planned_arrival_times(self) -> List[float]:
        """Returns list of planned arrival times for all stops"""
        return [stop.planned_arrival_time for stop in self.stops]


@dataclass
class State:
    """
    Represents complete state of the delivery system at a decision point.
    Contains all information needed to make assignment decisions.
    """

    time: float  # Current time in system (t_k)
    orders: List[Order]  # All orders in system (Φ_k)
    route_plan: List[Route]  # Current route plan for all vehicles (Θ_k)
    unassigned_orders: List[Order]  # Orders not yet assigned to vehicles
    vehicles: List[Vehicle]

    def get_loading_status(self, order: Order) -> bool:
        """
        Returns whether order is loaded on vehicle (L_D).
        True if order has been picked up but not yet delivered.
        """
        pass

    def get_assigned_vehicle(self, order: Order) -> Optional[int]:
        """
        Returns ID of vehicle assigned to order (V_D).
        Returns None if order is unassigned.
        """
        pass
