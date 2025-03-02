# Cheapest Insertion implementation
# 12 & 13, finds and assignes vehicles 
from dataclasses import dataclass
from .time_utils import TimeCalculator
from typing import List, Optional, Tuple, Set
from datatypes import Route, Node
import logging

@dataclass
class VehicleAssignment:
    vehicle_id: int
    tentative_route: List[Tuple[int, Set[int], Set[int]]]  # Correct type for route sequence
    delay: float

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_location(obj):
    """
    Recursively extract a Location from an object.
    If the object already has an 'x' attribute, we assume it’s a Location.
    Otherwise, if it has a 'location' attribute, we return get_location(obj.location).
    """
    if hasattr(obj, 'x'):
        return obj
    elif hasattr(obj, 'location'):
        return get_location(obj.location)
    else:
        return obj

class VehicleOperations:
    def __init__(
        self,
        service_time: float,
        vehicle_capacity: int,
        mean_prep_time: float,
        location_manager,
        delivery_window: float,
    ):
        self.service_time = service_time
        self.vehicle_capacity = vehicle_capacity
        self.mean_prep_time = mean_prep_time
        self.location_manager = location_manager
        self.time_calculator = TimeCalculator(
            delivery_window=delivery_window,
            mean_prep_time=mean_prep_time,
            service_time=service_time,
            location_manager=location_manager
        )
        # Cache for travel costs
        self.travel_costs = {}

    def find_vehicle(self, route_plan: dict, order_id: int, buffer: float, state: dict) -> Optional[VehicleAssignment]:
        """Find best vehicle for order insertion using cheapest insertion."""
        # logger.info(f"Finding vehicle for order {order_id}")
        best_assignment = None
        min_total_cost = float('inf')
        
        # Get order info
        order_info = state["unassigned_orders"].get(order_id)
        if not order_info:
            # logger.error(f"No info found for order {order_id} in unassigned_orders")
            return None

        # Get locations
        restaurant_node = order_info["pickup_node_id"]
        customer_node = order_info["delivery_node_id"]
        
        # TODO: Add vehicle preselection here, to save time
        # Could select vehicles based on:
        # - Distance to restaurant
        # - Current vehicle load
        # - General direction of travel
        
        for vehicle_id, route in route_plan.items():
            # Skip vehicles at capacity
            if self._count_active_orders(route) >= self.vehicle_capacity:
                # logger.info(f"Vehicle {vehicle_id} at capacity, skipping")
                continue
        
            # Inside the vehicle loop, before evaluating assignment
            # logger.info(f"Evaluating vehicle {vehicle_id} for order {order_id}")
            # logger.info(f"Current vehicle load: {self._count_active_orders(route)}/{self.vehicle_capacity}")
            
            assignment = self._evaluate_vehicle_assignment(
                vehicle_id, route, order_id,
                restaurant_node, customer_node, buffer, state
            )
            
            # check if makees sense.
            if assignment and assignment.delay < min_total_cost:
                min_total_cost = assignment.delay
                best_assignment = assignment
                if min_total_cost == 0:  # Perfect solution found
                    break
        
            if assignment:
                # logger.info(f"Found assignment with delay: {assignment.delay}")
                if assignment.delay < min_total_cost:
                    min_total_cost = assignment.delay
                    best_assignment = assignment
                    # logger.info(f"New best assignment: Vehicle {vehicle_id} with delay {min_total_cost}")
            #else:
                # logger.info(f"No valid assignment found for vehicle {vehicle_id}")
        
        # After finding best assignment
        #if best_assignment:
            # logger.info(f"Selected vehicle {best_assignment.vehicle_id} for order {order_id}")
        #else:
            # logger.info(f"No suitable vehicle found for order {order_id}") 
                   
        return best_assignment

    def _evaluate_vehicle_assignment(
        self, 
        vehicle_id: int,
        route: Route,
        order_id: int,
        restaurant_node: Node,
        customer_node: Node,
        buffer: float,
        state: dict
    ) -> Optional[VehicleAssignment]:
        """Evaluate all possible insertions for a vehicle and find the best one."""
        best_insertion = None
        min_delay = float('inf')

        # Get all feasible insertion positions
        positions = self._get_feasible_insertions(route)
        
        # Try each feasible position
        for r_pos, c_pos in positions:
            # Create test route with new order inserted
            test_route = self._create_test_route(
                route=route,
                order_id=order_id,
                restaurant_node=restaurant_node,
                customer_node=customer_node,
                r_pos=r_pos,
                c_pos=c_pos
            )
            
            # Calculate delay for this insertion
            delay = self.time_calculator._calculate_delay(
                state=state,
                route_plan={vehicle_id: test_route},
                buffer=buffer
            )

            # Update best insertion if this one is better
            if delay < min_delay:
                min_delay = delay
                best_insertion = VehicleAssignment(
                    vehicle_id=vehicle_id,
                    tentative_route=test_route.sequence,
                    delay=delay
                )

        return best_insertion

    def _get_feasible_insertions(self, route):
        """Get all feasible insertion position pairs for adding two stops to the route."""
        positions = []
        sequence = route.sequence if route.sequence else []
        
        # For empty route, only one possibility: pickup at 0, delivery at 1.
        if not sequence:
            return [(0, 1)]
        
        n = len(sequence)
        
        # Never insert before the first stop of an existing route
        # This prevents disrupting vehicles that are already moving
        start_pos = 1
        
        for r_pos in range(start_pos, n + 1):  # Always start from position 1 for non-empty routes
            for c_pos in range(r_pos + 1, n + 2):
                if self._is_feasible_capacity(sequence, r_pos, c_pos):
                    positions.append((r_pos, c_pos))
        return positions

    def _count_active_orders(self, route):
        if not route.sequence:
            return 0
        carried_orders = set()
        return max((len(carried_orders.union(pickups) - deliveries) 
                for _, pickups, deliveries in route.sequence), 
                default=0)

    def _is_feasible_capacity(self, sequence, r_pos, c_pos):
        max_load = 0
        current_load = 0
        
        for i, (_, pickups, deliveries) in enumerate(sequence):
            if i == r_pos:
                current_load += 1
            current_load += len(pickups)
            current_load -= len(deliveries)
            if i == c_pos:
                current_load -= 1
            max_load = max(max_load, current_load)

        return max_load <= self.vehicle_capacity

    def _create_test_route(self, route, order_id, restaurant_node, customer_node, r_pos, c_pos):
        new_route = route.copy()
        new_route.sequence.insert(r_pos, (restaurant_node.id, {order_id}, set()))
        new_route.sequence.insert(c_pos, (customer_node.id, set(), {order_id}))
        return new_route

