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

# TODO: Add vehicle preselection here, to save time
# Could select vehicles based on:
# - Distance to restaurant
# - Current vehicle load
# - General direction of travel

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
        max_slack = -float('inf')  # Track best slack for tie-breaking
        
        # Get order info
        order_info = state["unassigned_orders"].get(order_id)
        if not order_info:
            #logger.info(f"  No info found for order {order_id}")
            return None

        # Get locations
        restaurant_node = order_info["pickup_node_id"]
        customer_node = order_info["delivery_node_id"]
        
        vehicles_checked = 0
        vehicles_at_capacity = 0
        vehicles_no_insertions = 0
        
        for vehicle_id, route in route_plan.items():
            vehicles_checked += 1
            
            # Skip vehicles at capacity
            active_orders = self._count_active_orders(route)
            if active_orders >= self.vehicle_capacity:
                vehicles_at_capacity += 1
                # logger.info(f"  Vehicle {vehicle_id} at capacity: {active_orders}/{self.vehicle_capacity}")
                continue
            
            # Log before evaluating
            #logger.info(f"  Evaluating vehicle {vehicle_id}: {active_orders}/{self.vehicle_capacity} orders")
            
            assignment = self._evaluate_vehicle_assignment(
                vehicle_id, route, order_id,
                restaurant_node, customer_node, buffer, state
                # current_best_delay=min_total_cost
            )
            
            if assignment:
                # logger.info(f"  Found assignment with delay: {assignment.delay}")
                # Case 1: Better delay - always take it
                if assignment.delay < min_total_cost:
                    min_total_cost = assignment.delay
                    best_assignment = assignment
                    # Calculate slack for this assignment
                    test_route = Route(
                        vehicle_id=vehicle_id,
                        sequence=assignment.tentative_route,
                        total_distance=0.0,
                        total_time=0.0
                    )
                    max_slack = self.time_calculator._calculate_slack(state, {vehicle_id: test_route})
                    # logger.info(f"  New best assignment: Vehicle {vehicle_id} with delay {min_total_cost} and slack {max_slack}")
                                            
                # Case 2: Same delay - tie-breaking with slack
                elif assignment.delay == min_total_cost:
                    # Calculate slack for this assignment
                    test_route = Route(
                        vehicle_id=vehicle_id,
                        sequence=assignment.tentative_route,
                        total_distance=0.0,
                        total_time=0.0
                    )
                    current_slack = self.time_calculator._calculate_slack(state, {vehicle_id: test_route})
                    # logger.info(f"  Equal delay solution ({min_total_cost}). Current slack: {current_slack}, Best slack: {max_slack}")
                    
                    # If better slack, update best assignment
                    if current_slack > max_slack:
                        #logger.info(f"  Better slack found: {current_slack} > {max_slack}, updating best assignment")
                        max_slack = current_slack
                        best_assignment = assignment
            else:
                vehicles_no_insertions += 1
                # logger.info(f"  No valid insertions found for vehicle {vehicle_id}")
        
        # Log summary
        #logger.info(f"  Vehicles checked: {vehicles_checked}, at capacity: {vehicles_at_capacity}, no insertions: {vehicles_no_insertions}")
        # if best_assignment:
        #     logger.info(f"  Selected vehicle {best_assignment.vehicle_id} with delay {min_total_cost} and slack {max_slack}")
        # else:
        #     logger.info(f"  No suitable vehicle found for order {order_id}")
                
        return best_assignment

    # def _evaluate_vehicle_assignment(
    #     self, 
    #     vehicle_id: int,
    #     route: Route,
    #     order_id: int,
    #     restaurant_node: Node,
    #     customer_node: Node,
    #     buffer: float,
    #     state: dict
    # ) -> Optional[VehicleAssignment]:
    #     """Evaluate all possible insertions of the order for a vehicle and find the best one."""
    #     best_insertion = None
    #     min_delay = float('inf')

    #     # Get all feasible insertion positions
    #     positions = self._get_feasible_insertions(route)
        
    #     # Try each feasible position
    #     for r_pos, c_pos in positions:
    #         # Create test route with new order inserted
    #         test_route = self._create_test_route(
    #             route=route,
    #             order_id=order_id,
    #             restaurant_node=restaurant_node,
    #             customer_node=customer_node,
    #             r_pos=r_pos,
    #             c_pos=c_pos
    #         )
            
    #         # Calculate delay for this insertion
    #         delay = self.time_calculator._calculate_delay(
    #             state=state,
    #             route_plan={vehicle_id: test_route},
    #             buffer=buffer
    #         )

    #         # Update best insertion if this one is better
    #         if delay < min_delay:
    #             min_delay = delay
    #             best_insertion = VehicleAssignment(
    #                 vehicle_id=vehicle_id,
    #                 tentative_route=test_route.sequence,
    #                 delay=delay
    #             )

    #     return best_insertion

    def _evaluate_vehicle_assignment(
        self, 
        vehicle_id: int,
        route: Route,
        order_id: int,
        restaurant_node: Node,
        customer_node: Node,
        buffer: float,
        state: dict
       # current_best_delay: float = float('inf')  # Add this parameter with default value
    ) -> Optional[VehicleAssignment]:
        """Evaluate all possible insertions of the order for a vehicle and find the best one."""
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
            
            # Check if we have an optimal solution already (no delay)
            # If we already have a zero-delay solution, we can only improve on slack
            # So don't bother with expensive calculations for solutions with delay
            # if current_best_delay == 0:
            #     # Do quick check - if this insertion would cause delay, skip
            #     quick_check = self._quick_delay_check(test_route, vehicle_id, state, buffer)
            #     if quick_check > 0:  # If there would be delay, skip
            #         continue
            
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

# Idea to make simulation quicker, but changes results:
    # def _quick_delay_check(self, test_route, vehicle_id, state, buffer):
    #     """Do a quick check to see if this insertion would cause delay.
    #     Returns positive value if delay is likely, 0 if no delay is likely.
    #     """
    #     # Simple check: For each delivery, estimate if it would be late
    #     for node_id, pickups, deliveries in test_route.sequence:
    #         if deliveries:  # Only check delivery stops
    #             for order_id in deliveries:
    #                 # Get order info
    #                 order_info = next((o for o in state["orders"] if o.id == order_id), None)
    #                 if not order_info:
    #                     continue
                                        
    #                 # Very simple estimation - if the order appears late in the route
    #                 # with many stops before it, it might be delayed
    #                 # This is a very basic heuristic and could be improved
    #                 if len(test_route.sequence) > 3:  # If route has many stops
    #                     return 1  # Likely delay
        
    #     return 0  # No obvious delay











    def _is_clearly_worse_than_best(self, test_route, vehicle_id, current_best_delay, state, buffer):
        """Quick check to see if this insertion is clearly worse than current best."""
        # This is a placeholder for your optimization
        # You would implement simple logic to quickly check if this insertion
        # will definitely be worse than current_best_delay
        
        # For now, returning False (never skip) as a placeholder
        return False

    def _get_feasible_insertions(self, route):
        """Get all feasible insertion position pairs for adding two stops to the route."""
        positions = []
        sequence = route.sequence if route.sequence else []
        
        # For empty route, only one possibility: pickup at 0, delivery at 1.
        if not sequence:
            # logger.info("    Empty route: inserting at (0,1)")
            return [(0, 1)]
        
        n = len(sequence)
        
        # Log sequence info
        # logger.info(f"    Sequence length: {n}, evaluating insertion positions...")
        
        # Never insert before the first stop of an existing route
        # This prevents disrupting vehicles that are already moving
        start_pos = 1
        # logger.info(f"    Starting insertion from position {start_pos} (skipping position 0 for non-empty routes)")
        
        insertions_checked = 0
        capacity_failures = 0
        
        for r_pos in range(start_pos, n + 1):  # Always start from position 1 for non-empty routes
            for c_pos in range(r_pos + 1, n + 2):
                insertions_checked += 1
                if self._is_feasible_capacity(sequence, r_pos, c_pos):
                    positions.append((r_pos, c_pos))
                else:
                    capacity_failures += 1
        
        # logger.info(f"    Found {len(positions)} feasible positions out of {insertions_checked} checked. Capacity failures: {capacity_failures}")
        
        # If no positions found, log more details
        # if not positions and sequence:
        #     logger.info(f"    CRITICAL: No feasible insertion positions found for non-empty route!")
        #     logger.info(f"    Route details: {sequence}")
        #     logger.info(f"    Start position: {start_pos}, Sequence length: {n}")
            
        return positions

    def _count_active_orders(self, route):
        if not route.sequence:
            return 0
        
        carried_orders = set()
        max_load = 0
        
        for _, pickups, deliveries in route.sequence:
            carried_orders.update(pickups)  # Add new pickups
            carried_orders.difference_update(deliveries)  # Remove deliveries
            max_load = max(max_load, len(carried_orders))  # Track maximum load
        
        return max_load   

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

