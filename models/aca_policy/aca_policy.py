# rmdp_solver.py
# Implements the actual solving algorithm and solution logic
from typing import List, Tuple, Set, Dict
from datatypes import Order, State, Route
from .route_utils import RouteUtils
from .vehicle_ops import VehicleOperations
from .postponement import PostponementHandler
from .time_utils import TimeCalculator

class ACA:
    """
    Solver for the Restaurant Meal Delivery Problem (RMDP).

    This solver implements a sophisticated algorithm that:
    1. Considers multiple possible sequences of order assignments
    2. Allows for strategic postponement of orders
    3. Uses time buffers to handle uncertainties in delivery times

    The algorithm balances immediate service (quick delivery) with future flexibility
    (ability to handle upcoming orders efficiently).
    """
    def __init__(
        self,
        # Core algorithm parameters
        buffer: float,
        max_postponements: int,
        max_postpone_time: float,
        movement_per_step: float,
        location_manager,  # Add this new parameter
        # Vehicle parameters
        vehicle_capacity: int = 3,
        # Time parameters
        service_time: float = 2.0,
        mean_prep_time: float = 10.0,
        prep_time_var: float = 2.0,
        delay_normalization_factor: float = 10.0,
    ):
        self.buffer = buffer
        self.max_postponements = max_postponements
        self.max_postpone_time = max_postpone_time
        self.location_manager = location_manager  # Store it

        # Initialize utility classes 
        self.route_utils = RouteUtils(vehicle_capacity)
        self.time_calculator = TimeCalculator(
            mean_prep_time=mean_prep_time,
            prep_time_var=prep_time_var,
            service_time=service_time,
            delay_normalization_factor=delay_normalization_factor,
            location_manager=location_manager  # Pass to TimeCalculator
        )

        # Initialize component handlers with relevant parameters
        self.postponement = PostponementHandler(
            max_postponements=max_postponements,
            max_postpone_time=max_postpone_time,
        )

        self.vehicle_ops = VehicleOperations(
            service_time=service_time,
            movement_per_step=movement_per_step,
            vehicle_capacity=vehicle_capacity,
            mean_prep_time=mean_prep_time,
            prep_time_var=prep_time_var,
            delay_normalization_factor=delay_normalization_factor,
            location_manager=location_manager  # Pass to VehicleOperations
        )

    def solve(self, state_dict: dict) -> Tuple[Dict[int, Route], Set[int]]:
        """ACA Algorithm to solve the RMDP.
        
        This algorithm:
        1. Considers multiple possible sequences of order assignments
        2. Allows for strategic postponement of orders
        3. Uses time buffers to handle uncertainties in delivery times
        
        Input parameters:
            state (S): Current system state
            time (t in state): Current time point
            route plan (Θ in state): Current route plan
            unassigned orders ($o in state): Orders not yet assigned
            buffer (b as class parameter): Time buffer for handling uncertainties
            max_postponements (pmax as class parameter): Maximum number of orders that can be postponed
            max_postpone_time (tpmax as class parameter): Maximum time an order can be postponed
        
        The algorithm follows these steps:
        1-4. Initialization
            x ← ∅ // Best decision
            delay ← bigM // Delay
            slack ← 0 // Slack
        5-7. Start assignment procedure
            forall ̂$ ordered set of $o // Process all potential sequences
        8-10. For each sequence
            ̂Θ ← Θ // Create candidate route plan
            ̂P ← ∅ // Initialize set of postponements
            forall D ∈ ̂$ // Process all orders in sequence
            
        Output:
            Tuple containing:
            - Route plan as dictionary mapping vehicle IDs to Routes
            - Set of orders chosen for postponement
        """
        # Get set of valid order IDs at start
        active_order_ids = set(state_dict["unassigned_orders"].keys())

        # Initialize route plan dictionary
        route_plan = {}
        for vehicle_id, route in state_dict["route_plan"].items():
            print(f"Processing route for vehicle {vehicle_id}")
            print(f"Original sequence: {route.sequence}")
            
            # Each sequence element is a tuple (node_id, pickups, deliveries)
            cleaned_sequence = []
            for node_id, pickups, deliveries in route.sequence:
                # Filter pickups and deliveries to only include active orders
                valid_pickups = {pid for pid in pickups if pid in active_order_ids}
                valid_deliveries = {did for did in deliveries if did in active_order_ids}
                
                # Only keep stops that have valid orders
                if valid_pickups or valid_deliveries:
                    cleaned_sequence.append((node_id, valid_pickups, valid_deliveries))
            
            print(f"Cleaned sequence: {cleaned_sequence}")
            
            route_plan[vehicle_id] = Route(
                vehicle_id=vehicle_id,
                sequence=cleaned_sequence,
                total_distance=0.0,
                total_time=0.0
            )

        print(f"Starting solve with {len(state_dict['unassigned_orders'])} unassigned orders")

        # Handle empty state or no unassigned orders (Step 1: Initialization)
        if not state_dict["unassigned_orders"]:
            return route_plan, set()

        # Initialize tracking variables (Steps 1-4: Initialization)
        best_decision = {k: v.copy() for k, v in route_plan.items()}  # x ← ∅
        best_delay = float("inf")  # delay ← bigM
        best_slack = 0  # slack ← 0
        best_postponed = set()  # Initialize empty set for postponed orders
        
        # Step 5-7: Generate order sequences
        if isinstance(state_dict["unassigned_orders"], dict):
            order_items = list(state_dict["unassigned_orders"].items())
        else:
            order_items = state_dict["unassigned_orders"]

        # Generate order sequences with the correct format
        order_sequences = self.route_utils._generate_order_sequences(order_items)

        # Steps 8-10: Process each sequence
        for sequence in order_sequences:
            # Initialize candidate solution
            candidate_route = {k: v.copy() for k, v in route_plan.items()}  # ̂Θ ← Θ
            candidate_postponed = set()  # ̂P ← ∅

            # Process each order in sequence (forall D ∈ ̂$)
            for order_id, order_info in sequence:
                # Check postponement
                should_postpone = self.postponement.evaluate_postponement(
                    postponed=candidate_postponed,
                    route_plan=candidate_route,
                    order_id=order_id,
                    current_time=state_dict["time"],
                    state=state_dict,
                )

                if should_postpone:
                    candidate_postponed.add(order_id)
                    continue

                # Find best vehicle assignment
                assignment = self.vehicle_ops.find_vehicle(
                    candidate_route, order_id, self.buffer, state_dict
                )
                
                # if assignment is None:
                #     print(f"Warning: Could not find valid assignment for order {order_id}")
                #     continue

                # Update route for assigned vehicle
                candidate_route[assignment.vehicle_id] = Route(
                    vehicle_id=assignment.vehicle_id,
                    sequence=assignment.tentative_route,
                    total_distance=0.0,
                    total_time=0.0
                )

            # Evaluate solution
            current_delay, _ = self.time_calculator._calculate_delay(state_dict, candidate_route)
            current_slack = self.time_calculator._calculate_slack(state_dict, candidate_route)

            # Update best solution if better
            if current_delay < best_delay or (current_delay == best_delay and current_slack < best_slack):
                best_decision = {k: v.copy() for k, v in candidate_route.items()}
                best_delay = current_delay
                best_slack = current_slack
                best_postponed = candidate_postponed.copy()

        print(f"Solver postponing {len(best_postponed)} orders")

        # Remove postponed orders from final routes
        final_routes = {}
        for vehicle_id, route in best_decision.items():
            # Filter the sequence to exclude postponed orders
            new_sequence = []
            for node_id, pickups, deliveries in route.sequence:
                # Remove postponed orders from pickups and deliveries
                new_pickups = pickups - best_postponed
                new_deliveries = deliveries - best_postponed
                
                # Only keep stops that still have orders
                if new_pickups or new_deliveries:
                    new_sequence.append((node_id, new_pickups, new_deliveries))
            
            final_routes[vehicle_id] = Route(
                vehicle_id=vehicle_id,
                sequence=new_sequence,
                total_distance=route.total_distance,
                total_time=route.total_time
            )

        return final_routes, best_postponed