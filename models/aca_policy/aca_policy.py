# rmdp_solver.py
# Implements the actual solving algorithm and solution logic
from typing import List, Tuple, Set
from datatypes import Order, State
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

    def solve(self, state_dict: dict) -> Tuple[List[List[int]], Set[int]]:
        """ACA Algorithm to solve the RMDP
        Input parameters:
            state (S)
            time (t in state)
            route plan (Θ in state)
            unassigned orders ($o in state)
            buffer (b as class parameter)
            max_postponements (pmax as class parameter)
            max_postpone_time (tpmax as class parameter)
        Output:
            Tuple containing:
            - Final route plan (list of lists, each inner list represents a vehicle's route)
            - Set of orders chosen for postponement
        """

        # Get set of valid order IDs at start
        active_order_ids = set(state_dict["unassigned_orders"].keys())

        # Clean up route plan - remove any invalid orders
        cleaned_route_plan = []
        for route in state_dict["route_plan"].values():
            cleaned_route = [order_id for order_id in route.sequence if order_id in active_order_ids]
            cleaned_route_plan.append(cleaned_route)

        print(f"Starting solve with {len(state_dict['unassigned_orders'])} unassigned orders")

        # Update state's route plan
        route_plan = cleaned_route_plan

        print(f"Starting solve with {len(state_dict['unassigned_orders'])} unassigned orders")
        # Handle empty state or no unassigned orders then no need to run the algorithm
        if not state_dict["unassigned_orders"]:
            return route_plan, set()
        # Pseudo code: 1-4 Initialization
        # x ← ∅ // Best decision
        # delay ← bigM // Delay
        # slack ← 0 // Slack
        # Initialize best route plan found so far
        best_decision = route_plan.copy()
        # Initialize best delay with infinity (bigM in paper notation)
        best_delay = float("inf")
        # Initialize best slack time found
        best_slack = 0
        # Initialize set of orders that will be postponed
        best_postponed = set()
        postponed_orders = set()  # stores the overall solution for the best sequence
        # Pseudo code: 5-7 Start assignment procedure and get order sequences
        # forall ̂$ ordered set of $o // All potential sequences
        # Generate order sequences (n! sequences)
        order_sequences = self.route_utils._generate_order_sequences(
            list(state_dict["unassigned_orders"].items())  # Convert dict to list
        )
        for sequence in order_sequences:
            # Pseudo code: 8-10
            # ̂Θ ← Θ // Candidate route plan
            # ̂P ← ∅ // Set of postponements
            # forall D ∈ ̂$ // All orders in sequence
            # existing route plan
            candidate_route = route_plan.copy()  # ̂Θ
            candidate_postponed = set()  # ̂P, stores solution for current sequence
            # Process each order in sequence
            for order_id, order_info in sequence:  # forall D ∈ ̂$

                # Line 14: Check if order should be postponed using the existing postponement handler
                should_postpone = self.postponement.evaluate_postponement(
                    postponed=candidate_postponed,
                    route_plan=candidate_route,
                    order=order_id,  # Now passing ID instead of Order object
                    current_time=state_dict["time"],
                    state=state_dict,
                )

                if should_postpone:
                    candidate_postponed.add(order_id)
                    continue

                # Find best vehicle (line 12) & line 13
                # Assignement is handled with in the find_vehicle method
                # Find best vehicle and get updated route (lines 12-13)
                assignment = self.vehicle_ops.find_vehicle(
                    candidate_route, order_id, self.buffer, state_dict
                )
                
                if assignment is None:
                    print(f"Warning: Could not find valid assignment for order {order_id}")
                    continue

                # Update route for assigned vehicle
                # Create candidate decision (line 17)
                candidate_route[assignment.vehicle_id] = assignment.tentative_route

            # After processing all orders in sequence, evaluate solution
            # Check if this is the best solution (lines 19-23)
            current_delay, _ = self.time_calculator._calculate_delay(state_dict, candidate_route)
            current_slack = self.time_calculator._calculate_slack(state_dict, candidate_route)

            if current_delay < best_delay or (current_delay == best_delay and current_slack < best_slack):
                best_decision = candidate_route.copy()
                best_delay = current_delay
                best_slack = current_slack
                best_postponed = candidate_postponed.copy()  # Make sure this gets updated

        # Before returning, print debug info
        print(f"Solver postponing {len(postponed_orders)} orders")

        # Remove postponed orders from final route plan (line 26)
        final_route = self.route_utils._remove_postponed_orders(best_decision, best_postponed)

        # Output: route plan Θx & postponed orders 3x
        # Return IDs instead of Order objects
        return final_route, {order.id for order in best_postponed}  # Return IDs instead of Order objects
