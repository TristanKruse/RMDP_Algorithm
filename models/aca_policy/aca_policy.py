# rmdp_solver.py
# Implements the actual solving algorithm and solution logic
from typing import Tuple, Set, Dict
from datatypes import Route
from .route_utils import RouteUtils
from .vehicle_ops import VehicleOperations
from .postponement import PostponementHandler
from .rl_postponement import RLPostponementDecision
from .time_utils import TimeCalculator
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(module)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger instance
logger = logging.getLogger(__name__)


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
        location_manager,  # Add this new parameter
        # Vehicle parameters
        vehicle_capacity: int = 3,
        # Time parameters
        service_time: float = 2.0,
        mean_prep_time: float = 10.0,
        delivery_window: float = 40.0,
        # RL parameters
        postponement_method: str = "heuristic",  # "heuristic" or "rl"
        rl_training_mode: bool = True,
        rl_state_size: int = 6,
        rl_model_path: str = None,
        # Additional RL hyperparameters for tuning
        rl_learning_rate: float = 0.0005,
        rl_discount_factor: float = 0.95,
        rl_exploration_rate: float = 0.9,
        rl_exploration_decay: float = 0.99999,
        rl_min_exploration_rate: float = 0.2,
        rl_batch_size: int = 64,
        rl_target_update_frequency: int = 50,
        rl_replay_buffer_capacity: int = 50000,
        rl_bundling_reward: float = 0.05,
        rl_postponement_penalty: float = -0.005,
        rl_on_time_reward: float = 0.2
        ):
        self.buffer = buffer
        self.max_postponements = max_postponements
        self.max_postpone_time = max_postpone_time
        self.location_manager = location_manager  # Store it
        self.postponed_orders = {}  # order_id -> first_postpone_time


        # Initialize utility classes 
        self.route_utils = RouteUtils(vehicle_capacity)
        self.time_calculator = TimeCalculator(
            delivery_window=delivery_window,  # Or pass as parameter
            mean_prep_time=mean_prep_time,
            service_time=service_time,
            location_manager=location_manager
        )

        self.vehicle_ops = VehicleOperations(
            service_time=service_time,
            vehicle_capacity=vehicle_capacity,
            mean_prep_time=mean_prep_time,
            location_manager=location_manager,
            delivery_window=delivery_window,
        )

        # Initialize postponement handler based on selected method
        if postponement_method == "heuristic":
            self.postponement = PostponementHandler(
                max_postponements=max_postponements,
                max_postpone_time=max_postpone_time,
            )
            logger.info("Using heuristic-based postponement method")
        else:  # "rl"
            self.postponement = RLPostponementDecision(
                learning_rate=rl_learning_rate,
                discount_factor=rl_discount_factor,
                exploration_rate=rl_exploration_rate,
                exploration_decay=rl_exploration_decay,
                min_exploration_rate=rl_min_exploration_rate,
                batch_size=rl_batch_size,
                training_mode=rl_training_mode,
                state_size=rl_state_size,
                lns_sample_size=5,
                target_update_frequency=rl_target_update_frequency,
                replay_buffer_capacity=rl_replay_buffer_capacity,
                bundling_reward=rl_bundling_reward,
                postponement_penalty=rl_postponement_penalty,
                on_time_reward=rl_on_time_reward
            )
            # Load pre-trained model if path is provided
            if rl_model_path and not rl_training_mode:
                self.postponement.load_model(rl_model_path)
            logger.info("Using RL-based postponement method")
            
        # For storing metrics between steps
        self.previous_delay = 0.0
        self.reward_tracking = {}  # order_id -> reward
    
    def solve(self, state_dict: dict, exploration_rate: float = None) -> Tuple[Dict[int, Route], Set[int]]:
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
        # ----------------- RL Postponement -----------------
        # Calculate initial estimated delays for all orders
        # This updates the current_estimated_delay values for all orders
        self.time_calculator._calculate_delay(
            state=state_dict, 
            route_plan=state_dict["route_plan"],
            buffer=self.buffer
        )
        # ----------------- RL Postponement -----------------

        # Store total estimated delay before decision making
        self.previous_delay = sum(order.current_estimated_delay for order in state_dict.get('orders', []) if hasattr(order, 'current_estimated_delay'))
    
        # Count unassigned orders
        num_unassigned = len(state_dict['unassigned_orders'])
        # logger.info(f"============================================")
        # logger.info(f"TIMESTEP {state_dict['time']}: Processing {num_unassigned} unassigned orders")
        
        # Log vehicle statuses
        vehicles_with_orders = 0
        total_vehicles = len(state_dict["route_plan"])
        for vehicle_id, route in state_dict["route_plan"].items():
            if route.sequence:
                vehicles_with_orders += 1
        
        # logger.info(f"Vehicles with orders: {vehicles_with_orders}/{total_vehicles}")
        
        # Initialize route plan by directly using the provided state
        route_plan = {}
        for vehicle_id, route in state_dict["route_plan"].items():
            # Create a copy of the route to avoid modifying the original
            route_plan[vehicle_id] = Route(
                vehicle_id=vehicle_id,
                sequence=route.sequence.copy(),  # Just copy the existing sequence
                total_distance=0.0,
                total_time=0.0
            )

        # print(f"Starting solve with {len(state_dict['unassigned_orders'])} unassigned orders")
        # print(f"Starting solve with {route_plan} unassigned orders")

        # Handle empty state or no unassigned orders (Step 1: Initialization)
        if not state_dict["unassigned_orders"]:
            return route_plan, set()
        
        # Initialize tracking variables (Steps 1-4: Initialization)
        best_decision = {k: v.copy() for k, v in route_plan.items()}  # 1. x ← ∅
        best_delay = float("inf")  # 2. delay ← bigM
        best_slack = 0  # 3. slack ← 0
        best_postponed = set()  # Initialize empty set for postponed orders
        
        # Step 5-7: Generate order sequences
        # 6. forall̂$ ordered set of $o // All potential sequences
        if isinstance(state_dict["unassigned_orders"], dict):
            order_items = list(state_dict["unassigned_orders"].items())
        else:
            order_items = state_dict["unassigned_orders"]

        # Grows factorially with the number of unassigned orders
        # Generate order sequences with the correct format e.g., [A, B, C] or [B, A, C]
        # Since the order of assignment matters.
        order_sequences = self.route_utils._generate_order_sequences(order_items)

        # Track assignments for debugging
        assignments_made = 0
        assignments_failed = 0
        
        # Store current total delay for reward calculation
        current_delay = self.time_calculator._calculate_delay(state_dict, route_plan, buffer=self.buffer)
        
        # Determine if we're using RL-based postponement by checking the type
        is_rl_postponement = isinstance(self.postponement, RLPostponementDecision)

        # Phase 1: Pre-evaluate all orders for postponement ONLY for RL method
        rl_postponement_decisions = {}
        if is_rl_postponement:
            # Pre-evaluate all orders to get postponement decisions for RL
            for order_id, order_info in order_items:
                # Make postponement decision once for each order
                should_postpone = self.postponement.evaluate_postponement(
                    postponed=set(),  # Empty set for initial evaluation
                    route_plan=route_plan,  # Use initial route plan
                    order_id=order_id,
                    current_time=state_dict["time"],
                    state=state_dict,
                    exploration_rate=exploration_rate
                )
                rl_postponement_decisions[order_id] = should_postpone


        # Steps 8-10: Process each sequence
        for sequence in order_sequences:
            # Initialize candidate solution
            candidate_route = {k: v.copy() for k, v in route_plan.items()}  # ̂8. Θ ← Θ
            candidate_postponed = set()  # 9. ̂P ← ∅

            # Log sequence being processed
            # logger.info(f"Processing sequence with {len(sequence)} orders")
        
            # 10. Process each order in sequence (forall D ∈ ̂$)
            for order_id, order_info in sequence:

                # 14. Check postponement
                if is_rl_postponement:
                    # Use the pre-evaluated decision for RL
                    should_postpone = rl_postponement_decisions[order_id]
                else:
                    # Use normal evaluation for heuristic
                    should_postpone = self.postponement.evaluate_postponement(
                        postponed=candidate_postponed,
                        route_plan=candidate_route,
                        order_id=order_id,
                        current_time=state_dict["time"],
                        state=state_dict,
                    )

                # Line 16, postponed orders wouldn't need evaluation.
                if should_postpone:
                    candidate_postponed.add(order_id)
                    continue

                # Log before assignment attempt
                # logger.info(f"Attempting to assign order {order_id}")
                    
                # 12. Find best vehicle assignment
                assignment = self.vehicle_ops.find_vehicle(
                    candidate_route, order_id, buffer=self.buffer, state=state_dict
                )
                
                if assignment is None:
                    # If no vehicle available, just continue (don't postpone)
                    # logging.info(f"FAILED TO  to assign order {order_id}")
                    continue
                # Log successful assignment
                # logger.info(f"Assigned order {order_id} to vehicle {assignment.vehicle_id}")
                assignments_made += 1
                
                # 13. Update route for assigned vehicle
                candidate_route[assignment.vehicle_id] = Route(
                    vehicle_id=assignment.vehicle_id,
                    sequence=assignment.tentative_route,
                    total_distance=0.0,
                    total_time=0.0
                )

            # Evaluate solution, with buffer included
            current_delay = self.time_calculator._calculate_delay(state_dict, candidate_route, buffer=self.buffer)
            current_slack = self.time_calculator._calculate_slack(state_dict, candidate_route)

            # logger.info(f"Sequence evaluation: delay {current_delay}, slack {current_slack}")

            # 18., 19. Update best solution if better
            if current_delay < best_delay or (current_delay == best_delay and current_slack > best_slack):
                best_decision = {k: v.copy() for k, v in candidate_route.items()}
                best_delay = current_delay
                best_slack = current_slack
                best_postponed = candidate_postponed.copy()

        # print(f"Solver postponing {len(best_postponed)} orders")

        # 26. Remove postponed orders from final routes
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

        # ----------------- RL Postponement -----------------
        # Calculate final estimated delay for all orders based on the final route plan
        # This updates the current_estimated_delay values for orders
        self.time_calculator._calculate_delay(
            state=state_dict,
            route_plan=final_routes,
            buffer=self.buffer
        )
        
        # Store total estimated delay after decision making
        self.current_delay = sum(order.current_estimated_delay for order in state_dict.get('orders', []) if hasattr(order, 'current_estimated_delay'))
        # ----------------- RL Postponement -----------------

        return final_routes, best_postponed

    def save_rl_model(self, path: str) -> None:
        """Save the RL model to the specified path."""
        if hasattr(self, 'postponement') and hasattr(self.postponement, 'save_model'):
            self.postponement.save_model(path)
            return True
        return False
    
    def load_rl_model(self, path: str) -> None:
        """Load the RL model from the specified path."""
        if hasattr(self, 'postponement') and hasattr(self.postponement, 'load_model'):
            self.postponement.load_model(path)
            logger.info(f"RL model loaded from {path}")
            return True
        return False