   # def solve(self, state_dict: dict) -> Tuple[Dict[int, Route], Set[int]]:
    #     """ACA Algorithm to solve the RMDP.
        
    #     This algorithm:
    #     1. Considers multiple possible sequences of order assignments
    #     2. Allows for strategic postponement of orders
    #     3. Uses time buffers to handle uncertainties in delivery times
        
    #     Input parameters:
    #         state (S): Current system state
    #         time (t in state): Current time point
    #         route plan (Θ in state): Current route plan
    #         unassigned orders ($o in state): Orders not yet assigned
    #         buffer (b as class parameter): Time buffer for handling uncertainties
    #         max_postponements (pmax as class parameter): Maximum number of orders that can be postponed
    #         max_postpone_time (tpmax as class parameter): Maximum time an order can be postponed
        
    #     The algorithm follows these steps:
    #     1-4. Initialization
    #         x ← ∅ // Best decision
    #         delay ← bigM // Delay
    #         slack ← 0 // Slack
    #     5-7. Start assignment procedure
    #         forall ̂$ ordered set of $o // Process all potential sequences
    #     8-10. For each sequence
    #         ̂Θ ← Θ // Create candidate route plan
    #         ̂P ← ∅ // Initialize set of postponements
    #         forall D ∈ ̂$ // Process all orders in sequence
            
    #     Output:
    #         Tuple containing:
    #         - Route plan as dictionary mapping vehicle IDs to Routes
    #         - Set of orders chosen for postponement
    #     """
    #     # # Get set of valid order IDs at start
    #     # active_order_ids = set(state_dict["unassigned_orders"].keys())

    #     # # Initialize route plan dictionary
    #     # route_plan = {}
    #     # for vehicle_id, route in state_dict["route_plan"].items():

    #     #     # Each sequence element is a tuple (node_id, pickups, deliveries)
    #     #     cleaned_sequence = []
    #     #     for node_id, pickups, deliveries in route.sequence:
    #     #         # Filter pickups and deliveries to only include active orders
    #     #         valid_pickups = {pid for pid in pickups if pid in active_order_ids}
    #     #         valid_deliveries = {did for did in deliveries if did in active_order_ids}
                
    #     #         # Only keep stops that have valid orders
    #     #         if valid_pickups or valid_deliveries:
    #     #             cleaned_sequence.append((node_id, valid_pickups, valid_deliveries))
                        
    #     #     route_plan[vehicle_id] = Route(
    #     #         vehicle_id=vehicle_id,
    #     #         sequence=cleaned_sequence,
    #     #         total_distance=0.0,
    #     #         total_time=0.0
    #     #     )

    #     # print(f"Starting solve with {len(state_dict['unassigned_orders'])} unassigned orders")

    #     # Initialize route plan by directly using the provided state
    #     route_plan = {}
    #     for vehicle_id, route in state_dict["route_plan"].items():
    #         # Create a copy of the route to avoid modifying the original
    #         route_plan[vehicle_id] = Route(
    #             vehicle_id=vehicle_id,
    #             sequence=route.sequence.copy(),  # Just copy the existing sequence
    #             total_distance=0.0,
    #             total_time=0.0
    #         )
    #     # check if correct

    #     print(f"Starting solve with {len(state_dict['unassigned_orders'])} unassigned orders")
    #     print(f"Starting solve with {route_plan} unassigned orders")

    #     # Handle empty state or no unassigned orders (Step 1: Initialization)
    #     if not state_dict["unassigned_orders"]:
    #         return route_plan, set()
        

    #     # Initialize tracking variables (Steps 1-4: Initialization)
    #     best_decision = {k: v.copy() for k, v in route_plan.items()}  # 1. x ← ∅
    #     best_delay = float("inf")  # 2. delay ← bigM
    #     best_slack = 0  # 3. slack ← 0
    #     best_postponed = set()  # Initialize empty set for postponed orders
        
    #     # Step 5-7: Generate order sequences
    #     # 6. forall̂$ ordered set of $o // All potential sequences
    #     if isinstance(state_dict["unassigned_orders"], dict):
    #         order_items = list(state_dict["unassigned_orders"].items())
    #     else:
    #         order_items = state_dict["unassigned_orders"]

    #     # Grows factorially with the number of unassigned orders
    #     # Generate order sequences with the correct format e.g., [A, B, C] or [B, A, C]
    #     # Since the order of assignement matters.
    #     order_sequences = self.route_utils._generate_order_sequences(order_items)

    #     # Steps 8-10: Process each sequence
    #     for sequence in order_sequences:
    #         # Initialize candidate solution
    #         candidate_route = {k: v.copy() for k, v in route_plan.items()}  # ̂8. Θ ← Θ
    #         candidate_postponed = set()  # 9. ̂P ← ∅

    #         # 10. Process each order in sequence (forall D ∈ ̂$)
    #         for order_id, order_info in sequence:

    #             # print(f"Debug - State time: {state_dict['time']}")
    #             # print(f"Debug - Unassigned orders: {state_dict['unassigned_orders']}")
    #             # print(f"Debug - Route plan before assignment: {candidate_route}")
    #             # 14. Check postponement
    #             should_postpone = self.postponement.evaluate_postponement(
    #                 postponed=candidate_postponed,
    #                 route_plan=candidate_route,
    #                 order_id=order_id,
    #                 current_time=state_dict["time"],
    #                 state=state_dict,
    #             )

    #             # Line 16, postponed orders wouldn't need evaluation.
    #             if should_postpone:
    #                 candidate_postponed.add(order_id)
    #                 continue

    #             # 12. Find best vehicle assignment
    #             assignment = self.vehicle_ops.find_vehicle(
    #                 candidate_route, order_id, buffer=self.buffer, state = state_dict
    #             )
                
    #             if assignment is None:
    #                 # Add to postponed orders if no vehicle available
    #                 # candidate_postponed.add(order_id)    # maybe doesn't have to be postponed, is                 
    #                 continue

    #             # 13. Update route for assigned vehicle
    #             candidate_route[assignment.vehicle_id] = Route(
    #                 vehicle_id=assignment.vehicle_id,
    #                 sequence=assignment.tentative_route,
    #                 total_distance=0.0,
    #                 total_time=0.0
    #             )

    #         # Evaluate solution, no buffer included i.e. buffer=0
    #         current_delay = self.time_calculator.calculate_costs(state_dict, candidate_route, buffer=self.buffer)
    #         current_slack = self.time_calculator._calculate_slack(state_dict, candidate_route)

    #         # Update best solution if better
    #         if current_delay < best_delay or (current_delay == best_delay and current_slack > best_slack):
    #             best_decision = {k: v.copy() for k, v in candidate_route.items()}
    #             best_delay = current_delay
    #             best_slack = current_slack
    #             best_postponed = candidate_postponed.copy()

    #     print(f"Solver postponing {len(best_postponed)} orders")

    #     # Remove postponed orders from final routes
    #     final_routes = {}
    #     for vehicle_id, route in best_decision.items():
    #         # Filter the sequence to exclude postponed orders
    #         new_sequence = []
    #         for node_id, pickups, deliveries in route.sequence:
    #             # Remove postponed orders from pickups and deliveries
    #             new_pickups = pickups - best_postponed
    #             new_deliveries = deliveries - best_postponed
                
    #             # Only keep stops that still have orders
    #             if new_pickups or new_deliveries:
    #                 new_sequence.append((node_id, new_pickups, new_deliveries))
            
    #         final_routes[vehicle_id] = Route(
    #             vehicle_id=vehicle_id,
    #             sequence=new_sequence,
    #             total_distance=route.total_distance,
    #             total_time=route.total_time
    #         )



    #     # In aca_policy.py, just before returning final_routes and best_postponed
    #     logger.info(f"Final route assignments:")
    #     for vehicle_id, route in final_routes.items():
    #         if route.sequence:
    #             pickup_counts = sum(len(p) for _, p, _ in route.sequence)
    #             delivery_counts = sum(len(d) for _, _, d in route.sequence)
    #             logger.info(f"Vehicle {vehicle_id}: {pickup_counts} pickups, {delivery_counts} deliveries")
    #             for stop_idx, (node_id, pickups, deliveries) in enumerate(route.sequence):
    #                 if pickups:
    #                     logger.info(f"  Stop {stop_idx}: Node {node_id}, Pickup orders {pickups}")
    #                 if deliveries:
    #                     logger.info(f"  Stop {stop_idx}: Node {node_id}, Deliver orders {deliveries}")
    #     logger.info(f"Postponed orders: {best_postponed}")

    #     return final_routes, best_postponed