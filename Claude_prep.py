def find_vehicle(self, route_plan: dict, order_id: int, buffer: float, state: dict) -> Optional[VehicleAssignment]:
    """Find best vehicle for order insertion using cheapest insertion."""
    best_assignment = None
    min_total_cost = float('inf')
    max_slack = -float('inf')  # Track best slack for tie-breaking
    
    # Get order info
    order_info = state["unassigned_orders"].get(order_id)
    if not order_info:
        return None

    # Get locations
    restaurant_node = order_info["pickup_node_id"]
    customer_node = order_info["delivery_node_id"]
    
    for vehicle_id, route in route_plan.items():
        # Skip vehicles at capacity
        if self._count_active_orders(route) >= self.vehicle_capacity:
            continue
        
        assignment = self._evaluate_vehicle_assignment(
            vehicle_id, route, order_id,
            restaurant_node, customer_node, buffer, state
        )
        
        if assignment:
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
                
                # Early termination if perfect solution
                if min_total_cost == 0:
                    break
                    
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
                
                # If better slack, update best assignment
                if current_slack > max_slack:
                    max_slack = current_slack
                    best_assignment = assignment
    
    return best_assignment