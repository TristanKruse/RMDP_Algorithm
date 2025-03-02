from typing import Dict
from datatypes import Location, Route
import logging

# Minimize the total expected cost.
# min E[∑(D∈$) max{0, αDreal - (tD + t̄)}]
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class TimeCalculator:
    def __init__(self, delivery_window: float, mean_prep_time: float, service_time: float, location_manager):
        self.delivery_window = delivery_window
        self.mean_prep_time = mean_prep_time
        self.service_time = service_time
        self.location_manager = location_manager

    # Combining deterministic and stochastic costs
    # Total Cost = Cd(Sk, x) + Cs(Skx, ωk)
    def calculate_costs(self, state: dict, route_plan: Dict[int, Route], buffer: float = 0.0) -> float:
        """Calculate total costs for a decision by combining deterministic and stochastic costs.
        
        Following Ulmer et al.: Total Cost = Cd(Sk, x) + Cs(Skx, ωk)
        Where:
        - Cd: Change in planned delays from decision
        - Cs: Additional delays from uncertainty in ready times
        
        Args:
            state: Current system state with order and vehicle info
            route_plan: Proposed route plan after decision
            
        Returns:
            Total expected delay cost for the decision
        """
        # Calculate deterministic costs (planned delays)
        deterministic_cost = self._calculate_deterministic_cost(state, route_plan, buffer)
        
        # Calculate stochastic costs (uncertainty-based delays)
        stochastic_cost = self._calculate_stochastic_cost(state, route_plan)
        
        # Return total costs
        return deterministic_cost + stochastic_cost

    # Deterministic cost
    # Cd(Sk, x) = Δ(Skx) - Δ(Sk)
    def _calculate_deterministic_cost(self, state: dict, route_plan: Dict[int, Route], buffer: float = 0.0) -> float:
        """Calculate Cd(Sk, x) = Δ(Skx) - Δ(Sk)"""
        current_delay = self._calculate_delay(state, state["route_plan"], buffer)
        new_delay = self._calculate_delay(state, route_plan, buffer)
        return new_delay - current_delay
   
    def _calculate_delay(self, state: dict, route_plan: Dict[int, Route], buffer: float = 0.0) -> float:
        """Calculate Δ(S) = ∑max{0, (aD + b) - (tD + t̄)}"""
        total_delay = 0.0

        # First get planned arrival times for all stops
        planned_times = self._calculate_planned_arrival_times(state, route_plan)
        
        # Then calculate delays - simplify the loops since we're often dealing with just one vehicle
        for route in route_plan.values():
            for _, _, deliveries in route.sequence:
                for order_id in deliveries:
                    if order_id in planned_times:
                        # Find the order in either unassigned_orders or the main orders list
                        order_info = None
                        if order_id in state.get("unassigned_orders", {}):
                            order_info = state["unassigned_orders"][order_id]
                        else:
                            # Look for the order in the main orders list
                            for order in state.get("orders", []):
                                if order.id == order_id:
                                    # Create similar structure to unassigned_orders for consistency
                                    order_info = {"request_time": order.request_time}
                                    break
                        
                        # If we found order info, calculate delay
                        if order_info:
                            planned_arrival = planned_times[order_id]
                            request_time = order_info["request_time"]
                            
                            # Add buffer to planned arrival time
                            buffered_arrival = planned_arrival + buffer
                            
                            delay = max(0, buffered_arrival - (request_time + self.delivery_window))
                            total_delay += delay
        # logging.debug(f"Total delay: {total_delay}")               
        return total_delay


    def _calculate_planned_arrival_times(self, state: dict, route_plan: Dict[int, Route]) -> Dict[int, float]:
        """Calculate planned arrival times for all orders in route plan."""
        planned_times = {}
        
        for vid, route in route_plan.items():
            # Get vehicle's starting position
            try:
                current_loc = state["vehicle_positions"][vid]
            except KeyError:
                # logging.debug(f"No position found for vehicle {vid}, skipping")
                continue
                
            current_time = state["time"]
            # logging.debug(f"Processing route for vehicle {vid} with {len(route.sequence)} stops")
            
            # Keep track of visited nodes to handle revisits
            node_locations = {}
            
            # Process each stop in sequence
            for stop_idx, (node_id, pickups, deliveries) in enumerate(route.sequence):
                # Try to get node location - first check if we've seen it before
                if node_id in node_locations:
                    node_location = node_locations[node_id]
                else:
                    # Try to create location from orders
                    found_location = False
                    
                    # For pickup nodes, look for order pickup information
                    for order_id in pickups:
                        # Look in unassigned_orders
                        if order_id in state.get("unassigned_orders", {}):
                            pickup_node = state["unassigned_orders"][order_id]["pickup_node_id"]
                            node_location = self._get_location(pickup_node)
                            node_locations[node_id] = node_location
                            found_location = True
                            # logging.debug(f"Found pickup location for node {node_id} from order {order_id}")
                            break
                        
                        # Look in all orders if not found
                        if not found_location:
                            for order in state.get("orders", []):
                                if order.id == order_id:
                                    node_location = self._get_location(order.pickup_node_id)
                                    node_locations[node_id] = node_location
                                    found_location = True
                                    # logging.debug(f"Found pickup location for node {node_id} from orders list")
                                    break
                    
                    # For delivery nodes, look for order delivery information
                    if not found_location:
                        for order_id in deliveries:
                            # Look in unassigned_orders
                            if order_id in state.get("unassigned_orders", {}):
                                delivery_node = state["unassigned_orders"][order_id]["delivery_node_id"]
                                node_location = self._get_location(delivery_node)
                                node_locations[node_id] = node_location
                                found_location = True
                                # logging.debug(f"Found delivery location for node {node_id} from order {order_id}")
                                break
                            
                            # Look in all orders if not found
                            if not found_location:
                                for order in state.get("orders", []):
                                    if order.id == order_id:
                                        node_location = self._get_location(order.delivery_node_id)
                                        node_locations[node_id] = node_location
                                        found_location = True
                                        # logging.debug(f"Found delivery location for node {node_id} from orders list")
                                        break
                
                # Skip if we still couldn't find location
                if 'node_location' not in locals() or node_location is None:
                    # logging.debug(f"Could not find location for node {node_id}, skipping stop")
                    continue
                
                # Travel to node
                travel_time = self.location_manager.get_travel_time(current_loc, node_location)
                current_time += travel_time
                current_loc = node_location
                
                # logging.debug(f"Travel to node {node_id}: current_time now {current_time}")
                
                # For pickup nodes, wait for food to be ready if needed
                for order_id in pickups:
                    # Get request time and calculate expected ready time
                    request_time = None
                    if order_id in state.get("unassigned_orders", {}):
                        request_time = state["unassigned_orders"][order_id]["request_time"]
                    else:
                        for order in state.get("orders", []):
                            if order.id == order_id:
                                request_time = order.request_time
                                break
                    
                    if request_time is not None:
                        expected_ready = request_time + self.mean_prep_time
                        old_time = current_time
                        current_time = max(current_time, expected_ready)
                        # if current_time > old_time:
                            # logging.debug(f"Waiting for order {order_id} prep, current_time now {current_time}")
                
                # Add service time at this stop
                current_time += self.service_time
                # logging.debug(f"Added service time, current_time now {current_time}")
                
                # Record delivery times for all orders delivered at this stop
                for order_id in deliveries:
                    planned_times[order_id] = current_time
                    # logging.debug(f"Recorded planned delivery time {current_time} for order {order_id}")
        
        # logging.debug(f"Final planned arrival times: {planned_times}")
        return planned_times

    # Stochastic cost
    # Cs(Skx, ωk) = ∑(D∈$ω,Θk) [max(αDreal(ωk) - (tD + t̄), 0) - max(aD - (tD + t̄), 0)]
    def _calculate_stochastic_cost(self, state: dict, route_plan: Dict[int, Route]) -> float:
        """Calculate stochastic costs Cs(Skx, ωk) from difference between planned and actual times.
        
        Only considers orders that:
        - Were delivered in this time step
        - Are affected by next vehicle stops
        """
        stochastic_delay = 0.0
        
        # Get planned arrival times
        planned_times = self._calculate_planned_arrival_times(state, route_plan)
        
        # Get affected orders from active orders
        for order in state["orders"]:
            # Only consider orders that have been delivered (have delivery_time)
            # or are in next stops of vehicles
            is_delivered = order.delivery_time is not None
            is_next_stop = False
            
            # Check if order is in next stops
            for route in route_plan.values():
                if route.sequence:
                    _, pickups, deliveries = route.sequence[0]
                    if order.id in pickups or order.id in deliveries:
                        is_next_stop = True
                        break
            
            if is_delivered or is_next_stop:
                order_info = state["unassigned_orders"].get(order.id)
                if order_info and order.id in planned_times:
                    deadline = order_info["request_time"] + self.delivery_window
                    
                    # Use actual delivery time if available, else current time
                    realized_time = order.delivery_time if order.delivery_time is not None else state["time"]
                    planned_time = planned_times[order.id]
                    
                    actual_delay = max(0, realized_time - deadline)
                    planned_delay = max(0, planned_time - deadline)
                    
                    stochastic_delay += actual_delay - planned_delay
        
        return stochastic_delay
    
    def _calculate_slack(self, state: dict, route_plan: Dict[int, Route]) -> float:
        """Calculate total slack Slack(S,Θ) in route plan.
        
        Slack represents the sum of time margins between planned arrivals 
        and deadlines. Higher slack indicates more flexibility.
        
        Following Ulmer et al.: Slack(S,Θ) = ∑ max{0, (tD + t̄) - aD}
        Where:
        - tD: Request time
        - t̄: Delivery window
        - aD: Planned arrival time
        """
        total_slack = 0.0
        
        # Get planned arrival times using our existing method
        planned_times = self._calculate_planned_arrival_times(state, route_plan)
        
        # Calculate slack for each delivery
        for route in route_plan.values():
            if not route.sequence:
                continue
                
            for _, _, deliveries in route.sequence:
                for order_id in deliveries:
                    order_info = state["unassigned_orders"].get(order_id)
                    if order_info and order_id in planned_times:
                        deadline = order_info["request_time"] + self.delivery_window
                        planned_arrival = planned_times[order_id]
                        slack = max(0.0, deadline - planned_arrival)
                        total_slack += slack
        
        return total_slack

    def _get_location(self, obj):
        """
        Recursively extract a Location from an object.
        If the object already has an 'x' attribute, we assume it's a Location.
        Otherwise, if it has a 'location' attribute, we return _get_location(obj.location).
        """
        if isinstance(obj, Location):
            return obj
        if hasattr(obj, 'x') and hasattr(obj, 'y'):
            return obj
        if hasattr(obj, 'location'):
            return self._get_location(obj.location)
        return obj