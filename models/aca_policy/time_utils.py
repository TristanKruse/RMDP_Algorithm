from typing import List, Dict, Optional, Tuple, Set
from datatypes import State, Location, Route
import numpy as np

# vehicle seems to always be none

class TimeCalculator:
    def __init__(
        self,
        mean_prep_time: float,
        prep_time_var: float,
        service_time: float,
        delay_normalization_factor: float,
        location_manager,
        vehicle_speed: float = 40.0,
        street_network_factor: float = 1.4,
    ):
        self.mean_prep_time = mean_prep_time
        self.prep_time_var = prep_time_var
        self.service_time = service_time
        self.delay_normalization_factor = delay_normalization_factor
        self.vehicle_speed = vehicle_speed
        self.street_network_factor = street_network_factor
        self.location_manager = location_manager

    def _calculate_delay(self, state: dict, route_plan: Dict[int, Route], buffer: float = 0.0, vehicle_id: int = None) -> Tuple[float, Set[int]]:
        if not route_plan:
            return 0.0, set()

        total_delay = 0.0
        late_order_ids = set()

        # Process each route
        for vid, route in route_plan.items():
            if vehicle_id is not None and vid != vehicle_id:
                continue

            if not route.sequence:  # Skip empty routes
                continue

            # Process each node in sequence
            current_time = state["time"]
            
            # Get vehicle's current location
            if "vehicle_positions" in state:
                current_loc = state["vehicle_positions"][vid]
            else:
                print(f"Warning: No position found for vehicle {vid}")
                continue
        for node_id, pickups, deliveries in route.sequence:
            # Add travel time to node
            if node_id in state["nodes"]:
                node = state["nodes"][node_id]
                # Access location through node.location
                current_time += self._calculate_travel_time(current_loc, node.location)
                current_loc = node.location

            # Handle pickups (restaurants)
            for order_id in pickups:
                order_info = state["unassigned_orders"].get(order_id)
                if order_info:
                    # Access pickup location through node
                    pickup_node = order_info["pickup_node_id"]
                    pickup_loc = pickup_node.location
                    current_time += self._calculate_travel_time(current_loc, pickup_loc)
                    # Add prep time and service time
                    expected_ready = order_info["request_time"] + self.mean_prep_time
                    current_time = max(current_time, expected_ready)
                    current_time += self.service_time
                    current_loc = pickup_loc

            # Handle deliveries (customers)
            for order_id in deliveries:
                order_info = state["unassigned_orders"].get(order_id)
                if order_info:
                    # Access delivery location through node
                    delivery_node = order_info["delivery_node_id"]
                    delivery_loc = delivery_node.location
                    current_time += self._calculate_travel_time(current_loc, delivery_loc)
                    current_time += self.service_time
                    # Calculate delay
                    delivery_time = current_time + buffer
                    deadline = order_info["request_time"] + 40.0  # Hardcoded delivery window
                    delay = max(0, delivery_time - deadline)
                    
                    if delay > 0:
                        total_delay += delay * self.delay_normalization_factor
                        late_order_ids.add(order_id)
                    current_loc = delivery_loc

        return total_delay, late_order_ids




        #     for node_id, pickups, deliveries in route.sequence:
        #         # Add travel time to node
        #         if node_id in state["nodes"]:  # Now this check will work
        #             node_loc = state["nodes"][node_id].location
        #             current_time += self._calculate_travel_time(current_loc, node_loc)
        #             current_loc = node_loc

        #         # Handle pickups (restaurants)
        #         for order_id in pickups:
        #             order_info = state["unassigned_orders"].get(order_id)
        #             if order_info:
        #                 # Add prep time and service time
        #                 expected_ready = order_info["request_time"] + self.mean_prep_time
        #                 current_time = max(current_time, expected_ready)
        #                 current_time += self.service_time

        #         # Handle deliveries (customers)
        #         for order_id in deliveries:
        #             order_info = state["unassigned_orders"].get(order_id)
        #             if order_info:
        #                 # Add service time
        #                 current_time += self.service_time
        #                 # Calculate delay
        #                 delivery_time = current_time + buffer
        #                 deadline = order_info["request_time"] + 40.0  # Hardcoded delivery window
        #                 delay = max(0, delivery_time - deadline)
                        
        #                 if delay > 0:
        #                     total_delay += delay * self.delay_normalization_factor
        #                     late_order_ids.add(order_id)

        # return total_delay, late_order_ids

    def _calculate_slack(self, state: dict, route_plan: Dict[int, Route]) -> float:
        """Calculate total slack in route plan."""
        total_slack = 0.0

        for route in route_plan.values():
            if not route.sequence:
                continue

            for _, pickups, deliveries in route.sequence:
                # Only calculate slack for deliveries
                for order_id in deliveries:
                    order_info = state["unassigned_orders"].get(order_id)
                    if order_info:
                        deadline = order_info["request_time"] + 40.0  # Hardcoded delivery window
                        planned_arrival = self._calculate_planned_arrival(state, route, order_id)
                        slack = max(0.0, deadline - planned_arrival)
                        total_slack += slack

        return total_slack

    def _calculate_planned_arrival(self, state: dict, route: Route, target_order_id: int) -> float:
        """Calculate planned arrival time for a specific order."""
        current_time = state["time"]
        
        # Find target order in sequence
        target_delivery_found = False
        current_loc = None
        
        for node_id, pickups, deliveries in route.sequence:
            # Skip until we find either pickup or delivery for target order
            if target_order_id not in pickups and target_order_id not in deliveries:
                continue
                
            if target_order_id in deliveries:
                target_delivery_found = True
                if node_id in state["nodes"]:
                    node_loc = state["nodes"][node_id].location
                    if current_loc:
                        current_time += self._calculate_travel_time(current_loc, node_loc)
                    current_time += self.service_time
                    break
                    
            # Handle pickup
            if target_order_id in pickups:
                order_info = state["unassigned_orders"].get(target_order_id)
                if order_info:
                    if node_id in state["nodes"]:
                        node_loc = state["nodes"][node_id].location
                        if current_loc:
                            current_time += self._calculate_travel_time(current_loc, node_loc)
                        expected_ready = order_info["request_time"] + self.mean_prep_time
                        current_time = max(current_time, expected_ready)
                        current_time += self.service_time
                        current_loc = node_loc

        if not target_delivery_found:
            return float("inf")
            
        return current_time

    # def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
    #     """Calculate travel time between two locations."""
    #     return self.location_manager.get_travel_time(loc1, loc2) if self.location_manager else 0.0

    # def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
    #     """Calculate travel time between two locations."""
    #     if not isinstance(loc1, Location) or not isinstance(loc2, Location):
    #         print(f"Warning: Invalid location types - loc1: {type(loc1)}, loc2: {type(loc2)}")
    #         return 0.0
    #     return self.location_manager.get_travel_time(loc1, loc2) if self.location_manager else 0.0


    def _calculate_travel_time(self, loc1: Location, loc2: Location) -> float:
        """Calculate travel time between two locations."""
        try:
            # If we get a Node object instead of Location, extract its location
            if hasattr(loc1, 'location'):
                loc1 = loc1.location
            if hasattr(loc2, 'location'):
                loc2 = loc2.location
                
            # Now both should be Location objects
            if not isinstance(loc1, Location) or not isinstance(loc2, Location):
                print(f"Warning: Invalid location types after conversion - loc1: {type(loc1)}, loc2: {type(loc2)}")
                return 0.0
                
            return self.location_manager.get_travel_time(loc1, loc2) if self.location_manager else 0.0
            
        except Exception as e:
            print(f"Error in calculate_travel_time: {str(e)}")
            print(f"loc1: {loc1}, type: {type(loc1)}")
            print(f"loc2: {loc2}, type: {type(loc2)}")
            return 0.0




    def _find_vehicle_for_route(self, state: State, target_order_id: int) -> Optional[int]:
        """Helper method to find vehicle associated with a route containing target_order."""
        for v in state.vehicles:
            if v.id >= len(state.route_plan):
                continue
            route_orders = state.route_plan[v.id]
            # Handle both single integers and lists
            if isinstance(route_orders, int):
                if route_orders == target_order_id:
                    return v.id
            elif isinstance(route_orders, list):
                if target_order_id in route_orders:
                    return v.id
        return None
