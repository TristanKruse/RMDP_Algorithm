import sys
import os
from typing import List, Tuple
#from cheapest_insertion.cheapest_insertion import CheapestInsert
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Now try the import
from cheapest_insertion.cheapest_insertion import CheapestInsertion, Location, Stop



class CppRouteOptimizer:
    def __init__(
        self,
        service_time: float = 2.0,
        vehicle_speed: float = 40.0,  # km/h
        street_network_factor: float = 1.4
    ):
        # Pass arguments positionally instead of as kwargs
        self.insertion = CheapestInsertion(service_time, vehicle_speed, street_network_factor)

    # Rest of your code stays the same
    def find_best_insertion(
        self,
        current_sequence: List[Tuple[int, set, set]],
        order_id: int,
        restaurant_loc: Location,
        customer_loc: Location,
        current_time: float,
        deadline: float
    ) -> Tuple[List[Tuple[int, set, set]], float]:
        # Convert Python sequence to C++ format
        cpp_sequence = [
            Stop(
                node_id=node_id,
                pickups=list(pickups),
                deliveries=list(deliveries)
            )
            for node_id, pickups, deliveries in current_sequence
        ]

        # Call C++ implementation
        result = self.insertion.find_best_insertion(
            current_sequence=cpp_sequence,
            order_id=order_id,
            restaurant_loc=restaurant_loc,
            customer_loc=customer_loc,
            current_time=current_time,
            deadline=deadline
        )

        # Convert C++ result back to Python format
        new_sequence = [
            (stop.node_id, set(stop.pickups), set(stop.deliveries))
            for stop in result.new_sequence
        ]

        return new_sequence, result.insertion_cost

# Example usage:
if __name__ == "__main__":
    # Create optimizer
    optimizer = CppRouteOptimizer()

    # Example data
    current_sequence = [
        (1, {101}, set()),  # Restaurant stop for order 101
        (2, set(), {101})   # Customer stop for order 101
    ]
    
    new_order_id = 102
    restaurant_loc = Location(x=10.0, y=20.0)
    customer_loc = Location(x=15.0, y=25.0)
    
    # Find best insertion
    new_sequence, cost = optimizer.find_best_insertion(
        current_sequence=current_sequence,
        order_id=new_order_id,
        restaurant_loc=restaurant_loc,
        customer_loc=customer_loc,
        current_time=0.0,
        deadline=40.0
    )
    
    print(f"Insertion cost: {cost}")
    print("New sequence:")
    for node_id, pickups, deliveries in new_sequence:
        print(f"  Node {node_id}: pickups={pickups}, deliveries={deliveries}")