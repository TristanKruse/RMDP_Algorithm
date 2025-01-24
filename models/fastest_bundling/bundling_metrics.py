import numpy as np
from typing import List
from models.fastest_bundling.main import State


class BundlingMetrics:
    def __init__(self):
        self.total_orders = 0
        self.bundled_orders = 0
        self.bundle_sizes = []
        self.avg_travel_time = []
        self.same_restaurant_bundles = 0
        self.nearby_restaurant_bundles = 0

    def update(self, route_plan: List[List[int]], state: State):
        """Update metrics based on current route plan"""
        for route in route_plan:
            if len(route) > 1:  # Bundle found
                self.bundled_orders += len(route)
                self.bundle_sizes.append(len(route))

                # Check if same restaurant
                restaurants = set()
                for order_id in route:
                    order = next(o for o in state.orders if o.id == order_id)
                    restaurants.add((order.pickup_location.x, order.pickup_location.y))

                if len(restaurants) == 1:
                    self.same_restaurant_bundles += 1
                else:
                    self.nearby_restaurant_bundles += 1

    def print_summary(self):
        print("\nBundling Metrics Summary:")
        print(f"Total bundled orders: {self.bundled_orders}")
        print(f"Average bundle size: {np.mean(self.bundle_sizes):.2f}")
        print(f"Same restaurant bundles: {self.same_restaurant_bundles}")
        print(f"Nearby restaurant bundles: {self.nearby_restaurant_bundles}")
