import numpy as np
from models.fastest_bundling.main import FastestBundler
from environment.main import RestaurantMealDeliveryEnv
from datatypes import Location, Order
import logging

logging.basicConfig(level=logging.INFO)


def test_bundling_solver():
    """Test case to verify bundling behavior"""

    # Create environment with controlled parameters
    env = RestaurantMealDeliveryEnv(
        movement_per_step=0.5,
        num_restaurants=3,  # Small number for clarity
        num_vehicles=2,
        simulation_duration=300,
        cooldown_duration=30,
        mean_interarrival_time=1.0,  # Frequent orders
        visualize=True,  # To see what's happening
    )

    # Initialize solver
    solver = FastestBundler(movement_per_step=0.5, max_bundle_size=3, max_restaurant_distance=2.0)

    # Add debug logging to solver
    def log_bundle_creation(orders, vehicle_id):
        order_ids = [o.id for o in orders]
        print(f"\nBundle created:")
        print(f"Vehicle {vehicle_id} assigned orders: {order_ids}")
        print(f"Restaurant locations:")
        for o in orders:
            print(f"Order {o.id}: ({o.pickup_location.x:.2f}, {o.pickup_location.y:.2f})")

    solver._log_bundle = log_bundle_creation

    # Run simulation
    state = env.reset()
    total_bundles = 0
    bundle_sizes = []

    while True:
        # Get solver's decision
        route_plan, postponed = solver.solve(state)

        # Log route plan
        print("\nRoute Plan:")
        for vehicle_id, route in enumerate(route_plan):
            if route:
                print(f"Vehicle {vehicle_id}: {route}")
                if len(route) > 1:
                    total_bundles += 1
                    bundle_sizes.append(len(route))

        # Take step in environment
        state, reward, done, info = env.step((route_plan, postponed))

        if done:
            break

    # Print statistics
    print("\nBundling Statistics:")
    print(f"Total bundles created: {total_bundles}")
    print(f"Average bundle size: {np.mean(bundle_sizes):.2f}")
    print(f"Bundle size distribution: {np.bincount(bundle_sizes)[1:]}")


if __name__ == "__main__":
    test_bundling_solver()
