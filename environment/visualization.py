# environment/visualization_manager.py
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from datatypes import Location, Order, Vehicle
from nn_animation.animation import DeliveryNetworkViz


class VisualizationManager:
    """Manages visualization for the restaurant delivery environment."""

    def __init__(self, service_area_dimensions: Tuple[float, float], enabled: bool, update_interval: float):
        """
        Initialize the visualization manager.

        Args:
            service_area_dimensions: Dimensions of service area (width, height)
            enabled: Whether visualization is enabled
            update_interval: Time interval between visualization updates
        """
        self.enabled = enabled
        self.update_interval = update_interval
        self.service_area_dimensions = service_area_dimensions
        self.viz = None

    def initialize_visualization(self, restaurants=None, vehicles=None) -> None:
        """
        Initialize visualization components.

        Args:
            restaurants: Optional array of restaurant positions
            vehicles: Optional array of vehicle positions
        """
        if not self.enabled:
            return

        self.viz = DeliveryNetworkViz(service_area_dimensions=self.service_area_dimensions)
        self.viz.start_animation()

        # Update initial positions if provided
        if restaurants is not None or vehicles is not None:
            self.viz.set_static_positions(
                restaurants=restaurants if restaurants is not None else np.array([]),
                customers=np.array([]),  # Empty at initialization
            )
            if vehicles is not None:
                self.viz.update_vehicle_positions(vehicles)

    def update_step_visualization(
        self, vehicles: List[Vehicle], active_orders: List[Order], restaurants: List[Location], current_time: float
    ) -> None:
        """
        Update visualization for current simulation step.

        Args:
            vehicles: List of vehicles to visualize
            active_orders: List of active orders to visualize
            restaurants: List of restaurant locations
            current_time: Current simulation time
        """
        if not self.enabled or not self.viz:
            return

        # Update vehicle positions
        vehicle_positions = np.array([[v.current_location.x, v.current_location.y] for v in vehicles])
        self.viz.update_vehicle_positions(vehicle_positions, time_step=current_time)

        # Update customer and restaurant positions
        if restaurants:
            restaurant_positions = np.array([[r.x, r.y] for r in restaurants])
            if active_orders:
                customer_positions = np.array(
                    [[order.delivery_location.x, order.delivery_location.y] for order in active_orders]
                )
            else:
                customer_positions = np.array([])

            self.viz.set_static_positions(customers=customer_positions, restaurants=restaurant_positions)

        # Add delay to control visualization update rate
        plt.pause(self.update_interval)

    def reset(self) -> None:
        """Reset visualization to initial state."""
        if self.enabled and self.viz:
            # Clear current visualization
            plt.clf()

    def close(self) -> None:
        """Clean up visualization resources."""
        if self.enabled and self.viz:
            self.viz.close()
            plt.close("all")  # Ensure all matplotlib windows are closed

    def set_update_interval(self, interval: float) -> None:
        """
        Set the visualization update interval.

        Args:
            interval: New update interval in seconds
        """
        self.update_interval = max(0.001, interval)  # Ensure minimum interval

    def get_visualization_state(self) -> dict:
        """
        Get current visualization state.

        Returns:
            Dictionary containing current visualization state
        """
        if not self.enabled or not self.viz:
            return {}

        return {
            "enabled": self.enabled,
            "service_area": self.service_area_dimensions,
            "update_interval": self.update_interval,
        }
