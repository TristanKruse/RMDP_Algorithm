# environment/visualization_manager.py
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from datatypes import Location, Order, Vehicle
from nn_animation.animation import DeliveryNetworkViz


class VisualizationManager:
    def __init__(self, service_area_dimensions: Tuple[float, float], enabled: bool, update_interval: float):
        self.enabled = enabled
        self.update_interval = max(0.001, update_interval)
        self.service_area_dimensions = service_area_dimensions
        self.viz = None

    def initialize_visualization(self, restaurants=None, vehicles=None) -> None:
        if not self.enabled:
            return
        self.viz = DeliveryNetworkViz(service_area_dimensions=self.service_area_dimensions)
        self.viz.start_animation()
        if restaurants is not None or vehicles is not None:
            self.viz.set_static_positions(
                restaurants=restaurants if restaurants is not None else np.array([]), customers=np.array([])
            )
            if vehicles is not None:
                self.viz.update_vehicle_positions(vehicles)

    def update_step_visualization(
        self, vehicles: List[Vehicle], active_orders: List[Order], restaurants: List[Location], current_time: float
    ) -> None:
        if not self.enabled or not self.viz:
            return
        self.viz.update_vehicle_positions(
            np.array([[v.current_location.x, v.current_location.y] for v in vehicles]), time_step=current_time
        )
        if restaurants:
            self.viz.set_static_positions(
                customers=(
                    np.array([[o.delivery_location.x, o.delivery_location.y] for o in active_orders])
                    if active_orders
                    else np.array([])
                ),
                restaurants=np.array([[r.x, r.y] for r in restaurants]),
            )
        plt.pause(self.update_interval)

    def reset(self) -> None:
        if self.enabled and self.viz:
            plt.clf()

    def close(self) -> None:
        if self.enabled and self.viz:
            self.viz.close()
            plt.close("all")

    def get_visualization_state(self) -> dict:
        return (
            {}
            if not self.enabled or not self.viz
            else {
                "enabled": self.enabled,
                "service_area": self.service_area_dimensions,
                "update_interval": self.update_interval,
            }
        )
