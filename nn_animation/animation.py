import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


class DeliveryNetworkViz:
    def __init__(self, service_area_dimensions=(10.0, 10.0)):
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.service_area = service_area_dimensions
        self.previous_vehicle_positions = None
        self.time_text = None
        self.is_paused = False
        self.setup_style()

    def setup_style(self):
        # Previous style setup remains the same
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")
        self.ax.grid(False)
        self.ax.set_xlim(0, self.service_area[0])
        self.ax.set_ylim(0, self.service_area[1])

        # Style the axes
        self.ax.tick_params(colors="white", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#333333")

        # Add km markers to axes
        self.ax.set_xticks(np.arange(0, self.service_area[0] + 1, 2))
        self.ax.set_yticks(np.arange(0, self.service_area[1] + 1, 2))
        self.ax.set_xlabel("Distance (km)", color="white")
        self.ax.set_ylabel("Distance (km)", color="white")

        # Set title
        self.ax.set_title("Restaurant Meal Delivery (RMD)", color="white", pad=20, fontsize=14)

        # Initialize scatter plots
        self.vehicle_scatter = self.ax.scatter([], [], c="cyan", s=30, alpha=0.8, label="Vehicles")
        self.customer_scatter = None
        self.restaurant_scatter = None

        # Initialize time counter text
        self.time_text = self.ax.text(
            0.02,
            1.05,  # Position in axes coordinates
            "Time: 0",
            transform=self.ax.transAxes,  # Use axes coordinates
            color="white",
            fontsize=12,
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="white", pad=5),
        )

    def update_time(self, current_time):
        """Update the time counter"""
        if self.time_text:
            self.time_text.set_text(f"Minute: {current_time}")
            self.fig.canvas.draw_idle()

    def set_static_positions(self, customers: np.ndarray, restaurants: np.ndarray):
        """Set the static positions for customers and restaurants"""
        # Handle empty customer array at initialization
        if customers.size == 0:
            customers = np.zeros((0, 2))

        # Update or create scatter plots
        if self.customer_scatter is not None:
            self.customer_scatter.remove()
        if self.restaurant_scatter is not None:
            self.restaurant_scatter.remove()

        # Create new scatter plots with smaller markers
        self.customer_scatter = self.ax.scatter(
            customers[:, 0],
            customers[:, 1],
            c="purple",
            s=20,
            alpha=0.7,
            label="Customers",  # Reduced size from 80
            marker="o",
        )

        self.restaurant_scatter = self.ax.scatter(
            restaurants[:, 0],
            restaurants[:, 1],
            c="green",
            s=30,
            alpha=0.7,
            label="Restaurants",  # Reduced size from 100
            marker="s",
        )

        # Update legend
        handles = []
        labels = []
        if self.restaurant_scatter is not None:
            handles.append(self.restaurant_scatter)
            labels.append("Restaurants")
        if self.customer_scatter is not None and customers.size > 0:
            handles.append(self.customer_scatter)
            labels.append("Customers")
        if self.vehicle_scatter is not None:
            handles.append(self.vehicle_scatter)
            labels.append("Vehicles")

        self.ax.legend(
            handles=handles, labels=labels, loc="upper right", framealpha=0.3, fontsize=8
        )  # Reduced font size from 10

        # Only add connections for active customers
        if customers.size > 0:
            self._add_connections(customers, restaurants)

    def _add_connections(self, customers: np.ndarray, restaurants: np.ndarray):
        """Add subtle connecting lines between customers and restaurants"""
        # Clear previous lines
        for artist in self.ax.lines[:]:
            if not hasattr(artist, "is_vehicle_trail"):  # Don't remove vehicle trails
                artist.remove()

        # Only draw lines for the closest restaurant to each customer
        for customer in customers:
            # Find closest restaurant
            distances = np.sqrt(np.sum((restaurants - customer) ** 2, axis=1))
            closest_idx = np.argmin(distances)
            closest_restaurant = restaurants[closest_idx]

            # Draw line to closest restaurant only
            self.ax.plot(
                [customer[0], closest_restaurant[0]],
                [customer[1], closest_restaurant[1]],
                c="cyan",
                alpha=0.1,
                linestyle="--",
                linewidth=0.3,  # Reduced linewidth from 0.5
            )

    def update_vehicle_positions(self, positions: np.ndarray, vehicle_ids=None, time_step: int = None):
        """Update vehicle positions and draw movement trails"""
        if positions.size > 0:
            # Update scatter plot
            self.vehicle_scatter.set_offsets(positions)

            # Add or update vehicle ID labels
            if vehicle_ids is not None:
                # Remove old text annotations if they exist
                for txt in self.ax.texts[:]:
                    if hasattr(txt, "is_vehicle_id"):
                        txt.remove()

                # Add new text annotations
                for pos, vid in zip(positions, vehicle_ids):
                    txt = self.ax.text(pos[0], pos[1], str(vid), color="white", fontsize=8, ha="center", va="bottom")
                    txt.is_vehicle_id = True

            # Rest of the existing code for trails
            if self.previous_vehicle_positions is not None:
                for prev, curr in zip(self.previous_vehicle_positions, positions):
                    line = self.ax.plot([prev[0], curr[0]], [prev[1], curr[1]], c="cyan", alpha=0.3, linewidth=0.5)[0]
                    line.is_vehicle_trail = True

            self.previous_vehicle_positions = positions.copy()

        if time_step is not None:
            self.update_time(time_step)

    def close(self):
        """Close the visualization window"""
        plt.close(self.fig)

    def _on_key_press(self, event):
        """Handle key press events"""
        if event.key == " ":  # Space bar
            self.is_paused = not self.is_paused
            print(f"Simulation {'paused' if self.is_paused else 'resumed'}")
            plt.draw()

    def start_animation(self, interval: int = 50):
        """Start the animation"""
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        plt.show(block=False)
        plt.pause(0.1)

    def is_simulation_paused(self):
        """Return current pause state"""
        return self.is_paused
