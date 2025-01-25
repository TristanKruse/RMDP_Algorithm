# environment/route_processing/movement_location.py
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger instance
logger = logging.getLogger(__name__)


class MovementLocation:
    def __init__(self, location_manager):
        self.location_manager = location_manager

    def _calculate_movement(self, start_loc, target_loc, progress):
        new_loc = self.location_manager.interpolate_position(start_loc, target_loc, progress)
        return new_loc, self.location_manager.get_travel_time(start_loc, new_loc)

    def _update_phase_progress(self, phase):
        phase["time_spent"] += 1
        return min(1.0, phase["time_spent"] / phase["total_time"])
