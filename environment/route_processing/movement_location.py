# environment/route_processing/movement_location.py
import numpy as np
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

    # === Movement and Location Methods ===
    def _calculate_movement(self, start_loc, target_loc, progress):
        """Calculate new position and step distance"""
        new_loc = self.location_manager.interpolate_position(start_loc, target_loc, progress)
        step_distance = self.location_manager.get_travel_time(start_loc, new_loc)
        return new_loc, step_distance

    def _update_phase_progress(self, phase):
        """Update and return movement progress"""
        phase["time_spent"] += 1
        return min(1.0, phase["time_spent"] / phase["total_time"])


# class MovementLocation:
#     def __init__(self, location_manager):
#         self.location_manager = location_manager

#     def _calculate_movement(self, start_loc, target_loc, progress):
#         """Calculate new position and step distance"""
#         # Log initial positions and progress
#         logger.info(f"\n[MOVEMENT DEBUG] Calculating movement:")
#         logger.info(f"[MOVEMENT DEBUG] From: ({start_loc.x:.2f}, {start_loc.y:.2f})")
#         logger.info(f"[MOVEMENT DEBUG] To: ({target_loc.x:.2f}, {target_loc.y:.2f})")
#         logger.info(f"[MOVEMENT DEBUG] Current progress: {progress:.3f}")

#         # Calculate total distance
#         dx = target_loc.x - start_loc.x
#         dy = target_loc.y - start_loc.y
#         total_distance = np.sqrt(dx * dx + dy * dy)
#         logger.info(f"[MOVEMENT DEBUG] Total distance: {total_distance:.2f}")

#         # Calculate new position
#         new_loc = self.location_manager.interpolate_position(start_loc, target_loc, progress)
#         logger.info(f"[MOVEMENT DEBUG] New position: ({new_loc.x:.2f}, {new_loc.y:.2f})")

#         # Calculate step distance
#         step_distance = self.location_manager.get_travel_time(start_loc, new_loc)
#         logger.info(f"[MOVEMENT DEBUG] Step distance: {step_distance:.2f}")

#         return new_loc, step_distance

#     def _update_phase_progress(self, phase):
#         """Update and return movement progress"""
#         old_progress = phase["time_spent"] / phase["total_time"]
#         phase["time_spent"] += 1
#         new_progress = min(1.0, phase["time_spent"] / phase["total_time"])

#         logger.info(f"\n[PROGRESS DEBUG] Updating phase progress:")
#         logger.info(f"[PROGRESS DEBUG] Time spent: {phase['time_spent']}")
#         logger.info(f"[PROGRESS DEBUG] Total time: {phase['total_time']}")
#         logger.info(f"[PROGRESS DEBUG] Old progress: {old_progress:.3f}")
#         logger.info(f"[PROGRESS DEBUG] New progress: {new_progress:.3f}")
#         logger.info(f"[PROGRESS DEBUG] Progress increment: {(new_progress - old_progress):.3f}")

#         return new_progress
