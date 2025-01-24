# environment/route_processing/phase_management.py


class PhaseManagement:
    def __init__(self, location_manager):
        self.location_manager = location_manager

    # === Phase Management Methods ===
    def _initialize_vehicle_phase(self, order_id, order, current_loc):
        """Initialize new phase for vehicle"""
        is_pickup = order.status == "pending"
        target_loc = order.pickup_location if is_pickup else order.delivery_location

        phase = {
            "order_id": order_id,
            "stage": "pickup" if is_pickup else "delivery",
            "total_time": 0,
            "time_spent": 0,
            "start_loc": current_loc,
            "target_loc": target_loc,
            "service_time_remaining": None,
            "is_servicing": False,
        }

        # Initialize total time
        phase["total_time"] = max(0.001, self.location_manager.get_travel_time(phase["start_loc"], phase["target_loc"]))

        return phase

    def _initialize_delivery_phase(self, order_id, current_loc, delivery_loc):
        """Initialize delivery phase after pickup"""
        return {
            "order_id": order_id,
            "stage": "delivery",
            "total_time": max(0.001, self.location_manager.get_travel_time(current_loc, delivery_loc)),
            "time_spent": 0,
            "start_loc": current_loc,
            "target_loc": delivery_loc,
            "service_time_remaining": None,
            "is_servicing": False,
        }
