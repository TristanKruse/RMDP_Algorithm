# environment/route_processing/phase_management.py


class PhaseManagement:
    def __init__(self, location_manager):
        self.location_manager = location_manager

    def _initialize_vehicle_phase(self, order_id, order, current_loc):
        is_pickup = order.status == "pending"
        target_loc = order.pickup_location if is_pickup else order.delivery_location

        return {
            "order_id": order_id,
            "stage": "pickup" if is_pickup else "delivery",
            "total_time": max(0.001, self.location_manager.get_travel_time(current_loc, target_loc)),
            "time_spent": 0,
            "start_loc": current_loc,
            "target_loc": target_loc,
            "service_time_remaining": None,
            "is_servicing": False,
        }

    def _initialize_delivery_phase(self, order_id, current_loc, delivery_loc):
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
