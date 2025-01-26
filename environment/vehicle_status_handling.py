class VehicleStatusHandler:
    def __init__(self, vehicle_capacity: int):
        self.vehicle_capacity = vehicle_capacity

    def can_accept_orders(self, vehicle) -> bool:
        # Check vehicle capacity first
        if self.get_vehicle_load(vehicle) >= self.vehicle_capacity:
            return False

        # Idle vehicle can accept orders
        if not vehicle.current_phase:
            return True

        # Vehicle waiting at restaurant can accept orders
        if vehicle.current_phase["stage"] == "pickup" and vehicle.current_phase["is_servicing"]:
            return True

        # Vehicle just completed delivery can accept orders
        if vehicle.current_phase["stage"] == "delivery" and vehicle.current_phase["service_time_remaining"] <= 0:
            return True

        return False

    def get_vehicle_load(self, vehicle) -> int:
        return len(getattr(vehicle, "current_orders", []))
