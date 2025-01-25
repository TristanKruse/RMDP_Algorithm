# environment/route_processing/handlers.py
from environment.route_processing.phase_management import PhaseManagement
from environment.route_processing.service_time import ServiceTime


class Handlers:
    def __init__(self, location_manager, service_time):
        self.location_manager = location_manager
        self.phase_management = PhaseManagement(location_manager)
        self.service_time = ServiceTime(service_time)

    def _handle_arrival(self, vehicle, order, new_loc, current_time):
        if vehicle.current_phase["stage"] == "pickup":
            if current_time >= (order.ready_time or 0):
                vehicle.current_phase = self.service_time._initialize_service(vehicle.current_phase)
            else:
                vehicle.current_phase["time_spent"] = vehicle.current_phase["total_time"]
        else:
            vehicle.current_phase = self.service_time._initialize_service(vehicle.current_phase)

        return new_loc, self.location_manager.get_travel_time(vehicle.current_location, new_loc), 0.0, False

    def _handle_service_completion(self, vehicle, order, current_loc, current_time):
        if vehicle.current_phase["stage"] == "pickup":
            order.status, order.pickup_time = "picked_up", current_time
            vehicle.current_phase = self.phase_management._initialize_delivery_phase(
                order.id, current_loc, order.delivery_location
            )
            return current_loc, 0.0, 0.0, False

        order.status, order.delivery_time = "delivered", current_time
        delay = max(0, current_time - order.deadline)
        vehicle.current_phase = None
        return current_loc, 0.0, delay, True
