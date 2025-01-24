# environment/route_processing/handlers.py
from environment.route_processing.phase_management import PhaseManagement
from environment.route_processing.service_time import ServiceTime


class Handlers:
    def __init__(self, location_manager, service_time):
        self.location_manager = location_manager
        self.service_time = service_time
        self.phase_management = PhaseManagement(location_manager)
        self.service_time = ServiceTime(service_time)

    # def _handle_arrival(self, vehicle, order, new_loc, current_time, order_id):
    #     """Handle vehicle arrival at destination."""
    #     if vehicle.current_phase["stage"] == "pickup":
    #         if not order.ready_time or current_time >= order.ready_time:
    #             # print(f"[Vehicle {vehicle.id}] Starting pickup service at t={current_time}")
    #             vehicle.current_phase = self.service_time._initialize_service(vehicle.current_phase)
    #         else:
    #             # print(f"[Order {order.id}] Waiting at restaurant - food not ready yet")
    #             vehicle.current_phase["time_spent"] = vehicle.current_phase["total_time"]
    #     else:  # arrived for delivery
    #         # print(f"[Vehicle {vehicle.id}] Starting delivery service at t={current_time}")
    #         vehicle.current_phase = self.service_time._initialize_service(vehicle.current_phase)

    #     return new_loc, self.location_manager.get_travel_time(vehicle.current_location, new_loc), 0.0, False, [order_id]

    # def _handle_service_completion(self, vehicle, order, current_loc, current_time):
    #     """Handle completion of service at pickup or delivery."""
    #     if vehicle.current_phase["stage"] == "pickup":
    #         # print(f"[Order {order.id}] Pickup service completed at t={current_time}")
    #         order.status = "picked_up"
    #         order.pickup_time = current_time

    #         # Initialize delivery phase
    #         vehicle.current_phase = self.phase_management._initialize_delivery_phase(
    #             order.id, current_loc, order.delivery_location
    #         )
    #         return current_loc, 0.0, 0.0, False, [order.id]
    #     else:  # delivery service completed
    #         # print(f"[Order {order.id}] Delivery service completed at t={current_time}")
    #         order.status = "delivered"
    #         order.delivery_time = current_time
    #         delay = max(0, current_time - order.deadline)
    #         # if delay > 0:
    #         #     print(f"[Order {order.id}] Delivery delayed by {delay:.1f} minutes")
    #         vehicle.current_phase = None
    #         return current_loc, 0.0, delay, True, []

    def _handle_arrival(self, vehicle, order, new_loc, current_time, order_id):
        """Handle vehicle arrival at destination."""
        if vehicle.current_phase["stage"] == "pickup":
            if not order.ready_time or current_time >= order.ready_time:
                vehicle.current_phase = self.service_time._initialize_service(vehicle.current_phase)
            else:
                vehicle.current_phase["time_spent"] = vehicle.current_phase["total_time"]
        else:  # arrived for delivery
            vehicle.current_phase = self.service_time._initialize_service(vehicle.current_phase)

        return new_loc, self.location_manager.get_travel_time(vehicle.current_location, new_loc), 0.0, False

    def _handle_service_completion(self, vehicle, order, current_loc, current_time):
        """Handle completion of service at pickup or delivery."""
        if vehicle.current_phase["stage"] == "pickup":
            order.status = "picked_up"
            order.pickup_time = current_time
            vehicle.current_phase = self.phase_management._initialize_delivery_phase(
                order.id, current_loc, order.delivery_location
            )
            return current_loc, 0.0, 0.0, False
        else:  # delivery service completed
            order.status = "delivered"
            order.delivery_time = current_time
            delay = max(0, current_time - order.deadline)
            vehicle.current_phase = None
            return current_loc, 0.0, delay, True
