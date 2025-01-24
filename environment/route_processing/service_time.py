# environment/route_processing/service_time.py


class ServiceTime:
    def __init__(self, service_time: float):
        """Initialize ServiceTime with service time parameter"""
        self.service_time = service_time

    # === Service Time Methods ===
    def _initialize_service(self, phase):
        """Initialize service at location"""
        phase["service_time_remaining"] = self.service_time
        phase["is_servicing"] = True
        return phase

    def _process_service_time(self, phase):
        """Process service time and return if complete"""
        phase["service_time_remaining"] -= 1.0
        return phase["service_time_remaining"] <= 0
