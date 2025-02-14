# environment/route_processing/service_time.py


class ServiceTime:
    def __init__(self, service_time: float):
        self.service_time = service_time

    def _initialize_service(self, phase):
        """Initialize service phase while maintaining bundle information."""
        # Create new phase while preserving bundle information
        new_phase = phase.copy()  # Copy existing phase to maintain all info
        new_phase["service_time_remaining"] = self.service_time
        new_phase["is_servicing"] = True
        return new_phase


    def _process_service_time(self, phase):
        phase["service_time_remaining"] -= 1.0
        return phase["service_time_remaining"] <= 0
