# environment/route_processing/metrics_methods.py


class MetricsMethods:
    def _initialize_metrics(self):
        return {"distance": 0, "deliveries": 0, "delays": [], "late_orders": set(), "delivered_orders": set()}
