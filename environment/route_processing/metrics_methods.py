# environment/route_processing/metrics_methods.py


class MetricsMethods:
    # === Metrics Methods ===
    def _initialize_metrics(self):
        """Initialize metrics dictionary"""
        return {"distance": 0, "deliveries": 0, "delays": [], "late_orders": set(), "delivered_orders": set()}

    def _update_metrics(self, metrics, route_metrics):
        """Update overall metrics with route metrics"""
        for key in metrics:
            if isinstance(metrics[key], (list, set)):
                if isinstance(metrics[key], list):
                    metrics[key].extend(route_metrics[key])
                else:
                    metrics[key].update(route_metrics[key])
            else:
                metrics[key] += route_metrics[key]
