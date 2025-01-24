# environment/route_processing/assignement_manager.py


# class AssignmentManager:
#     def __init__(self):
#         self.vehicle_assignments = {}

#     # === Assignment Management Methods ===
#     def _cleanup_stale_assignments(self, order_manager):
#         """Clean up stale assignments and return active orders"""
#         for vehicle_id, order_id in list(self.vehicle_assignments.items()):
#             if order_id is not None:
#                 order = next((o for o in order_manager.active_orders if o.id == order_id), None)
#                 if not order:
#                     # print(f"[DEBUG] Cleaning up stale assignment: Vehicle {vehicle_id}, Order {order_id}")
#                     self.vehicle_assignments[vehicle_id] = None
#         return set(assignment for assignment in self.vehicle_assignments.values() if assignment is not None)

#     def _assign_new_order(self, vehicle_id: int, order_id: int, active_orders: set):
#         """Assign a new order to a vehicle"""
#         if order_id not in active_orders:
#             self.vehicle_assignments[vehicle_id] = order_id
#             active_orders.add(order_id)
#             # print(f"[DEBUG] New assignment: Vehicle {vehicle_id} -> Order {order_id}")


class AssignmentManager:
    def __init__(self):
        self.vehicle_assignments = {}  # vehicle_id -> order_id
        self.assigned_orders = set()  # set of assigned order_ids

    def _cleanup_stale_assignments(self, order_manager):
        """Clean up stale assignments and return active orders"""
        for vehicle_id, order_id in list(self.vehicle_assignments.items()):
            if order_id is not None:
                order = next((o for o in order_manager.active_orders if o.id == order_id), None)
                if not order:
                    self.vehicle_assignments[vehicle_id] = None
                    self.assigned_orders.discard(order_id)
        return self.assigned_orders

    def _assign_new_order(self, vehicle_id: int, order_id: int, active_orders: set):
        """Assign a new order to a vehicle"""
        # Check if order is already assigned
        if order_id not in self.assigned_orders:
            self.vehicle_assignments[vehicle_id] = order_id
            self.assigned_orders.add(order_id)
            print(f"[DEBUG] New assignment: Vehicle {vehicle_id} -> Order {order_id}")
        else:
            print(f"[WARNING] Attempted to assign already assigned order {order_id} to vehicle {vehicle_id}")
