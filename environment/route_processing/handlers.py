# environment/route_processing/handlers.py
from environment.route_processing.phase_management import PhaseManagement
from environment.route_processing.service_time import ServiceTime
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(module)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create logger instance
logger = logging.getLogger(__name__)

class Handlers:
    def __init__(self, location_manager, service_time):
        self.location_manager = location_manager
        self.phase_management = PhaseManagement(location_manager)
        self.service_time = ServiceTime(service_time)

    def _are_all_orders_ready(self, order_manager, first_stop, current_time):
        """Check if all orders in a pickup stop are ready."""
        node_id, pickups, _ = first_stop
        for order_id in pickups:
            order = next((o for o in order_manager.active_orders if o.id == order_id), None)
            if not order or current_time < order.ready_time:
                return False
        return True

    def _handle_arrival(self, vehicle, orders, new_loc, current_time, order_manager=None, route_plan=None, pickup_orders=None):
        """Handle vehicle arrival at a location."""
        # If we're in pickup stage, check ready times
        if vehicle.current_phase["stage"] == "pickup":
            # Get bundle orders from both sources
            bundle_orders = set()
            if route_plan and vehicle.id in route_plan and route_plan[vehicle.id].sequence:
                first_stop = route_plan[vehicle.id].sequence[0]
                bundle_orders = first_stop[1]  # Get pickup set
            elif pickup_orders:  # Fallback to pickup_orders if route_plan extraction fails
                bundle_orders = set(pickup_orders)


            # Validate bundle information
            if not bundle_orders:
                logger.warning("No bundle orders found in either route plan or pickup orders")
                return new_loc, 0.0, 0.0, False

            if pickup_orders and bundle_orders != set(pickup_orders):
                logger.warning(f"Bundle mismatch - Route plan: {bundle_orders}, Pickup orders: {pickup_orders}")

            # Check readiness of all orders
            all_orders_ready = True
            for order_id in bundle_orders:
                order = next((o for o in order_manager.active_orders if o.id == order_id), None)
                if order:
                    if current_time < order.ready_time:
                        all_orders_ready = False
                        break
                else:
                    all_orders_ready = False
                    break

            if all_orders_ready:
                # Initialize service phase while maintaining complete bundle information
                service_phase = self.service_time._initialize_service(vehicle.current_phase)
                
                # Preserve all existing bundle information
                for key in ['is_bundle', 'bundle_orders', 'order_ids', 'initial_bundle_size']:
                    if key in vehicle.current_phase:
                        service_phase[key] = vehicle.current_phase[key]
                
                # Update with current bundle information
                service_phase.update({
                    "is_bundle": True,
                    "bundle_orders": bundle_orders,
                    "order_ids": bundle_orders
                })
                vehicle.current_phase = service_phase

                # Mark all orders in bundle as picked up
                for order_id in bundle_orders:
                    order = next((o for o in order_manager.active_orders if o.id == order_id), None)
                    if order:
                        order.status = "picked_up"
                        order.pickup_time = current_time
                        logger.info(f"Marked order {order_id} as picked up")
                
                return new_loc, self.location_manager.get_travel_time(vehicle.current_location, new_loc), 0.0, False
            else:
                return new_loc, 0.0, 0.0, False

        # For delivery arrivals
        elif vehicle.current_phase["stage"] == "delivery":
            service_phase = self.service_time._initialize_service(vehicle.current_phase)
            
            # Maintain complete bundle information through delivery
            if vehicle.current_phase.get("is_bundle"):
                # Preserve all bundle information
                for key in ['is_bundle', 'bundle_orders', 'order_ids', 'initial_bundle_size']:
                    if key in vehicle.current_phase:
                        service_phase[key] = vehicle.current_phase[key]
            
            vehicle.current_phase = service_phase
            return new_loc, self.location_manager.get_travel_time(vehicle.current_location, new_loc), 0.0, False

        return new_loc, self.location_manager.get_travel_time(vehicle.current_location, new_loc), 0.0, False

    def _handle_service_completion(self, vehicle, orders, current_loc, current_time, order_manager=None, route_plan=None, pickup_orders=None):

        if vehicle.current_phase["stage"] == "pickup":

            # Get bundle orders from all available sources
            bundle_orders = set()
            if route_plan and vehicle.id in route_plan and route_plan[vehicle.id].sequence:
                first_stop = route_plan[vehicle.id].sequence[0]
                bundle_orders = first_stop[1]  # Get pickup set
            elif pickup_orders:
                bundle_orders = set(pickup_orders)
            else:
                bundle_orders = vehicle.current_phase.get("bundle_orders", set())
            
            if not bundle_orders:  # Safety check
                return current_loc, 0.0, 0.0, False
                    
            # Mark all orders in bundle as picked up
            for order_id in bundle_orders:
                order = next((o for o in order_manager.active_orders if o.id == order_id), None)
                if order:
                    order.status = "picked_up"
                    order.pickup_time = current_time
            
            # Initialize delivery for first order
            first_order_id = min(bundle_orders)
            first_order = next((o for o in order_manager.active_orders if o.id == first_order_id), None)
            if not first_order:
                return current_loc, 0.0, 0.0, False
                
            vehicle.current_phase = self.phase_management._initialize_delivery_phase(
                first_order_id, current_loc, first_order.delivery_node_id
            )
            # Preserve all bundle information
            vehicle.current_phase.update({
                "is_bundle": True,
                "bundle_orders": bundle_orders,
                "order_ids": bundle_orders,
                "initial_bundle_size": len(bundle_orders)
            })
            return current_loc, 0.0, 0.0, False

        elif vehicle.current_phase["stage"] == "delivery":
            current_order_id = vehicle.current_phase.get("order_id")
            bundle_orders = vehicle.current_phase.get("bundle_orders", set())
            
            current_order = next((o for o in order_manager.active_orders if o.id == current_order_id), None)
            if current_order:
                current_order.status = "delivered"
                current_order.delivery_time = current_time
                delay = max(0, current_time - current_order.deadline)
                
                remaining_orders = bundle_orders - {current_order_id}
                if remaining_orders:
                    next_order_id = min(remaining_orders)
                    next_order = next((o for o in order_manager.active_orders if o.id == next_order_id), None)
                    if next_order:
                        vehicle.current_phase = self.phase_management._initialize_delivery_phase(
                            next_order_id, current_loc, next_order.delivery_node_id
                        )
                        # Preserve bundle information for remaining deliveries
                        vehicle.current_phase.update({
                            "is_bundle": True,
                            "bundle_orders": remaining_orders,
                            "order_ids": remaining_orders,
                            "initial_bundle_size": len(bundle_orders)
                        })
                        return current_loc, 0.0, delay, False  
                vehicle.current_phase = None
                return current_loc, 0.0, delay, True

            return current_loc, 0.0, 0.0, False

        return current_loc, 0.0, 0.0, False