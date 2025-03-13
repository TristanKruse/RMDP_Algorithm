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
            # ----- KPI Tracking -----
            waiting_orders = []
            # ----- KPI Tracking -----

            for order_id in bundle_orders:
                order = next((o for o in order_manager.active_orders if o.id == order_id), None)
                if order:
                    # ----- KPI Tracking -----
                    # Track driver arrival time at restaurant for this order
                    if not hasattr(order, 'driver_arrival_time'):
                        order.driver_arrival_time = current_time
                    # ----- KPI Tracking -----

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
                old_phase = "None" if vehicle.current_phase is None else vehicle.current_phase.get("stage", "unknown")
                vehicle.current_phase = service_phase
                new_phase = "None" if vehicle.current_phase is None else vehicle.current_phase.get("stage", "unknown")
                logger.info(f"Vehicle {vehicle.id} phase changed: {old_phase} -> {new_phase}")

                # Mark all orders in bundle as picked up
                for order_id in bundle_orders:
                    order = next((o for o in order_manager.active_orders if o.id == order_id), None)
                    if order:
                        order.status = "picked_up"
                        order.pickup_time = current_time

                        # ----- KPI Tracking -----
                        # Calculate times correctly
                        true_prep_time = order.ready_time - order.request_time  # Actual meal preparation time
                        order_wait_time = max(0, current_time - order.ready_time)  # Time order waited after being ready
                        total_time_to_pickup = current_time - order.request_time  # Total time from order to pickup

                        # Store these values
                        order.true_prep_time = true_prep_time
                        order.order_wait_time = order_wait_time
                        order.total_time_to_pickup = total_time_to_pickup

                        logger.info(f"Order {order_id} picked up: true prep time={true_prep_time:.1f} min, " +
                                f"wait time after ready={order_wait_time:.1f} min, " +
                                f"total time to pickup={total_time_to_pickup:.1f} min")
                        # ----- KPI Tracking -----

                        logger.info(f"Marked order {order_id} as picked up")
                
                return new_loc, self.location_manager.get_travel_time(vehicle.current_location, new_loc), 0.0, False
            else:
                # ----- KPI Tracking -----
                # Store the waiting time information
                for order_id in waiting_orders:
                    order = next((o for o in order_manager.active_orders if o.id == order_id), None)
                    if order:
                        # Mark that the driver is waiting for this order
                        order.driver_waiting = True
                        logger.info(f"Driver waiting for order {order_id} - ready in {order.ready_time - current_time:.1f} min")
                # ----- KPI Tracking -----
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

                    # ----- KPI Tracking -----
                    # Calculate prep and wait times
                    expected_prep_time = order.ready_time - order.request_time
                    actual_prep_time = current_time - order.request_time
                    
                    # Calculate wait time if the driver arrived before food was ready
                    driver_wait_time = 0.0
                    if hasattr(order, 'driver_arrival_time'):
                        driver_wait_time = max(0.0, order.ready_time - order.driver_arrival_time)
                    
                    # Store these values
                    order.expected_prep_time = expected_prep_time
                    order.actual_prep_time = actual_prep_time
                    order.driver_wait_time = driver_wait_time
                    # ----- KPI Tracking -----
            
            # Initialize delivery for first order
            first_order_id = min(bundle_orders)
            first_order = next((o for o in order_manager.active_orders if o.id == first_order_id), None)
            if not first_order:
                return current_loc, 0.0, 0.0, False
            old_phase = "None" if vehicle.current_phase is None else vehicle.current_phase.get("stage", "unknown")

            vehicle.current_phase = self.phase_management._initialize_delivery_phase(
                first_order_id, current_loc, first_order.delivery_node_id
            )
            new_phase = "None" if vehicle.current_phase is None else vehicle.current_phase.get("stage", "unknown")
            logger.info(f"Vehicle {vehicle.id} phase changed: {old_phase} -> {new_phase}")
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