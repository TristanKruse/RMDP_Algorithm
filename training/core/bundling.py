def detect_bundles(route_plan, state, episode_stats):
    """
    Detects bundles in the current route plan.
    A bundle is defined as multiple orders being picked up at the same stop.
    """
    # Initialize tracker if it doesn't exist
    if "counted_bundles" not in episode_stats:
        episode_stats["counted_bundles"] = set()

    for vehicle_id, route in route_plan.items():
        if not route.sequence:
            continue

        # Check each stop in the route
        for _, pickups, _ in route.sequence:
            # If multiple orders are being picked up at the same stop, it's a bundle
            if len(pickups) > 1:
                # Create a unique identifier for this bundle (sorted tuple of order IDs)
                bundle_id = tuple(sorted(pickups))

                # Only count if we haven't seen this exact bundle before
                if bundle_id not in episode_stats["counted_bundles"]:
                    # Record bundle information
                    bundle_size = len(pickups)
                    episode_stats["bundles_formed"] += 1
                    episode_stats["bundle_sizes"].append(bundle_size)
                    episode_stats["bundled_orders"].update(pickups)
                    episode_stats["counted_bundles"].add(bundle_id)

                    # Check if all orders in this bundle are from the same restaurant
                    restaurant_ids = set()
                    for order_id in pickups:
                        order = next((o for o in state.orders if o.id == order_id), None)
                        if order and hasattr(order, "pickup_node_id"):
                            restaurant_ids.add(order.pickup_node_id.id)

                    # If only one restaurant ID, it's a same-restaurant bundle
                    if len(restaurant_ids) == 1:
                        episode_stats["same_restaurant_bundles"] += 1
