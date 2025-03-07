def geo_to_sim_coords(lat, lng, geo_bounds):
    """
    Convert geographic coordinates to simulation coordinates
    
    Args:
        lat: Latitude value
        lng: Longitude value
        geo_bounds: Dictionary containing coordinate system information
            with keys: min_lat, max_lat, min_lng, max_lng, width_km, height_km
    
    Returns:
        Tuple (x, y) of simulation coordinates
    """
    if not geo_bounds:
        # If no bounds provided, return original coordinates
        return lat, lng
    
    # Convert from geographic coordinates to normalized simulation coordinates (0-1)
    norm_x = (lng - geo_bounds["min_lng"]) / (geo_bounds["max_lng"] - geo_bounds["min_lng"])
    norm_y = (lat - geo_bounds["min_lat"]) / (geo_bounds["max_lat"] - geo_bounds["min_lat"])
    
    # Scale to simulation dimensions
    sim_x = norm_x * geo_bounds["width_km"]
    sim_y = norm_y * geo_bounds["height_km"]
    
    return sim_x, sim_y