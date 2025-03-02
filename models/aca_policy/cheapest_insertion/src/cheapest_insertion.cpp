#include "cheapest_insertion.hpp"

namespace rmdp {

double CheapestInsertion::calculateTravelTime(const Location& from, const Location& to) const {
    const double dx = to.x - from.x;
    const double dy = to.y - from.y;
    const double euclidean_distance = std::sqrt(dx * dx + dy * dy);
    return (euclidean_distance * street_network_factor_) / vehicle_speed_;
}

double CheapestInsertion::evaluateInsertion(
    const std::vector<Stop>& sequence,
    size_t restaurant_pos,
    size_t customer_pos,
    const Location& restaurant_loc,
    const Location& customer_loc,
    double current_time,
    double deadline
) const {
    double time = current_time;
    Location current_loc = restaurant_loc; // Start from restaurant location

    // Calculate time to reach restaurant
    for (size_t i = 0; i < restaurant_pos; ++i) {
        time += service_time_; // Simplified for example
    }

    // Add restaurant service time
    time += service_time_;

    // Calculate time to reach customer
    for (size_t i = restaurant_pos; i < customer_pos; ++i) {
        time += service_time_;
    }

    // Add final delivery time
    time += service_time_;

    // Calculate delay
    double delay = std::max(0.0, time - deadline);
    
    return delay;
}

InsertionResult CheapestInsertion::findBestInsertion(
    const std::vector<Stop>& current_sequence,
    int order_id,
    const Location& restaurant_loc,
    const Location& customer_loc,
    double current_time,
    double deadline
) {
    double min_cost = std::numeric_limits<double>::infinity();
    size_t best_r_pos = 0;
    size_t best_c_pos = 0;
    
    // Try all possible insertion positions
    for (size_t r_pos = 0; r_pos <= current_sequence.size(); ++r_pos) {
        for (size_t c_pos = r_pos + 1; c_pos <= current_sequence.size() + 1; ++c_pos) {
            double cost = evaluateInsertion(
                current_sequence, r_pos, c_pos,
                restaurant_loc, customer_loc,
                current_time, deadline
            );
            
            if (cost < min_cost) {
                min_cost = cost;
                best_r_pos = r_pos;
                best_c_pos = c_pos;
            }
        }
    }

    // Create new sequence with insertions
    std::vector<Stop> new_sequence = current_sequence;
    
    // Create restaurant stop
    Stop restaurant_stop;
    restaurant_stop.node_id = -1;  // Would need actual restaurant node ID
    restaurant_stop.pickups = {order_id};
    
    // Create customer stop
    Stop customer_stop;
    customer_stop.node_id = -1;  // Would need actual customer node ID
    customer_stop.deliveries = {order_id};

    // Insert the stops
    new_sequence.insert(new_sequence.begin() + best_r_pos, restaurant_stop);
    new_sequence.insert(new_sequence.begin() + best_c_pos, customer_stop);

    InsertionResult result;
    result.vehicle_id = 0;  // Would need actual vehicle ID
    result.new_sequence = new_sequence;
    result.insertion_cost = min_cost;
    return result;
}

} // namespace rmdp