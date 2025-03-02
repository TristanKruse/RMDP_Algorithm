#pragma once
#include <vector>
#include <cmath>
#include <limits>

namespace rmdp {

struct Location {
    double x;
    double y;
    
    Location() : x(0), y(0) {}
    Location(double x_, double y_) : x(x_), y(y_) {}
};

struct Stop {
    int node_id;
    std::vector<int> pickups;
    std::vector<int> deliveries;
};

struct Route {
    std::vector<Stop> sequence;
    double total_distance;
    double total_time;
};

struct InsertionResult {
    int vehicle_id;
    std::vector<Stop> new_sequence;
    double insertion_cost;
};

class CheapestInsertion {  // This class name needs to match what we're binding
public:
    CheapestInsertion(double service_time, double vehicle_speed, double street_network_factor)
        : service_time_(service_time), 
          vehicle_speed_(vehicle_speed),
          street_network_factor_(street_network_factor) {}

    // Main function that will be called from Python
    InsertionResult findBestInsertion(
        const std::vector<Stop>& current_sequence,
        int order_id,
        const Location& restaurant_loc,
        const Location& customer_loc,
        double current_time,
        double deadline
    );

private:
    double service_time_;
    double vehicle_speed_;
    double street_network_factor_;

    double calculateTravelTime(const Location& from, const Location& to) const;
    double evaluateInsertion(
        const std::vector<Stop>& sequence,
        size_t restaurant_pos,
        size_t customer_pos,
        const Location& restaurant_loc,
        const Location& customer_loc,
        double current_time,
        double deadline
    ) const;
};

} // namespace rmdp