// cheapest_insertion.hpp
#pragma once
#include <vector>
#include <set>
#include <limits>
#include <cmath>

struct Location {
    double x;
    double y;
};

struct Stop {
    int node_id;
    std::set<int> pickups;
    std::set<int> deliveries;
};

struct InsertionResult {
    std::vector<Stop> new_sequence;
    double insertion_cost;
    double delay;
    int vehicle_id;
};

class CheapestInsertion {
public:
    CheapestInsertion(
        double service_time,
        double vehicle_speed,
        double street_network_factor,
        int vehicle_capacity,
        double mean_prep_time,
        double prep_time_var,
        double delay_factor
    ) : service_time(service_time),
        vehicle_speed(vehicle_speed),
        street_network_factor(street_network_factor),
        vehicle_capacity(vehicle_capacity),
        mean_prep_time(mean_prep_time),
        prep_time_var(prep_time_var),
        delay_factor(delay_factor) {}

    InsertionResult find_best_insertion(
        const std::vector<Stop>& current_sequence,
        int order_id,
        const Location& restaurant_loc,
        const Location& customer_loc,
        double current_time,
        double deadline,
        double buffer = 0.0
    ) {
        InsertionResult best_result;
        best_result.insertion_cost = std::numeric_limits<double>::infinity();
        best_result.delay = std::numeric_limits<double>::infinity();

        // Handle empty sequence case
        if (current_sequence.empty()) {
            std::vector<Stop> new_sequence;
            Stop r_stop{0, {order_id}, {}};
            new_sequence.push_back(r_stop);
            Stop c_stop{1, {}, {order_id}};
            new_sequence.push_back(c_stop);
            
            double cost = evaluate_insertion(new_sequence, 0, 1, order_id, 
                                          restaurant_loc, customer_loc,
                                          current_time, deadline, buffer);
            return {new_sequence, cost, 0.0, -1};  // -1 indicates no specific vehicle
        }

        // Try all feasible insertion positions
        auto positions = get_feasible_insertions(current_sequence);
        for (const auto& [r_pos, c_pos] : positions) {
            std::vector<Stop> test_sequence = current_sequence;
            
            // Insert restaurant stop
            Stop r_stop{0, {order_id}, {}};
            test_sequence.insert(test_sequence.begin() + r_pos, r_stop);
            
            // Insert customer stop
            Stop c_stop{1, {}, {order_id}};
            test_sequence.insert(test_sequence.begin() + c_pos, c_stop);

            auto [cost, delay] = evaluate_insertion(test_sequence, r_pos, c_pos, 
                                                  order_id, restaurant_loc, customer_loc,
                                                  current_time, deadline, buffer);

            if (cost < best_result.insertion_cost) {
                best_result.new_sequence = test_sequence;
                best_result.insertion_cost = cost;
                best_result.delay = delay;
            }
        }

        return best_result;
    }

private:
    double service_time;
    double vehicle_speed;
    double street_network_factor;
    int vehicle_capacity;
    double mean_prep_time;
    double prep_time_var;
    double delay_factor;

    double calculate_travel_time(const Location& from, const Location& to) {
        double dx = to.x - from.x;
        double dy = to.y - from.y;
        double euclidean_distance = std::sqrt(dx*dx + dy*dy);
        double street_distance = euclidean_distance * street_network_factor;
        return street_distance / vehicle_speed;
    }

    bool is_feasible_capacity(const std::vector<Stop>& sequence, int r_pos, int c_pos) {
        int max_load = 0;
        int current_load = 0;
        
        for (int i = 0; i < sequence.size(); i++) {
            if (i == r_pos) current_load++;
            current_load += sequence[i].pickups.size();
            current_load -= sequence[i].deliveries.size();
            if (i == c_pos) current_load--;
            max_load = std::max(max_load, current_load);
        }
        return max_load <= vehicle_capacity;
    }

    std::vector<std::pair<int,int>> get_feasible_insertions(
        const std::vector<Stop>& sequence
    ) {
        std::vector<std::pair<int,int>> positions;
        int n = sequence.size();
        
        for (int r_pos = 0; r_pos <= n; ++r_pos) {
            for (int c_pos = r_pos + 1; c_pos <= n + 1; ++c_pos) {
                if (is_feasible_capacity(sequence, r_pos, c_pos)) {
                    positions.push_back({r_pos, c_pos});
                }
            }
        }
        return positions;
    }

    std::pair<double, double> evaluate_insertion(
        const std::vector<Stop>& sequence,
        int r_pos, int c_pos,
        int order_id,
        const Location& restaurant_loc,
        const Location& customer_loc,
        double current_time,
        double deadline,
        double buffer
    ) {
        double travel_cost = 0.0;
        double delay = 0.0;
        double time = current_time;
        Location prev_loc = {0, 0};  // Starting location

        for (size_t i = 0; i < sequence.size(); i++) {
            Location current_loc;
            bool is_pickup = sequence[i].pickups.count(order_id) > 0;
            bool is_delivery = sequence[i].deliveries.count(order_id) > 0;

            if (is_pickup) {
                current_loc = restaurant_loc;
                // Add preparation time consideration
                time = std::max(time, current_time + mean_prep_time);
            } else if (is_delivery) {
                current_loc = customer_loc;
                // Calculate delay with buffer
                double delivery_time = time + buffer;
                delay = std::max(0.0, delivery_time - deadline);
            } else {
                continue;
            }

            double travel_time = calculate_travel_time(prev_loc, current_loc);
            travel_cost += travel_time;
            time += travel_time + service_time;
            prev_loc = current_loc;
        }

        return {travel_cost + delay * delay_factor, delay};
    }
};