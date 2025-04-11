# PLAN
1. alle offenen Punkte durchgehen
2. Trainieren
3. Multi Objective Function implementieren
3. Auswertung bauen, einmal alle Szenarien durchrechen
    I. Benchmarken fastest ACA,
    II. Benchmarken best ACA
    III. Benchmarken simple value function
    IV. Multiple Objective value fucntion
4. Ergebnisse aufschreiben, Graphen etc.


#  Converging, Training verbessern RL:

(loss tracking funktioniert noch nicht richtig, schwankt immer, sollte aber komplette Historie haben.)

1. I would like to find out how the losses are developing, i.e.  I would like to somehow plot them in the end of a model run through. -> plotten im train_rl.py


3. I would like to find out if the rewards are correctly maximized? Since currently to me it seems like we are minimizing them, we get worse and wors

4. I would like to find out, if the timeout rewards are applied correctly, since they don't seem to show for some reason, in my print out statem

5. I would like to find out, when the replay buffer is being eptied, making sure it's not eptied after each episode, but carries the experiences forward ...

->try out higher learning rate e.g. 0.005

-> Dann wieder ohne seed probieren?? -> vielleicht stattdessen, seed incrementen jede 100 episoden oder so ähnlich.



- auf manchen Szenarios rl funktiniert nicht.
- RL, algorithmus lernt schlechter zu werden, gleichzeitig steigt Volatilität
->Mit Grok einmal Logik hier durchgeheen, von Rewards in train.py zu rl_postponement.py etc.


Target Network: True DDQN uses a separate target network for next_q_values to stabilize learning, which isn't implemented here.

->ACA: Wie ist find vehicleund assign order implementiert?  außerdem Postponed orders am Anfang, entsprechend müsste man die am Ende eigentlich nicht mehr rausnehmen?!



1. Test Cases "final festlegen" -> sinnvoll begründen
( episodes, demand, restaurants, customers, vehicle capacity, ...)
- Phase 1 einmal mit viel mehr Phasen durchlaufen lassen bspw. 3,000
- Bspw. Prep time var.
-> alles aufschreiben


2.
Simon Notes:
->Bundling potential, bei Meituan schauen. 
-> schauen, was im Kleinen rauskommt. (genaue Benchmark)
-> viel früher auf die Meituan Daten gehen, den Ansatz müsste ich schlagen, ... -> nochmal darauf schauen, ...


3. Hyperparameter tuning: Entscheiden, wissenschaftlich welchen Buffer nehmen für rl
sample average approximation (SAA) on the restricted policy class. See Fu (2015) 
learning_rate (0.001), discount_factor (1.0), and batch_size


-vielleicht komplizierteren Approach mit  PPO mal anschauen?
2. collect pros and cons for different architectures and maybe redo them. -> more research, what rl-model to choose (Deep-Q Network/Double Deep Q, Soft Actor cirtic, vfa mit large neighbourhood, ...)

- vielleicht interessant zu Untersuchen previous postponements, postponement ratio & v.a. Anzahl Bundling (bspw. zweimal hintereinander customer, dann gebunddlet, als ein Bundle gezählt + Größe der Bundle messen) -> in Lernmetriken, die am Ende ausgegeben werden+

-> wieder hinzufügen randomisierter Seeds, -> schauen, ob dass das Lernen negativ beeinflusst.

2. multi objective Function erstellen:
- Einen Simplen und einen der eine komplexere hat, dann vergleichen, sicherstellen, dass man switchen kann und das beide gespeichert werden
->multi objective: idle rate, total travel, total delay


-> Bundling Bonus -> aber gleichzeitig sicherstellen, dass gutes Bundling also irgendwie total travel time reduzieren.


->Cleanup run episode, and train.py
->Timestep auf 15 Sekunden umwandeln statt einer Minute? -> feasible, dass meiste ist in train.py, es gibt ggf. aber auch hardcoded Sachen.


-Hier könnte man ggf. live positions rausfinden!!!!
Assignment inputs: orders to be assigned and candidate couriers. At a dispatch time,
orders to be assigned can be retrieved from Table 4 (File: dispatch waybill meituan.csv).
Details of candidate couriers eligible for the orders, e.g., the couriers’ current geographical
coordinates and the orders they are presently carrying, can be retrieved from Table
5 (File: dispatch rider meituan.csv). Table 4 contains 15,921 rows in total; Table
5 contains 62,044 rows in total.


Notes:
->assumed delivery window is between 10 and 120 minutes, otherwise use default gamma dist.  (see order generator)
->vehicle capacities ...
->ggf. vehicles langsamer machen
->Limiting permutations, when bigger 4 (see route_utils), um es computationally thesable zu machen.
->Aca hyperparameters relevant, v.a. auch, wie lange maximal postponed werden darf, bei 15 min. signifikant schlechtere Ergebnisse wie bei 10 min.
->Wichtig auch, accuracy von mean prep time und deadline, (es geht, ggf. nochmal dispatch to dispatch anschauen.)
->Travel speeds differ significantly based on time of day, -> see analysis -> dispatch to dispatch mean ca. 9.5 -> Haversine distance (along earth not euclidean) -> taking this and then without the service time, to get an accurate speed.
->exploration biased towards not postponeing (90 % of time choses not to postpone)
->ohne Seed converged auch nicht.
->nehmen die actual delays nicht into account für die Rewards (Man könnte noch überlegen, ob das irgendwie möglich wäre)
-> Für RL wird for den ACA loop einmal für die Orders gechecked, ob postponenen oder nicht.
-> Man kann nicht auf den ersten Platz inserten, außer Bundling.


# ACA

- maybe caching in time_utils.py?
- schauen wo node objekt falsch gefüllt wird, weil das extracten der Location kostet Zeit.
- Weitere Skripte durchgehen, effizienter machen und an Ulmers Implementierung anpassen.
    -postponement
    -route utils.
Ziele:
Versuchen so effizient zu kriegen, dass einmal Resultate reingeholt werden können via Python Implementation only
Dann c++ Implementierung machen?!

- Improvement based on Meituan, add preselection of potential vehicles, to cut down computational time, also bspw. mit den MD-Scores top ten evaluaten, wie es da aufgeht, weil ansonten hunderte vehicles zu testen vermutlich nicht sinnvoll, oder andere ko.Kriterien noch einführen, welche Vehicles nicht getestet werden. ->bei find_vehicle in ACA

Cheapest Insertion Algorithm (in vehicle_ops.py)
---> Check what we need for in and output
Route Cost Calculation (in time_utils.py)
Travel Time Calculations (in location_manager.py)
Order Sequence Generation (in route_utils.py)
Bundle Validation (in order_validation.py)

# Back to Restaurant issue: 
speed issues, zurück zu schnelle, manchmal teleportation.
ggf. wird genau in dem Moment eine Order assigned. oder es wird nicht richtig geresettet (bspw. current location/destination)

# mit dynamischer Zuweisung machen?
Further, sending the closest driver may limit the opportunities to assign orders to drivers that are already en route to a
nearby restaurant.
-> also muss es doch möglich sein, dass auch Vehicles die schon unterwegs sind eine Order zugeteilt kriegen.
-> ja, nur kann der Fahrer seine Route nichtmehr ändern
Aber in der Praxis muss Fahrer dem ja zustimmen oder nicht


# decision point logik einführen, entweder bundlen oder delivern, oder zum nächsten Restaurant???
# If the following conditions
hold, a decision is feasible:
1. The arrival time of the first entry of each route
θ ∈ Θk remains the same.
2. The value VD stays unaltered for all orders D
with VD > 0 (assignments are permanent).
3. The sequencing in Θxk
reflects the driver’s routing routine.


# ggf. umprogrammieren, dass vehicle standby gegeben werden kann, von Algorithmus
# wird hier immer richtig inserted?, mal double checken.

# vehicle ist schon im route plan, entsprechend könnte man Decision State noch optimieren.
















Yeah, it seems to me that this is due to the sequentially of orders being considered and that we are not really considering the delay of orders that aren't assigned?!

E.g., when running our analyze bundling opportunity analysis, as results we see that bundling is indeed more effective: 
2025-03-27 13:14:36 - train.py:1455 - INFO - BUNDLE-ANALYSIS: Bundled planned times has 3 entries
2025-03-27 13:14:36 - train.py:1458 - INFO - BUNDLE-ANALYSIS: Bundled - Order 20 planned arrival: 430.48
2025-03-27 13:14:36 - train.py:1466 - INFO -   Deadline: 219.01, Margin: -211.48
2025-03-27 13:14:36 - train.py:1458 - INFO - BUNDLE-ANALYSIS: Bundled - Order 21 planned arrival: 437.74
2025-03-27 13:14:36 - train.py:1466 - INFO -   Deadline: 227.84, Margin: -209.90
2025-03-27 13:14:36 - train.py:1458 - INFO - BUNDLE-ANALYSIS: Bundled - Order 22 planned arrival: 452.64
2025-03-27 13:14:36 - train.py:1466 - INFO -   Deadline: 229.32, Margin: -223.32
2025-03-27 13:14:36 - train.py:1471 - INFO - BUNDLE-ANALYSIS: Sequential planned times has 3 entries
2025-03-27 13:14:36 - train.py:1474 - INFO - BUNDLE-ANALYSIS: Sequential - Order 20 planned arrival: 430.48
2025-03-27 13:14:36 - train.py:1482 - INFO -   Deadline: 219.01, Margin: -211.48
2025-03-27 13:14:36 - train.py:1474 - INFO - BUNDLE-ANALYSIS: Sequential - Order 21 planned arrival: 452.05
2025-03-27 13:14:36 - train.py:1482 - INFO -   Deadline: 227.84, Margin: -224.21
2025-03-27 13:14:36 - train.py:1474 - INFO - BUNDLE-ANALYSIS: Sequential - Order 22 planned arrival: 468.96
2025-03-27 13:14:36 - train.py:1482 - INFO -   Deadline: 229.32, Margin: -239.64  ###### So, we just have to deeply understand the logic behind our algorithm and how it's implemented to find out, why this is not considered.

Cause in the end we always try to build the most efficient route and the idea is to first generate all the possible sequences in which we can assign our order, which is done in the solve method of the ACA class. 
Then we check for each order in that sequence, whether to postpone it or not, and if we don't postpone we then look into the assignment, so for which vehicle best to assign the order and where exactly in the current route of the vehicle.

So, this method is called in the solve function:                 # 12. Find best vehicle assignment
                assignment = self.vehicle_ops.find_vehicle(
                    candidate_route, order_id, buffer=self.buffer, state=state_dict
                )

This method looks like so: 
    def find_vehicle(self, route_plan: dict, order_id: int, buffer: float, state: dict) -> Optional[VehicleAssignment]:
        """Find best vehicle for order insertion using cheapest insertion."""
        best_assignment = None
        min_total_cost = float('inf')
        max_slack = -float('inf')  # Track best slack for tie-breaking
        
        # Get order info
        order_info = state["unassigned_orders"].get(order_id)
        if not order_info:
            return None

        # Get locations
        restaurant_node = order_info["pickup_node_id"]
        customer_node = order_info["delivery_node_id"]
        
        vehicles_checked = 0
        vehicles_at_capacity = 0
        vehicles_no_insertions = 0
        
        for vehicle_id, route in route_plan.items():
            vehicles_checked += 1
            
            # Skip vehicles at capacity
            active_orders = self._count_active_orders(route)
            if active_orders >= self.vehicle_capacity:
                vehicles_at_capacity += 1
                continue

            assignment = self._evaluate_vehicle_assignment(
                vehicle_id, route, order_id,
                restaurant_node, customer_node, buffer, state
                # current_best_delay=min_total_cost
            )
            
            if assignment:
                # Case 1: Better delay - always take it
                if assignment.delay < min_total_cost:
                    min_total_cost = assignment.delay
                    best_assignment = assignment
                    # Calculate slack for this assignment
                    test_route = Route(
                        vehicle_id=vehicle_id,
                        sequence=assignment.tentative_route,
                        total_distance=0.0,
                        total_time=0.0
                    )
                    max_slack = self.time_calculator._calculate_slack(state, {vehicle_id: test_route})
                                            
                # Case 2: Same delay - tie-breaking with slack
                elif assignment.delay == min_total_cost:
                    # Calculate slack for this assignment
                    test_route = Route(
                        vehicle_id=vehicle_id,
                        sequence=assignment.tentative_route,
                        total_distance=0.0,
                        total_time=0.0
                    )
                    current_slack = self.time_calculator._calculate_slack(state, {vehicle_id: test_route})
                    
                    # If better slack, update best assignment
                    if current_slack > max_slack:
                        max_slack = current_slack
                        best_assignment = assignment
            else:
                vehicles_no_insertions += 1
        
        return best_assignment


Furthermore, we then evaluate the vehicle using this method:    def _evaluate_vehicle_assignment(
        self, 
        vehicle_id: int,
        route: Route,
        order_id: int,
        restaurant_node: Node,
        customer_node: Node,
        buffer: float,
        state: dict
    ) -> Optional[VehicleAssignment]:
        """Evaluate all possible insertions of the order for a vehicle and find the best one."""
        best_insertion = None
        min_delay = float('inf')

        # Get all feasible insertion positions
        positions = self._get_feasible_insertions(route)
        
        # Try each feasible position
        for r_pos, c_pos in positions:
            # Create test route with new order inserted
            test_route = self._create_test_route(
                route=route,
                order_id=order_id,
                restaurant_node=restaurant_node,
                customer_node=customer_node,
                r_pos=r_pos,
                c_pos=c_pos
            )
            

            # Calculate delay for this insertion
            delay = self.time_calculator._calculate_delay(
                state=state,
                route_plan={vehicle_id: test_route},
                buffer=buffer
            )

            # Update best insertion if this one is better
            if delay < min_delay:
                min_delay = delay
                best_insertion = VehicleAssignment(
                    vehicle_id=vehicle_id,
                    tentative_route=test_route.sequence,
                    delay=delay
                )

        return best_insertion

######## And in this method we of course then call the calculate_delay method from the time_utils.py: 
    def _calculate_delay(self, state: dict, route_plan: Dict[int, Route], buffer: float = 0.0) -> float:
        """Calculate Δ(S) = ∑max{0, (aD + b) - (tD + t̄)}"""

        total_delay = 0.0
        per_order_delays = {}  # Store delays per order for updating order.current_estimated_delay

        # First get planned arrival times for all stops
        planned_times = self._calculate_planned_arrival_times(state, route_plan)
        
        # Then calculate delays
        for route in route_plan.values():
            for _, _, deliveries in route.sequence:
                for order_id in deliveries:
                    if order_id in planned_times:
                        # Find the order in either unassigned_orders or the main orders list
                        order_info = None
                        if order_id in state.get("unassigned_orders", {}):
                            order_info = state["unassigned_orders"][order_id]
                        else:
                            # Look for the order in the main orders list
                            for order in state.get("orders", []):
                                if order.id == order_id:
                                    # Create similar structure to unassigned_orders for consistency
                                    order_info = {"request_time": order.request_time}
                                    break
                        
                        # If we found order info, calculate delay
                        if order_info:
                            planned_arrival = planned_times[order_id]
                            request_time = order_info["request_time"]
                            
                            # Add buffer to planned arrival time for the total delay calculation
                            buffered_arrival = planned_arrival + buffer
                            delay = max(0, buffered_arrival - (request_time + self.delivery_window))
                            total_delay += delay
                            
                            # Calculate estimated delay WITHOUT buffer for more accurate estimation
                            estimated_delay = max(0, planned_arrival - (request_time + self.delivery_window))
                            per_order_delays[order_id] = estimated_delay
                                
        return total_delay

######### and in here we are calling for example the planned_arrival times method: (see pasted)




So, in theory even if we have no delay for both a route where we drive back to the restaurant after each delivery and a route where we bundle the orders, even then we should still have lower slack in general.


For the slack calculation we use the following method in the time_utils.py:    def _calculate_slack(self, state: dict, route_plan: Dict[int, Route]) -> float:
        """Calculate total slack Slack(S,Θ) in route plan.
        
        Slack represents the sum of time margins between planned arrivals 
        and deadlines. Higher slack indicates more flexibility.
        
        Following Ulmer et al.: Slack(S,Θ) = ∑ max{0, (tD + t̄) - aD}
        Where:
        - tD: Request time
        - t̄: Delivery window
        - aD: Planned arrival time
        """
        total_slack = 0.0
        
        # Get planned arrival times using our existing method
        planned_times = self._calculate_planned_arrival_times(state, route_plan)
        
        # Calculate slack for each delivery
        for route in route_plan.values():
            if not route.sequence:
                continue
                
            for _, _, deliveries in route.sequence:
                for order_id in deliveries:
                    order_info = state["unassigned_orders"].get(order_id)
                    if order_info and order_id in planned_times:
                        deadline = order_info["request_time"] + self.delivery_window
                        planned_arrival = planned_times[order_id]
                        slack = max(0.0, deadline - planned_arrival)
                        total_slack += slack
        
        return total_slack


########## Furthermore, I'm not sure if we maybe shouldn't call calculate costs instead of calculate dealy when evaluating vehicle assignments, as this also encompasses the stochastic costs, what do you think?