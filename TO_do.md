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



- Ontime wird nicht mehr richtig ausgegeben.

- teilweise positiven reward, generell nachdenken, ab wann updates immer Sinn ergeben, ggf. immer wenn eine Order delivered wurde anstelle von nach jedem Timestep??? 


2. collect pros and cons for different architectures and maybe redo them.

- vielleicht interessant zu Untersuchen previous postponements, postponement ratio.
Go through each of these points:
3. Batch Size: The batch_size parameter in _update_model determines how many experiences are used in each update. If it's too small, learning can be unstable.
5. Replay Buffer Size: Make sure the replay buffer is large enough to store a diverse set of experiences.


ASK CLAUDE:
For the large neighbourhood search I was thinking more about the state features, so that we could maybe generalize the geographic factors, that it would search for how near is good for postponeing ...
target = reward + discount_factor * max_future_value -> miteinbeziehen? gerade angeblich target = reward (kein long term learning.)
->Deep RL to learn important features???

-> wieder hinzufügen randomisierter Seeds, 

2. multi objective Function erstellen:
- Einen Simplen und einen der eine komplexere hat, dann vergleichen, sicherstellen, dass man switchen kann und das beide gespeichert werden
->multi objective: idle rate, total travel, total delay


-> Bundling Bonus -> aber gleichzeitig sicherstellen, dass gutes Bundling also irgendwie total travel time reduzieren.










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
