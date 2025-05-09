#  Converging, Training verbessern RL:

->Größte Problem, manchmal wird zu viele Postponed.

-> Stochastic deadlines noch zum Problem hinzufügen??? (ggf. dann auch in Literaturüberblick darauf eingehen, also unterschied zwischen den verschieden "RMDP's", weil nicht exakt gleiche Annahmen, sondern etwas unterschiedliche.)

-> ggf. Simulation bei Dispatch records anfangen, sodass hier die original Positionen genommen werden können? (problem, einige haben auch schon orders)
-> Problem, es wird nicht genug decision space durchsucht, 100 % postponement rate.
-> Dann wieder ohne seed probieren?? -> vielleicht stattdessen, seed incrementen jede 100 episoden oder so ähnlich.
- RL, algorithmus lernt schlechter zu werden, gleichzeitig steigt Volatilität
->Mit Grok einmal Logik hier durchgeheen, von Rewards in train.py zu rl_postponement.py etc.
->ACA: Wie ist find vehicleund assign order implementiert?  außerdem Postponed orders am Anfang, entsprechend müsste man die am Ende eigentlich nicht mehr rausnehmen?!
1. Test Cases "final festlegen" -> sinnvoll begründen
( episodes, demand, restaurants, customers, vehicle capacity, ...)
- Phase 1 einmal mit viel mehr Phasen durchlaufen lassen bspw. 3,000
- Bspw. Prep time var.
-> alles aufschreiben


-vielleicht komplizierteren Approach mit  PPO mal anschauen?

- vielleicht interessant zu Untersuchen previous postponements, postponement ratio & v.a. Anzahl Bundling (bspw. zweimal hintereinander customer, dann gebunddlet, als ein Bundle gezählt + Größe der Bundle messen) -> in Lernmetriken, die am Ende ausgegeben werden+

-> wieder hinzufügen randomisierter Seeds, -> schauen, ob dass das Lernen negativ beeinflusst.

2. multi objective Function erstellen:
- Einen Simplen und einen der eine komplexere hat, dann vergleichen, sicherstellen, dass man switchen kann und das beide gespeichert werden
->multi objective: idle rate, total travel, total delay


->Cleanup run episode, and train.py
->Timestep auf 15 Sekunden umwandeln statt einer Minute? -> feasible, dass meiste ist in train.py, es gibt ggf. aber auch hardcoded Sachen.


-Hier könnte man ggf. live positions rausfinden!!!!
Assignment inputs: orders to be assigned and candidate couriers. At a dispatch time,
orders to be assigned can be retrieved from Table 4 (File: dispatch waybill meituan.csv).
Details of candidate couriers eligible for the orders, e.g., the couriers’ current geographical
coordinates and the orders they are presently carrying, can be retrieved from Table
5 (File: dispatch rider meituan.csv). Table 4 contains 15,921 rows in total; Table
5 contains 62,044 rows in total.


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

