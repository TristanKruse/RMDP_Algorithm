-Simulation weiterfüttern mit Daten, Restaurant positions bspw.
    -> einfach mal für ein Business District alles rein.


1. Geographic Data (Highly Feasible)

Restaurant locations: You could replace your randomly generated locations with actual Meituan restaurant coordinates
Business district modeling: Define different business districts based on Meituan's data to create multiple scenario configurations
Vehicle starting positions: Use real delivery driver starting positions


2. Order Characteristics (Mostly Feasible)

Order volumes: Match total order counts to real data
Order distribution: Adjust time-based order patterns (instead of Poisson)
Restaurant popularity: Weight order generation by restaurant popularity from Meituan



3. Timing Parameters (Feasible with Adjustments)

Preparation times: Use restaurant-specific prep times from real data
Delivery windows: Adjust deadlines based on actual consumer expectations



4. Vehicle Parameters (Challenging but Possible)

Fleet size: Match to real fleet sizes
Vehicle capacities: Adjust based on data.





-Debugen von ACA, warum Ergebnisse nicht klar besser, so wie bei Ulmer?

-postponement Logik überprüfen. -> hängt sich auf, wenn man postponement miteinführt. 

- "seasonal" demand einbauen, also über den Tag hinweg verschiedener Demand -> orientieren an Meituan.
- vehicle capacity, considered nur current load und nicht total route length im ACA
- maybe caching in time_utils.py?
- schauen wo node objekt falsch gefüllt wird, weil das extracten der Location kostet Zeit.

- Weitere Skripte durchgehen, effizienter machen und an Ulmers Implementierung anpassen.
    -postponement
    -route utils.



Ziele:
Versuchen so effizient zu kriegen, dass einmal Resultate reingeholt werden können via Python Implementation only
Dann c++ Implementierung machen?!


# ACA
- Improvement based on Meituan, add preselection of potential vehicles, to cut down computational time, also bspw. mit den MD-Scores top ten evaluaten, wie es da aufgeht, weil ansonten hunderte vehicles zu testen vermutlich nicht sinnvoll, oder andere ko.Kriterien noch einführen, welche Vehicles nicht getestet werden. ->bei find_vehicle in ACA

hängt sich be i10 unassigned orders auf
wie schneller machen? Wo sind die Bottlenecks? -> Sachen in C++ code umprogrammieren.
Kleine Tests erstellen, die die Funktionsweise testen, bspw. ob bundling funktioniert


-->Dann mal testen mit Ulmer daten, ob gleiche/ähnliche Resultate.


Cheapest Insertion Algorithm (in vehicle_ops.py):

---> Check what we need for in and output.


Route Cost Calculation (in time_utils.py):

Called frequently during route evaluation
Heavy on floating-point operations
C++ benefits:

Native arrays for better cache utilization
More efficient floating-point operations
Potential for parallel processing of route segments




Travel Time Calculations (in location_manager.py):

Basic but extremely frequent operations
Perfect candidate for SIMD optimization in C++
Could process multiple distance calculations simultaneously


Order Sequence Generation (in route_utils.py):


Permutation generation is computationally intensive
C++ could provide:

More efficient permutation algorithms
Better memory management
Bit manipulation for faster sequence generation




Bundle Validation (in order_validation.py):


Frequent operation during route construction
C++ could optimize with:

Bitsets for state tracking
More efficient data structures
Better branching optimization
























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

# Pre delivery: 
-take demand generation from the data
-vehicle fleet from the data
-working hours from the data
-Take Restaurants positioning from the data
-Vehicle speed from the data
