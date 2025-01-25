
speed issues, zurück zu schnelle, manchmal teleportation?

ggf. wird genau in dem Moment eine Order assigned. oder es wird nicht richtig geresettet (bspw. current location/destination)


# hin zu langsam, stimmt irgendwie nicht mit expected speed überein.

# moving to the nearest “unoccupied” restaurant
# when there are no orders to assign to a vehicle


# decision point logik einführen, entweder bundlen oder delivern, oder zum nächsten Restaurant???


# mit dynamischer zUWEISUNG MACHENß

# Bundling.
# Bundling funktioniert nicht - Logik überlegen, wie man in route_processor.py
# integrieren kann und wo, auf jeden Fall als separate Methode (route_processor.py, auch jetzt schon zu lang und groß)
# Test Bundling und dann RMDPSolver step by step,

# Weitere KPIs
# KPI: idle time
# KPI: driver workload
# Total numbers of orders the system can handle? -> average orders delivered per timestep * simulation time


# Further, sending the
# closest driver may limit the opportunities to assign orders to drivers that are already en route to a
# nearby restaurant.
# -> also muss es doch möglich sein, dass auch Vehicles die schon unterwegs sind eine Order zugeteilt kriegen.


# If the following conditions
# hold, a decision is feasible:
# 1. The arrival time of the first entry of each route
# θ ∈ Θk remains the same.
# 2. The value VD stays unaltered for all orders D
# with VD > 0 (assignments are permanent).
# 3. The sequencing in Θxk
# reflects the driver’s routing
# routine.

# expand bundling to also include how close customers are to each other??? Also als generelle Erweiterung


# ggf. umprogrammieren, dass vehicle standby gegeben werden kann, von Algorithmus
# vehicles are skipped that already have an order 54 route_processor.py
# programmieren, dass vehiclees immer wieder richtung mitte gehen, wenn idle.

# wird hier immer richtig inserted?
