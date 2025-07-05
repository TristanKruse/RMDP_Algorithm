#!/usr/bin/env python3
"""
Quick check to see what KPIs are actually calculated and available in the stats.
"""

from training.train import run_test_episode
from environment.meituan_data.meituan_data_config import MeituanDataConfig
from pathlib import Path
import numpy as np


def check_kpi_extraction():
    print("🔍 CHECKING KPI EXTRACTION")
    print("=" * 40)

    # Test on District 1
    config = MeituanDataConfig(
        district_id=3,
        day="20221017",
        use_restaurant_positions=True,
        use_vehicle_count=True,
        order_generation_mode="replay",
    )

    model_path = str(Path("data/models/rl_aca_phase4_final.pt"))

    print(f"Running test episode...")
    stats = run_test_episode(
        solver_name="rl_aca",
        meituan_config=config,
        seed=1,
        exploration_rate=0,
        visualize=False,
        training_mode=False,
        rl_model_path=model_path,
        save_results_to_disk=False,
    )

    # Check what the benchmarking expects vs what's available
    expected_kpis = [
        "on_time_delivery_rate",
        "active_period_idle_rate",
        "avg_delay_late_orders",
        "max_delay",
        "avg_distance_per_order",
        "total_delay",
        "postponement_rate",
    ]

    print(f"\n📊 KPI AVAILABILITY CHECK:")
    print(f"   Total stats keys: {len(stats.keys())}")

    for kpi in expected_kpis:
        value = stats.get(kpi, "MISSING")
        status = "✅" if kpi in stats else "❌"
        print(f"   {status} {kpi}: {value}")

    # Calculate the missing KPIs manually from available data
    total_orders = stats.get("total_orders", 0)
    orders_delivered = stats.get("orders_delivered", 0)

    print(f"\n📈 MANUAL CALCULATIONS FROM AVAILABLE DATA:")
    print(f"   Total orders: {total_orders}")
    print(f"   Orders delivered: {orders_delivered}")
    print(
        f"   Delivery rate: {(orders_delivered / total_orders * 100):.1f}%"
        if total_orders > 0
        else "   Delivery rate: N/A"
    )

    # Calculate KPIs using EXACT same logic as train_rl.py compare_models function

    late_orders = stats.get("late_orders", set())
    delay_values = stats.get("delay_values", [])
    total_distance = stats.get("total_distance", 0)
    postponed_orders = stats.get("postponed_orders", set())

    # 1. total_delay: same as compare_models
    total_delay = sum(delay_values) if delay_values else 0
    print(f"   ✅ total_delay (calculated): {total_delay:.1f} min (from delay_values)")

    # 2. on_time_rate: EXACT same logic as compare_models
    total_orders_calc = max(1, orders_delivered)  # compare_models uses orders_delivered
    late_count = len(late_orders)
    on_time_rate = ((total_orders_calc - late_count) / total_orders_calc) * 100
    print(f"   Late orders: {late_count}")
    print(f"   Orders delivered: {orders_delivered}")
    print(f"   On-time orders: {total_orders_calc - late_count}")
    print(f"   ✅ on_time_delivery_rate: {on_time_rate:.1f}%")

    # 3. avg_delay_late_orders: average of delay_values (same as compare_models)
    avg_delay_late = sum(delay_values) / len(delay_values) if delay_values else 0
    print(f"   ✅ avg_delay_late_orders: {avg_delay_late:.1f} minutes")

    # 4. postponement_rate: EXACT same logic as compare_models
    postponement_rate = len(postponed_orders) / max(1, total_orders) * 100
    print(f"   ✅ postponement_rate: {postponement_rate:.1f}% ({len(postponed_orders)}/{total_orders} postponed)")

    # 5. avg_distance_per_order: Use idle-rate method for realistic estimates
    from training.core.stats import calculate_idle_rate_distance
    simulation_duration = 720  # 12 hours in minutes
    total_productive_distance = calculate_idle_rate_distance(stats, simulation_duration)
    idle_rate_distance_per_order = total_productive_distance / max(1, orders_delivered)
    
    # Also show the old method for comparison
    old_method_distance_per_order = total_distance / max(1, orders_delivered)
    print(f"   ✅ avg_distance_per_order (NEW idle-rate method): {idle_rate_distance_per_order:.1f} km")
    print(f"   📊 avg_distance_per_order (OLD buggy method): {old_method_distance_per_order:.1f} km")
    print(f"      Expected realistic range: 2-5 km per order")
    
    # Get idle rates data for alternative calculation
    idle_rates = stats.get("active_period_idle_rates_by_vehicle", {})
    
    # Alternative calculation using idle rates (user's suggestion)
    print(f"\n🚗 ALTERNATIVE DISTANCE CALCULATION (Idle-Rate Based):")
    if idle_rates:
        # Calculate average vehicle utilization across all vehicles
        vehicle_utilizations = [1 - np.mean(rates) for rates in idle_rates.values() if rates]
        avg_utilization = np.mean(vehicle_utilizations) if vehicle_utilizations else 0
        
        # Assume 12-hour simulation (720 minutes) and 16 km/h speed
        simulation_time_minutes = 720  # 12 hours
        vehicle_speed_km_per_minute = 8 / 60  # 8 km/h = 0.133 km/min (30-sec intervals)
        num_vehicles = len(idle_rates)
        
        # Total productive distance = utilization × time × vehicles × speed
        total_productive_distance = avg_utilization * simulation_time_minutes * num_vehicles * vehicle_speed_km_per_minute
        productive_distance_per_order = total_productive_distance / max(1, orders_delivered)
        
        print(f"   📊 Calculation details:")
        print(f"      Average utilization: {avg_utilization:.1%}")
        print(f"      Simulation time: {simulation_time_minutes} minutes")
        print(f"      Vehicle speed: {vehicle_speed_km_per_minute:.3f} km/min")
        print(f"      Number of vehicles: {num_vehicles}")
        print(f"   ✅ Total productive distance: {total_productive_distance:.1f} km")
        print(f"   ✅ Distance per order (idle-based): {productive_distance_per_order:.1f} km")
        print(f"   📈 Comparison: Current method {old_method_distance_per_order:.1f} km vs Idle-based {productive_distance_per_order:.1f} km")
    else:
        print(f"   ❌ Cannot calculate - no idle rate data available")

    # 6. vehicle_utilization: same as compare_models
    if idle_rates:
        vehicle_utilizations = [1 - np.mean(rates) for rates in idle_rates.values() if rates]
        active_period_idle_rate = np.mean(vehicle_utilizations) if vehicle_utilizations else 0
        print(f"   ✅ active_period_idle_rate (calculated): {active_period_idle_rate:.3f}")
        print(f"      (from {len(idle_rates)} vehicles)")
    else:
        print(f"   ⚠️  active_period_idle_rates_by_vehicle: MISSING")

    # Check the correct key names based on train_rl.py compare_models
    stats_total_delays = stats.get("total_delays", "MISSING")  # Plural form
    stats_total_delay = stats.get("total_delay", "MISSING")  # Singular form
    print(f"   Stats total_delays (plural): {stats_total_delays}")
    print(f"   Stats total_delay (singular): {stats_total_delay}")
    print(f"   (logs showed 8776.6 - checking which key matches!)")

    # Calculate vehicle utilization same as train_rl.py compare_models
    if idle_rates:
        vehicle_utilizations = [1 - np.mean(rates) for rates in idle_rates.values() if rates]
        avg_vehicle_utilization = np.mean(vehicle_utilizations) if vehicle_utilizations else 0
        print(f"   ✅ active_period_idle_rate (calculated): {avg_vehicle_utilization:.3f}")
        print(f"      (from {len(idle_rates)} vehicles)")
    else:
        print(f"   ⚠️  active_period_idle_rates_by_vehicle: MISSING")

    print(f"\n🔍 KEY INSIGHTS:")
    print(f"   • late_orders type: {type(late_orders)}, count: {len(late_orders)}")
    print(f"   • postponed_orders type: {type(postponed_orders)}, count: {len(postponed_orders)}")
    print(
        f"   • delay_values type: {type(delay_values)}, length: {len(delay_values) if hasattr(delay_values, '__len__') else 'N/A'}"
    )
    print(f"   • total_distance: {total_distance}")
    print(
        f"   • idle_rates type: {type(idle_rates)}, vehicles: {len(idle_rates) if hasattr(idle_rates, '__len__') else 'N/A'}"
    )
    
    print(f"\n🐛 TIME vs DISTANCE BUG ANALYSIS:")
    print(f"   The current system uses get_travel_time() as distance:")
    print(f"   • get_travel_time() formula: sqrt(dx² + dy²) / movement_per_step")
    print(f"   • movement_per_step = 16 km/h / 60 = 0.267 km/min")
    print(f"   • So get_travel_time() returns MINUTES, not kilometers")
    print(f"   • But code treats result as kilometers → {total_distance / (16/60):.1f} minutes actual travel time")
    print(f"   • Corrected distance would be: {total_distance * (16/60):.1f} km")
    print(f"   • Corrected distance per order: {(total_distance * (16/60)) / max(1, orders_delivered):.1f} km")

    print(f"\n🔍 DIAGNOSIS:")
    missing_kpis = [kpi for kpi in expected_kpis if kpi not in stats]
    if missing_kpis:
        print(f"   ❌ Missing KPIs: {missing_kpis}")
        print(f"   💡 These will show as 0 in the benchmark results!")
        print(f"   🛠️  Need to either calculate them or fix the stats generation")
    else:
        print(f"   ✅ All expected KPIs are present!")


if __name__ == "__main__":
    check_kpi_extraction()
