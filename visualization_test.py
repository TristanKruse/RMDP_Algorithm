#!/usr/bin/env python3
"""
Simple script to test visualization with real Meituan data.
This will show you the real data in action with visual output.
"""

import os
import logging
from datetime import datetime
from training.train import run_test_episode
from environment.meituan_data.meituan_data_config import MeituanDataConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_simple_visualization():
    """
    Run a simple visualization test with real Meituan data.
    """
    logger.info("🎬 Starting Simple Visualization Test")
    logger.info("=" * 60)

    # Test parameters
    district_id = 1
    day = "20221017"

    logger.info(f"📍 Location: District {district_id}, Day {day}")
    logger.info(f"🎯 Goal: Visualize real Meituan data in action")

    try:
        # Create MeituanDataConfig with real data and vehicle scaling
        logger.info("📊 Loading real Meituan data with vehicle scaling...")
        meituan_config = MeituanDataConfig(
            district_id=district_id,
            day=day,
            # Enable ALL real data features
            use_restaurant_positions=True,  # ✅ Real restaurant locations
            use_vehicle_count=True,  # ✅ Real vehicle counts
            use_vehicle_positions=True,  # ✅ Real vehicle starting positions
            use_service_area=True,  # ✅ Real geographic boundaries
            use_deadlines=True,  # ✅ Real order deadlines
            # Use real order data and demand patterns
            order_generation_mode="replay",  # ✅ Real historical orders & demand
            temporal_pattern=None,  # Not needed for replay mode
            simulation_start_hour=10,  # Start at 10 AM
            simulation_duration_hours=12,  # Run for 12 hours
            # Vehicle scaling
            scale_vehicles_to_restaurants=True,  # Enable scaling
            vehicles_per_restaurant_ratio=0.54,  # 0.54 couriers per restaurant
        )

        # Display dataset characteristics
        logger.info("✅ Real data loaded successfully!")
        logger.info(f"🏪 Restaurants: {meituan_config.get_restaurant_count()}")
        logger.info(f"🚗 Original vehicles: {meituan_config.get_vehicle_count()}")
        logger.info(f"🚗 Scaled vehicles: {meituan_config.get_scaled_vehicle_count()} (0.54 per restaurant)")
        logger.info(f"📏 Service area: {meituan_config.get_service_area_dimensions()}")

        logger.info("\n🎬 Starting simulation with VISUALIZATION enabled...")
        logger.info("💡 This will show you:")
        logger.info("   - Real restaurant locations plotted on map")
        logger.info("   - Real vehicle movements in real-time")
        logger.info("   - Real order pickups and deliveries")
        logger.info("   - Live simulation progress")
        logger.info("\n⏳ Please wait for simulation to complete...")

        # Create necessary directories
        os.makedirs("data/results", exist_ok=True)
        os.makedirs("data/visualizations", exist_ok=True)

        # Run simulation with ACA Buffer 17 and visualization
        logger.info("🤖 Running ACA (Buffer 17) with visualization...")
        episode_stats = run_test_episode(
            solver_name="aca",
            meituan_config=meituan_config,
            seed=42,
            reposition_idle_vehicles=False,
            visualize=True,  # 🎥 ENABLE VISUALIZATION
            warmup_duration=0,
            aca_buffer=17,
            exploration_rate=0,
            save_results_to_disk=True,  # Save results and visualizations
            training_mode=False,  # Evaluation mode only
        )

        # Extract and display key results
        total_delay = sum(episode_stats.get("delay_values", []))
        total_orders = episode_stats.get("total_orders", 0)
        orders_delivered = episode_stats.get("orders_delivered", 0)
        max_delay = episode_stats.get("max_delay", 0)
        late_orders_count = len(episode_stats.get("late_orders", set()))
        on_time_orders = orders_delivered - late_orders_count
        on_time_rate = (on_time_orders / total_orders * 100) if total_orders > 0 else 0

        # Display comprehensive results
        logger.info(f"\n🎯 SIMULATION RESULTS")
        logger.info("=" * 60)

        logger.info(f"📦 Orders:")
        logger.info(f"   Total orders processed: {total_orders}")
        logger.info(f"   Orders delivered: {orders_delivered}")
        logger.info(f"   Undelivered orders: {total_orders - orders_delivered}")
        logger.info(f"   Late orders: {late_orders_count}")

        logger.info(f"⏱️ Performance:")
        logger.info(f"   Total delay: {total_delay:.1f} minutes")
        logger.info(
            f"   Average delay: {total_delay/orders_delivered:.1f} minutes per order"
            if orders_delivered > 0
            else "   Average delay: N/A"
        )
        logger.info(f"   Maximum delay: {max_delay:.1f} minutes")
        logger.info(f"   On-time delivery rate: {on_time_rate:.1f}%")

        logger.info(f"🚛 Fleet:")
        logger.info(f"   Restaurants: {meituan_config.get_restaurant_count()}")
        logger.info(f"   Vehicles (scaled): {meituan_config.get_scaled_vehicle_count()}")
        logger.info(f"   Vehicle-to-restaurant ratio: 0.54")

        # Show what real data features were used
        logger.info(f"\n✅ REAL DATA FEATURES CONFIRMED")
        logger.info("=" * 60)
        logger.info("🏪 Real restaurant positions from Meituan dataset")
        logger.info("🚗 Real vehicle counts with 0.54 scaling ratio")
        logger.info("📋 Real historical order data with actual timestamps")
        logger.info("🌍 Real geographic service area boundaries")
        logger.info("⏰ Real demand patterns from historical order data")
        logger.info("📅 Real delivery deadlines from order data")

        logger.info(f"\n🎬 Outputs saved to:")
        logger.info(f"📈 Visualizations: data/visualizations/")
        logger.info(f"📊 Results: data/results/")

        # Performance summary
        logger.info(f"\n📋 PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Algorithm: ACA with Buffer 17")
        logger.info(f"Dataset: District {district_id}, Day {day}")
        logger.info(f"Total Delay: {total_delay:.1f} minutes")
        logger.info(f"On-time Rate: {on_time_rate:.1f}%")
        logger.info(f"Delivery Rate: {orders_delivered}/{total_orders} ({(orders_delivered/total_orders*100):.1f}%)")

        return True

    except FileNotFoundError as e:
        logger.error(f"❌ Dataset not found: {e}")
        logger.info("💡 Make sure the Meituan data files exist in:")
        logger.info(f"   data/meituan_data/processed/daily_orders/{day}/")
        return False

    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        logger.exception("Full error:")
        return False


def compare_methods_quick():
    """
    Quick comparison of methods without full visualization.
    """
    logger.info(f"\n🔬 QUICK METHOD COMPARISON")
    logger.info("=" * 60)

    methods = [
        {"name": "ACA Buffer 17", "solver": "aca", "buffer": 17},
        {"name": "ACA Max Buffer", "solver": "aca", "buffer": 999},
    ]

    results = {}

    for method in methods:
        logger.info(f"\n🧪 Testing: {method['name']}")
        logger.info("-" * 40)

        try:
            # Create config
            meituan_config = MeituanDataConfig(
                district_id=1,
                day="20221017",
                use_restaurant_positions=True,
                use_vehicle_count=True,
                use_vehicle_positions=True,
                use_service_area=True,
                use_deadlines=True,
                order_generation_mode="replay",
                simulation_start_hour=10,
                simulation_duration_hours=12,
                scale_vehicles_to_restaurants=True,
                vehicles_per_restaurant_ratio=0.54,
            )

            # Run simulation (no visualization for speed)
            episode_stats = run_test_episode(
                solver_name=method["solver"],
                meituan_config=meituan_config,
                seed=42,
                reposition_idle_vehicles=False,
                visualize=False,  # No visualization for quick comparison
                warmup_duration=0,
                aca_buffer=method["buffer"],
                exploration_rate=0,
                save_results_to_disk=False,
                training_mode=False,
            )

            # Calculate metrics
            total_delay = sum(episode_stats.get("delay_values", []))
            total_orders = episode_stats.get("total_orders", 0)
            orders_delivered = episode_stats.get("orders_delivered", 0)
            late_orders = len(episode_stats.get("late_orders", set()))
            on_time_rate = ((orders_delivered - late_orders) / total_orders * 100) if total_orders > 0 else 0

            results[method["name"]] = {
                "total_delay": total_delay,
                "on_time_rate": on_time_rate,
                "orders_delivered": orders_delivered,
                "total_orders": total_orders,
            }

            logger.info(f"✅ {method['name']} completed")
            logger.info(f"  Total delay: {total_delay:.1f} minutes")
            logger.info(f"  On-time rate: {on_time_rate:.1f}%")
            logger.info(f"  Orders: {orders_delivered}/{total_orders}")

        except Exception as e:
            logger.error(f"❌ {method['name']} failed: {e}")
            results[method["name"]] = None

    # Display comparison
    logger.info(f"\n📊 COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Method':<20} {'Total Delay':<12} {'On-time Rate':<12} {'Orders':<10}")
    logger.info("-" * 60)

    for method_name, result in results.items():
        if result:
            logger.info(
                f"{method_name:<20} {result['total_delay']:<12.1f} {result['on_time_rate']:<12.1f}% {result['orders_delivered']}/{result['total_orders']}"
            )
        else:
            logger.info(f"{method_name:<20} {'FAILED':<12} {'FAILED':<12} {'FAILED':<10}")

    return results


if __name__ == "__main__":
    logger.info("🚀 Starting Real Data Visualization Test")
    logger.info("This will run simulations using REAL Meituan data with scaled vehicles")

    # Test 1: Run single simulation with visualization
    logger.info("\n1️⃣ Running visualization test...")
    success = run_simple_visualization()

    if success:
        logger.info("✅ Visualization test completed successfully!")

        # Test 2: Quick method comparison (optional)
        user_input = input("\n🤔 Would you like to run a quick method comparison? (y/n): ").lower().strip()
        if user_input == "y":
            logger.info("\n2️⃣ Running quick method comparison...")
            comparison_results = compare_methods_quick()
            logger.info("✅ Method comparison completed!")
        else:
            logger.info("👍 Skipping method comparison")
    else:
        logger.error("❌ Visualization test failed")

    logger.info(f"\n🎉 Test completed!")
    logger.info("📁 Check the following directories for outputs:")
    logger.info("   📊 Results: data/results/")
    logger.info("   📈 Visualizations: data/visualizations/")
    logger.info("\n💡 If visualization worked, you can now run the full benchmarking!")
