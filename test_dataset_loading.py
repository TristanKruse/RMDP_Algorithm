#!/usr/bin/env python3
"""
Test script to verify individual Meituan dataset loading and processing.
This script tests data loading from a single district/day combination.
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


def test_single_dataset(district_id=1, day="20221017"):
    """
    Test loading and running a single Meituan dataset.

    Args:
        district_id: District to test (1-22)
        day: Day to test (format: 20221017-20221024)
    """
    logger.info(f"Testing dataset: District {district_id}, Day {day}")

    try:
        # Step 1: Test MeituanDataConfig creation
        logger.info("Step 1: Creating MeituanDataConfig with vehicle scaling...")
        meituan_config = MeituanDataConfig(
            district_id=district_id,
            day=day,
            use_restaurant_positions=True,  # Enable real restaurant positions
            use_vehicle_count=True,  # Enable real vehicle counts
            use_vehicle_positions=True,  # Enable real vehicle positions
            use_service_area=True,  # Enable real geographic boundaries
            use_deadlines=True,  # Enable real order deadlines
            order_generation_mode="replay",  # Use real order data
            temporal_pattern=None,  # Not needed for replay mode
            simulation_start_hour=10,  # Start at 10 AM
            simulation_duration_hours=12,  # Run for 12 hours
            # NEW: Vehicle scaling parameters
            scale_vehicles_to_restaurants=True,  # Enable scaling
            vehicles_per_restaurant_ratio=0.54,  # 0.54 couriers per restaurant
        )
        logger.info("✓ MeituanDataConfig created successfully")

        # Step 2: Test data loading and extraction
        logger.info("Step 2: Testing data extraction...")
        try:
            num_restaurants = meituan_config.get_restaurant_count()
            original_vehicles = meituan_config.get_vehicle_count()
            scaled_vehicles = meituan_config.get_scaled_vehicle_count()
            service_area = meituan_config.get_service_area_dimensions()

            logger.info(f"✓ Dataset characteristics:")
            logger.info(f"  - Restaurants: {num_restaurants}")
            logger.info(f"  - Original vehicles: {original_vehicles}")
            logger.info(
                f"  - Scaled vehicles: {scaled_vehicles} (ratio: {scaled_vehicles/num_restaurants:.2f} per restaurant)"
            )
            logger.info(f"  - Service area: {service_area}")
        except Exception as e:
            logger.warning(f"⚠ Could not extract dataset characteristics: {e}")

        # Step 3: Test running a single episode with ACA (buffer 17)
        logger.info("Step 3: Testing single episode with ACA (buffer 17)...")
        episode_stats = run_test_episode(
            solver_name="aca",
            meituan_config=meituan_config,
            seed=42,
            reposition_idle_vehicles=False,
            visualize=False,
            warmup_duration=0,
            aca_buffer=17,
            exploration_rate=0,
            save_results_to_disk=False,  # Don't save during testing
            training_mode=False,  # 🔧 CRITICAL: Disable training mode for evaluation
        )

        # Import and calculate proper metrics
        from training.utils.metrics import calculate_all_metrics
        from training.core.stats import calculate_capacity_metrics

        # Calculate capacity metrics first (this adds derived metrics to episode_stats)
        episode_stats = calculate_capacity_metrics(episode_stats, 600, 60, 0)  # 10hr sim, 1hr cooldown, 0 warmup

        # Calculate all metrics from episode stats
        metrics = calculate_all_metrics(episode_stats)

        logger.info("✓ Episode completed successfully")
        logger.info(f"Episode results (raw stats):")
        logger.info(f"  - Total orders: {episode_stats.get('total_orders', 'N/A')}")
        logger.info(f"  - Orders delivered: {episode_stats.get('orders_delivered', 'N/A')}")
        logger.info(f"  - Raw delay values count: {len(episode_stats.get('delay_values', []))}")
        logger.info(f"  - Total delay (sum): {sum(episode_stats.get('delay_values', [])):.1f}")
        logger.info(f"  - Max delay: {episode_stats.get('max_delay', 'N/A')}")

        logger.info(f"Episode results (calculated metrics):")
        logger.info(f"  - Mean delay: {metrics.get('mean_delay', 'N/A'):.2f}")
        logger.info(f"  - Total orders: {metrics.get('total_orders', 'N/A')}")
        logger.info(f"  - Orders delivered: {metrics.get('orders_delivered', 'N/A')}")

        # Create a combined stats dict with both raw and calculated metrics
        stats = {**episode_stats, **metrics}

        return True, stats

    except FileNotFoundError as e:
        logger.error(f"✗ Data files not found: {e}")
        logger.info("Check if the Meituan dataset files exist in the expected location:")
        logger.info(f"  Expected path: data/meituan_data/processed/daily_orders/{day}/")
        return False, None

    except Exception as e:
        logger.error(f"✗ Error during testing: {e}")
        logger.exception("Full error traceback:")
        return False, None


def test_multiple_configurations(district_id=1, day="20221017"):
    """
    Test different solver configurations on the same dataset.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING MULTIPLE SOLVER CONFIGURATIONS")
    logger.info(f"Dataset: District {district_id}, Day {day}")
    logger.info(f"{'='*60}")

    # Common MeituanDataConfig
    try:
        meituan_config = MeituanDataConfig(
            district_id=district_id,
            day=day,
            use_restaurant_positions=True,
            use_vehicle_count=True,
            use_vehicle_positions=True,
            use_service_area=True,
            use_deadlines=True,
            order_generation_mode="replay",
            simulation_start_hour=10,
            simulation_duration_hours=12,
            # Enable vehicle scaling
            scale_vehicles_to_restaurants=True,
            vehicles_per_restaurant_ratio=0.54,
        )
    except Exception as e:
        logger.error(f"Failed to create MeituanDataConfig: {e}")
        return

    # Test different solver configurations
    test_configs = [
        {
            "name": "ACA Buffer 17",
            "solver_name": "aca",
            "aca_buffer": 17,
        },
        {
            "name": "ACA Max Buffer",
            "solver_name": "aca",
            "aca_buffer": 999,  # Max buffer for "fastest" behavior
        },
        {
            "name": "RL-ACA",
            "solver_name": "rl_aca",
            "aca_buffer": 17,
        },
    ]

    results = {}

    for config in test_configs:
        logger.info(f"\nTesting: {config['name']}")
        logger.info("-" * 40)

        try:
            run_params = {
                "solver_name": config["solver_name"],
                "meituan_config": meituan_config,
                "seed": 42,
                "reposition_idle_vehicles": False,
                "visualize": False,
                "warmup_duration": 0,
                "exploration_rate": 0,
                "save_results_to_disk": False,
            }

            # Add solver-specific parameters
            if "aca_buffer" in config:
                run_params["aca_buffer"] = config["aca_buffer"]

            stats = run_test_episode(**run_params)

            results[config["name"]] = stats
            logger.info(f"✓ {config['name']} completed successfully")
            logger.info(f"  Total delay: {stats.get('total_delay', 'N/A'):.1f}")
            logger.info(f"  On-time rate: {stats.get('on_time_delivery_rate', 'N/A'):.1f}%")

        except Exception as e:
            logger.error(f"✗ {config['name']} failed: {e}")
            results[config["name"]] = None

    # Summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Method':<20} {'Total Delay':<12} {'On-time Rate':<12}")
    logger.info("-" * 60)

    for method_name, stats in results.items():
        if stats and isinstance(stats, dict):
            total_delay = stats.get("total_delay", 0)
            on_time_rate = stats.get("on_time_rate", 0)
            logger.info(f"{method_name:<20} {total_delay:<12.1f} {on_time_rate:<12.1f}%")
        else:
            logger.info(f"{method_name:<20} {'FAILED':<12} {'FAILED':<12}")

    return results


def test_data_availability():
    """
    Check which datasets are available.
    """
    logger.info(f"\n{'='*60}")
    logger.info("CHECKING DATA AVAILABILITY")
    logger.info(f"{'='*60}")

    base_path = "data/meituan_data/processed/daily_orders"

    if not os.path.exists(base_path):
        logger.error(f"✗ Base data directory not found: {base_path}")
        return

    districts = list(range(1, 23))  # Districts 1-22
    days = [f"202210{day:02d}" for day in range(17, 25)]  # Oct 17-24

    available_count = 0
    total_count = len(districts) * len(days)

    logger.info("Checking dataset availability...")

    for day in days:
        day_path = os.path.join(base_path, day)
        if os.path.exists(day_path):
            logger.info(f"Day {day}: Available")
            for district in districts:
                files_needed = [
                    f"district_{district}_restaurants.csv",
                    f"district_{district}_vehicles.csv",
                    f"district_{district}_orders.csv",
                ]

                all_files_exist = all(os.path.exists(os.path.join(day_path, filename)) for filename in files_needed)

                if all_files_exist:
                    available_count += 1

        else:
            logger.warning(f"Day {day}: Missing")

    logger.info(f"\nSummary: {available_count}/{total_count} datasets available")

    if available_count > 0:
        # Find first available dataset for testing
        for day in days:
            day_path = os.path.join(base_path, day)
            if os.path.exists(day_path):
                for district in districts:
                    files_needed = [
                        f"district_{district}_restaurants.csv",
                        f"district_{district}_vehicles.csv",
                        f"district_{district}_orders.csv",
                    ]

                    all_files_exist = all(os.path.exists(os.path.join(day_path, filename)) for filename in files_needed)

                    if all_files_exist:
                        logger.info(f"✓ Found available dataset: District {district}, Day {day}")
                        return district, day

    return None, None


if __name__ == "__main__":
    logger.info("Starting Meituan dataset testing...")

    # Step 1: Check data availability
    district, day = test_data_availability()

    if district and day:
        # Step 2: Test single dataset loading
        logger.info(f"\n{'='*60}")
        logger.info("TESTING SINGLE DATASET")
        logger.info(f"{'='*60}")

        success, stats = test_single_dataset(district, day)

        if success:
            # Step 3: Test multiple configurations
            test_multiple_configurations(district, day)
        else:
            logger.error("Single dataset test failed. Cannot proceed with multiple configurations.")
    else:
        logger.error("No datasets available for testing. Please check your data directory structure.")
        logger.info("\nExpected structure:")
        logger.info("data/meituan_data/processed/daily_orders/")
        logger.info("├── 20221017/")
        logger.info("│   ├── district_1_restaurants.csv")
        logger.info("│   ├── district_1_vehicles.csv")
        logger.info("│   ├── district_1_orders.csv")
        logger.info("│   └── ...")
        logger.info("└── ...")

    logger.info("\nTesting completed!")
