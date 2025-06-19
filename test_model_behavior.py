import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def patch_rl_model_for_testing():
    """Patch the RL model to always return NO_POSTPONE for testing."""

    try:
        from models.aca_policy import rl_postponement

        # Save the original method
        original_evaluate = rl_postponement.RLPostponementDecision.evaluate_postponement

        def forced_no_postpone(self, postponed, route_plan, order_id, current_time, state, exploration_rate=None):
            """Override to always return False (NO_POSTPONE)."""
            return False  # Always NO_POSTPONE

        # Patch the method
        rl_postponement.RLPostponementDecision.evaluate_postponement = forced_no_postpone

        print("✅ Successfully patched RL model to always return NO_POSTPONE")
        return True

    except Exception as e:
        print(f"❌ Failed to patch RL model: {e}")
        return False


def run_single_simulation_test():
    """Run a single simulation with the patched RL model."""

    try:
        from training.train import run_test_episode, MeituanDataConfig

        print("🚀 Running single simulation test with forced NO_POSTPONE...")

        # Use a simple district/day combination
        meituan_config = MeituanDataConfig(
            district_id=3,  # Use a working district
            day="20221017",
            use_restaurant_positions=True,
            use_vehicle_count=True,
            use_vehicle_positions=True,
            use_service_area=True,
            use_deadlines=True,
            order_generation_mode="replay",
            temporal_pattern=None,
            simulation_start_hour=10,
            simulation_duration_hours=12,
        )

        # Run simulation
        stats = run_test_episode(
            solver_name="rl_aca",
            meituan_config=meituan_config,
            seed=12345,
            reposition_idle_vehicles=False,
            visualize=False,
            warmup_duration=0,
            save_results_to_disk=False,
            aca_buffer=17,
            exploration_rate=0,  # No exploration
        )

        print("✅ Simulation completed!")
        print(f"📊 Results with FORCED NO_POSTPONE:")
        print(f"   On-time delivery rate: {stats.get('on_time_delivery_rate', 'N/A'):.1f}%")
        print(f"   Total delay: {stats.get('total_delay', 'N/A'):.1f} minutes")
        print(f"   Total orders: {stats.get('total_orders', 'N/A')}")
        print(f"   Orders delivered: {stats.get('orders_delivered', 'N/A')}")
        print(f"   Undelivered orders: {stats.get('total_orders', 0) - stats.get('orders_delivered', 0)}")
        print(f"   Max delay: {stats.get('max_delay', 'N/A'):.1f} minutes")

        # Compare with expected performance
        on_time_rate = stats.get("on_time_delivery_rate", 0)
        if on_time_rate > 50:
            print("🎉 SUCCESS: Performance dramatically improved!")
            print("✅ CONFIRMED: Problem was over-postponement")
        elif on_time_rate > 10:
            print("🟡 PARTIAL: Some improvement, but still issues")
            print("⚠️  May be postponement + other issues")
        else:
            print("❌ FAILED: Still poor performance")
            print("🔍 Problem is not just postponement")

        return stats

    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main test function."""
    print("🧪 TESTING HYPOTHESIS: RL Model Over-Postponement")
    print("=" * 60)
    print("❓ Current problem: RL model postpones ~67% of orders")
    print("❓ Hypothesis: This causes orders to pile up and never get delivered")
    print("🔧 Test: Force RL to NEVER postpone and see if performance improves")
    print("")

    # Step 1: Patch the RL model
    if not patch_rl_model_for_testing():
        print("❌ Cannot proceed - patching failed")
        return

    # Step 2: Run simulation test
    stats = run_single_simulation_test()

    if stats:
        print("\n" + "=" * 60)
        print("🎯 TEST CONCLUSIONS:")
        on_time_rate = stats.get("on_time_delivery_rate", 0)

        if on_time_rate > 50:
            print("✅ HYPOTHESIS CONFIRMED: Over-postponement was the main issue")
            print("📊 Expected performance with no postponement: 60-80%")
            print("🛠️  SOLUTION: Retrain RL model with better reward structure")
        elif on_time_rate > 10:
            print("🟡 HYPOTHESIS PARTIALLY CONFIRMED: Over-postponement is a major issue")
            print("📊 Some improvement, but other issues exist too")
            print("🛠️  SOLUTION: Fix postponement + investigate other bugs")
        else:
            print("❌ HYPOTHESIS REJECTED: Over-postponement is not the main issue")
            print("📊 Even with no postponement, performance is still terrible")
            print("🔍 Need deeper investigation of RL integration")


if __name__ == "__main__":
    main()
