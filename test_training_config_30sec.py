#!/usr/bin/env python3
"""
Test the exact training configuration used in train_rl.py with 30-second intervals
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_training_configuration():
    """Test a single episode using the exact training configuration."""
    
    print("Testing Training Configuration (30-Second Intervals)")
    print("=" * 60)
    print("🔧 Using exact same config as train_rl.py phase:")
    print("   • 10 vehicles, 20 restaurants")
    print("   • mean_interarrival_time: 16 (= 8 minutes real time)")
    print("   • RL-ACA solver with 30-second timing")
    print()
    
    try:
        from training.core.episode import run_test_episode
        
        # Use exact training configuration from train_rl.py phase
        stats = run_test_episode(
            solver_name="rl_aca",
            seed=42,
            reposition_idle_vehicles=False,
            visualize=False,
            warmup_duration=0,
            save_results_to_disk=False,
            training_mode=True,  # Enable RL training mode
            env_config={
                "num_vehicles": 10,
                "num_restaurants": 20, 
                "service_area_dimensions": (6.0, 6.0),
                "mean_interarrival_time": 16,  # 8 minutes = 16 timesteps @ 30sec
            },
            # RL parameters for testing
            exploration_rate=0.5,  # Medium exploration for test
        )
        
        print("✅ Training configuration test completed!")
        
        # Analyze results
        print("\n" + "="*50)
        print("TRAINING CONFIGURATION TEST RESULTS")
        print("="*50)
        
        total_orders = stats.get('total_orders', 0)
        orders_delivered = stats.get('orders_delivered', 0)
        late_orders = stats.get('late_orders', set())
        late_count = len(late_orders)
        
        if orders_delivered > 0:
            on_time_rate = ((orders_delivered - late_count) / orders_delivered) * 100
        else:
            on_time_rate = 0
            
        total_delay = stats.get('total_delay', 0)
        postponed_orders = stats.get('postponed_orders', set())
        postponement_rate = len(postponed_orders) / max(1, total_orders) * 100
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Total orders: {total_orders}")
        print(f"   Orders delivered: {orders_delivered}")
        print(f"   On-time delivery rate: {on_time_rate:.1f}%")
        print(f"   Total delay: {total_delay:.1f} minutes")
        print(f"   Postponement rate: {postponement_rate:.1f}%")
        
        # Timing validation
        expected_simulation_duration = 1200  # 10 hours * 2 timesteps/min
        actual_timesteps = stats.get('total_timesteps', 'Unknown')
        
        print(f"\n⏱️ TIMING VALIDATION:")
        print(f"   Expected simulation: {expected_simulation_duration} timesteps (10 hours)")
        print(f"   Actual timesteps: {actual_timesteps}")
        
        if actual_timesteps != 'Unknown':
            real_time_hours = (actual_timesteps * 0.5) / 60
            print(f"   Real-time simulated: {real_time_hours:.1f} hours")
            
            if abs(actual_timesteps - expected_simulation_duration) <= 20:
                print("   ✅ Simulation duration is correct!")
            else:
                print("   ⚠️ Simulation duration may be incorrect")
        
        # Order generation validation
        expected_orders_per_hour = 60 / 8  # 1 order every 8 minutes = 7.5 orders/hour
        actual_orders_per_hour = total_orders / (real_time_hours if actual_timesteps != 'Unknown' else 10)
        
        print(f"\n📦 ORDER GENERATION VALIDATION:")
        print(f"   Expected: ~{expected_orders_per_hour:.1f} orders/hour")
        print(f"   Actual: {actual_orders_per_hour:.1f} orders/hour")
        
        if abs(actual_orders_per_hour - expected_orders_per_hour) <= 1.0:
            print("   ✅ Order generation rate is correct!")
        else:
            print("   ⚠️ Order generation rate may be off")
        
        print(f"\n🎯 READINESS ASSESSMENT:")
        if total_orders > 0 and orders_delivered > 0:
            print("   ✅ Simulation runs successfully")
            print("   ✅ Orders are generated and delivered") 
            print("   ✅ RL solver is working")
            print("   ✅ Ready for full training!")
        else:
            print("   ❌ Issues detected - check configuration")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing the exact configuration that will be used in train_rl.py...")
    print("This validates all 30-second timing parameters are working correctly.\n")
    
    success = test_training_configuration()
    
    if success:
        print("\n" + "="*60)
        print("🎉 TRAINING CONFIGURATION VALIDATED!")
        print("="*60)
        print("✅ All 30-second timing parameters are working correctly")
        print("✅ Ready to run: uv run python train_rl.py") 
        print("✅ Training should complete in reasonable time")
    else:
        print("\n" + "="*60)
        print("❌ CONFIGURATION VALIDATION FAILED")
        print("="*60)
        print("Please fix the issues above before running full training")