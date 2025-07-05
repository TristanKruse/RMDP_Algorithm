#!/usr/bin/env python3
"""
Quick analysis of RL methods performance comparison
"""

import pandas as pd
import numpy as np

def analyze_rl_performance():
    """Compare RL-ACA vs 1-Phase RL-ACA performance."""
    
    # Load data
    df = pd.read_csv("data/simulation_results/benchmark_results.csv")
    
    # Filter for just the RL methods
    rl_data = df[df["method"].isin(["rl_aca", "rl_aca_phase1_final"])].copy()
    
    # Key metrics for comparison
    metrics = [
        "on_time_delivery_rate", 
        "avg_delay_late_orders", 
        "total_delay",
        "avg_distance_per_order",
        "active_period_idle_rate",
        "postponement_rate"
    ]
    
    print("RL-ACA Performance Comparison")
    print("=" * 50)
    print("RL-ACA = rl_aca (curriculum learning)")
    print("1-Phase = rl_aca_phase1_final (hypertuned single phase)\n")
    
    for metric in metrics:
        print(f"{metric.upper().replace('_', ' ')}:")
        
        phase4_data = rl_data[rl_data["method"] == "rl_aca"][metric]
        phase1_data = rl_data[rl_data["method"] == "rl_aca_phase1_final"][metric]
        
        phase4_mean = phase4_data.mean()
        phase1_mean = phase1_data.mean()
        
        # Determine if higher or lower is better
        better_higher = metric in ["on_time_delivery_rate"]
        
        if better_higher:
            winner = "RL-ACA" if phase4_mean > phase1_mean else "1-Phase"
            improvement = abs(phase4_mean - phase1_mean)
        else:
            winner = "RL-ACA" if phase4_mean < phase1_mean else "1-Phase"  
            improvement = abs(phase4_mean - phase1_mean)
        
        print(f"  RL-ACA: {phase4_mean:.2f}")
        print(f"  1-Phase: {phase1_mean:.2f}")
        print(f"  Winner: {winner} (difference: {improvement:.2f})")
        print()
    
    # Overall summary
    print("SUMMARY:")
    print("=" * 30)
    
    # Count wins
    wins_4phase = 0
    wins_1phase = 0
    
    for metric in metrics:
        phase4_mean = rl_data[rl_data["method"] == "rl_aca"][metric].mean()
        phase1_mean = rl_data[rl_data["method"] == "rl_aca_phase1_final"][metric].mean()
        
        better_higher = metric in ["on_time_delivery_rate"]
        
        if better_higher:
            if phase4_mean > phase1_mean:
                wins_4phase += 1
            else:
                wins_1phase += 1
        else:
            if phase4_mean < phase1_mean:
                wins_4phase += 1
            else:
                wins_1phase += 1
    
    print(f"RL-ACA (Curriculum) wins: {wins_4phase}/{len(metrics)} metrics")
    print(f"1-Phase (Hypertuned) wins: {wins_1phase}/{len(metrics)} metrics")
    
    if wins_1phase > wins_4phase:
        print("\n🔍 FINDING: 1-Phase hypertuned model outperforms curriculum learning approach")
    else:
        print("\n🔍 FINDING: RL-ACA curriculum learning approach is superior")

if __name__ == "__main__":
    analyze_rl_performance()