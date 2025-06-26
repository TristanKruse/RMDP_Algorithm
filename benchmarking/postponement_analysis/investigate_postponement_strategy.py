#!/usr/bin/env python3
"""
Deep Investigation into RL-ACA Postponement Strategy

This script conducts a comprehensive analysis of RL-ACA's postponement decisions
to understand why the algorithm underperforms despite learning an active postponement strategy.

Analysis includes:
1. Postponement pattern analysis across contexts
2. Counterfactual simulation framework  
3. Feature importance investigation
4. Training-evaluation mismatch detection
5. Optimization recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PostponementInvestigator:
    """Comprehensive investigation of RL-ACA postponement strategy."""
    
    def __init__(self, results_dir: str = "data/simulation_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "postponement_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.df = self._load_data()
        
        # Key findings storage
        self.findings = {}
        
    def _load_data(self) -> pd.DataFrame:
        """Load the most recent filtered results."""
        filtered_files = list(self.results_dir.glob("fastest_aca_filtered_results_*.csv"))
        if not filtered_files:
            raise FileNotFoundError("No filtered results found!")
        
        latest_file = max(filtered_files, key=lambda x: x.stat().st_mtime)
        print(f"📊 Loading data from: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        df['date'] = pd.to_datetime(df['day'], format='%Y%m%d')
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_type'] = df['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        
        return df

    def analyze_postponement_patterns(self):
        """Analyze postponement behavior patterns across different contexts."""
        print("\n" + "="*60)
        print("POSTPONEMENT PATTERN ANALYSIS")
        print("="*60)
        
        # Basic postponement statistics
        rl_data = self.df[self.df['method'] == 'rl_aca']
        
        print(f"RL-ACA Postponement Statistics:")
        print(f"  Mean postponement rate: {rl_data['postponement_rate'].mean():.1f}%")
        print(f"  Std deviation: {rl_data['postponement_rate'].std():.1f}%")
        print(f"  Range: {rl_data['postponement_rate'].min():.1f}% - {rl_data['postponement_rate'].max():.1f}%")
        
        # Store finding
        self.findings['postponement_stats'] = {
            'mean': rl_data['postponement_rate'].mean(),
            'std': rl_data['postponement_rate'].std(),
            'min': rl_data['postponement_rate'].min(),
            'max': rl_data['postponement_rate'].max()
        }
        
        # Postponement by context analysis
        self._analyze_postponement_by_context(rl_data)
        self._analyze_postponement_performance_correlation(rl_data)
        self._create_postponement_visualizations(rl_data)
        
    def _analyze_postponement_by_context(self, rl_data: pd.DataFrame):
        """Analyze how postponement varies by operational context."""
        print(f"\n--- Postponement by Context ---")
        
        # By district
        district_postponement = rl_data.groupby('district').agg({
            'postponement_rate': ['mean', 'std'],
            'on_time_delivery_rate': 'mean',
            'total_delay': 'mean'
        }).round(2)
        
        district_postponement.columns = ['postponement_mean', 'postponement_std', 'ontime_rate', 'total_delay']
        district_postponement = district_postponement.sort_values('postponement_mean', ascending=False)
        
        print(f"\nPostponement by District (sorted by postponement rate):")
        print(district_postponement.head(10))
        
        # By day type
        day_type_postponement = rl_data.groupby('day_type').agg({
            'postponement_rate': ['mean', 'std'],
            'on_time_delivery_rate': 'mean',
            'total_delay': 'mean'
        }).round(2)
        
        print(f"\nPostponement by Day Type:")
        print(day_type_postponement)
        
        # Store findings
        self.findings['postponement_by_context'] = {
            'by_district': district_postponement.to_dict('index'),
            'by_day_type': day_type_postponement.to_dict('index')
        }
        
    def _analyze_postponement_performance_correlation(self, rl_data: pd.DataFrame):
        """Analyze correlation between postponement rate and performance."""
        print(f"\n--- Postponement-Performance Correlation ---")
        
        # Calculate correlations
        correlations = {}
        performance_metrics = ['on_time_delivery_rate', 'total_delay', 'avg_delay_late_orders', 'max_delay', 'avg_distance_per_order']
        
        for metric in performance_metrics:
            corr = rl_data['postponement_rate'].corr(rl_data[metric])
            correlations[metric] = corr
            print(f"  Postponement vs {metric}: {corr:.3f}")
        
        # Find strongest correlations
        strong_correlations = {k: v for k, v in correlations.items() if abs(v) > 0.3}
        if strong_correlations:
            print(f"\nStrong correlations (|r| > 0.3):")
            for metric, corr in strong_correlations.items():
                direction = "increases" if corr > 0 else "decreases"
                print(f"  Higher postponement → {direction} {metric} (r={corr:.3f})")
        else:
            print(f"\nNo strong correlations found (all |r| < 0.3)")
        
        self.findings['postponement_correlations'] = correlations
        
    def _create_postponement_visualizations(self, rl_data: pd.DataFrame):
        """Create visualizations of postponement patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RL-ACA Postponement Strategy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Postponement rate distribution
        axes[0,0].hist(rl_data['postponement_rate'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(rl_data['postponement_rate'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {rl_data["postponement_rate"].mean():.1f}%')
        axes[0,0].set_xlabel('Postponement Rate (%)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of Postponement Rates')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Postponement vs On-time Performance
        axes[0,1].scatter(rl_data['postponement_rate'], rl_data['on_time_delivery_rate'], 
                         alpha=0.6, color='green')
        z = np.polyfit(rl_data['postponement_rate'], rl_data['on_time_delivery_rate'], 1)
        p = np.poly1d(z)
        axes[0,1].plot(rl_data['postponement_rate'], p(rl_data['postponement_rate']), "r--", alpha=0.8)
        axes[0,1].set_xlabel('Postponement Rate (%)')
        axes[0,1].set_ylabel('On-Time Delivery Rate (%)')
        axes[0,1].set_title('Postponement vs Performance')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Postponement by District
        district_means = rl_data.groupby('district')['postponement_rate'].mean().sort_values(ascending=False)
        axes[1,0].bar(range(len(district_means)), district_means.values, color='orange', alpha=0.7)
        axes[1,0].set_xlabel('District (sorted by postponement rate)')
        axes[1,0].set_ylabel('Mean Postponement Rate (%)')
        axes[1,0].set_title('Postponement Rate by District')
        axes[1,0].set_xticks(range(len(district_means)))
        axes[1,0].set_xticklabels(district_means.index, rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Postponement vs Total Delay
        axes[1,1].scatter(rl_data['postponement_rate'], rl_data['total_delay'], 
                         alpha=0.6, color='purple')
        z = np.polyfit(rl_data['postponement_rate'], rl_data['total_delay'], 1)
        p = np.poly1d(z)
        axes[1,1].plot(rl_data['postponement_rate'], p(rl_data['postponement_rate']), "r--", alpha=0.8)
        axes[1,1].set_xlabel('Postponement Rate (%)')
        axes[1,1].set_ylabel('Total Delay (minutes)')
        axes[1,1].set_title('Postponement vs Total Delay')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "postponement_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Postponement analysis plot saved: {output_file}")
        plt.close()

    def simulate_counterfactual_strategies(self):
        """Simulate alternative postponement strategies to assess potential improvements."""
        print("\n" + "="*60)
        print("COUNTERFACTUAL STRATEGY SIMULATION")
        print("="*60)
        
        rl_data = self.df[self.df['method'] == 'rl_aca'].copy()
        aca_data = self.df[self.df['method'] == 'aca_17'].copy()
        
        # Strategy 1: No Postponement (RL-ACA with 0% postponement)
        print(f"\n--- Strategy 1: Zero Postponement ---")
        self._simulate_zero_postponement(rl_data, aca_data)
        
        # Strategy 2: Selective Postponement (only high-bundling scenarios)
        print(f"\n--- Strategy 2: Selective Postponement ---")
        self._simulate_selective_postponement(rl_data)
        
        # Strategy 3: Optimal Postponement (theoretical upper bound)
        print(f"\n--- Strategy 3: Theoretical Optimal ---")
        self._simulate_optimal_postponement(rl_data, aca_data)
        
    def _simulate_zero_postponement(self, rl_data: pd.DataFrame, aca_data: pd.DataFrame):
        """Simulate RL-ACA with zero postponement."""
        # Hypothesis: RL-ACA performance would improve if it never postponed
        # Use ACA-17 performance as proxy for RL without postponement
        
        avg_performance_gap = aca_data['on_time_delivery_rate'].mean() - rl_data['on_time_delivery_rate'].mean()
        avg_delay_gap = rl_data['total_delay'].mean() - aca_data['total_delay'].mean()
        
        print(f"  Current RL-ACA performance: {rl_data['on_time_delivery_rate'].mean():.1f}% on-time")
        print(f"  ACA-17 performance (no postponement): {aca_data['on_time_delivery_rate'].mean():.1f}% on-time")
        print(f"  Potential improvement from zero postponement: +{avg_performance_gap:.1f}pp")
        print(f"  Potential delay reduction: -{avg_delay_gap:.1f} minutes")
        
        self.findings['zero_postponement_potential'] = {
            'current_ontime': rl_data['on_time_delivery_rate'].mean(),
            'zero_postpone_ontime': aca_data['on_time_delivery_rate'].mean(),
            'potential_improvement': avg_performance_gap,
            'potential_delay_reduction': avg_delay_gap
        }
        
    def _simulate_selective_postponement(self, rl_data: pd.DataFrame):
        """Simulate selective postponement based on hypothetical optimization."""
        # Hypothesis: Only postpone in high-bundling-potential scenarios
        
        # Use postponement rate variation as proxy for selectivity
        # Districts with higher postponement variance might indicate more selective behavior
        district_stats = rl_data.groupby('district').agg({
            'postponement_rate': ['mean', 'std'],
            'on_time_delivery_rate': 'mean'
        })
        
        district_stats.columns = ['postpone_mean', 'postpone_std', 'performance']
        
        # Find districts with selective postponement (high variance)
        selective_districts = district_stats[district_stats['postpone_std'] > district_stats['postpone_std'].median()]
        uniform_districts = district_stats[district_stats['postpone_std'] <= district_stats['postpone_std'].median()]
        
        print(f"  Districts with selective postponement (high variance): {len(selective_districts)}")
        print(f"    Average performance: {selective_districts['performance'].mean():.1f}%")
        print(f"    Average postponement: {selective_districts['postpone_mean'].mean():.1f}%")
        
        print(f"  Districts with uniform postponement (low variance): {len(uniform_districts)}")
        print(f"    Average performance: {uniform_districts['performance'].mean():.1f}%")
        print(f"    Average postponement: {uniform_districts['postpone_mean'].mean():.1f}%")
        
        selectivity_benefit = selective_districts['performance'].mean() - uniform_districts['performance'].mean()
        print(f"  Selectivity benefit: {selectivity_benefit:.1f}pp")
        
        self.findings['selective_postponement'] = {
            'selective_performance': selective_districts['performance'].mean(),
            'uniform_performance': uniform_districts['performance'].mean(),
            'selectivity_benefit': selectivity_benefit
        }
        
    def _simulate_optimal_postponement(self, rl_data: pd.DataFrame, aca_data: pd.DataFrame):
        """Estimate theoretical optimal postponement performance."""
        # Hypothesis: Optimal postponement could achieve best of both worlds
        
        # Use best-case scenarios from each method
        rl_best = rl_data['on_time_delivery_rate'].max()
        aca_best = aca_data['on_time_delivery_rate'].max()
        theoretical_optimal = max(rl_best, aca_best)
        
        # Calculate potential if RL could achieve ACA efficiency + optimal postponement
        current_rl_avg = rl_data['on_time_delivery_rate'].mean()
        aca_avg = aca_data['on_time_delivery_rate'].mean()
        
        # Conservative estimate: ACA baseline + 50% of postponement benefit in best cases
        postponement_benefit = rl_best - current_rl_avg
        conservative_optimal = aca_avg + (postponement_benefit * 0.5)
        
        print(f"  Current RL-ACA average: {current_rl_avg:.1f}%")
        print(f"  ACA-17 average: {aca_avg:.1f}%")
        print(f"  RL-ACA best case: {rl_best:.1f}%")
        print(f"  Theoretical optimal (conservative): {conservative_optimal:.1f}%")
        print(f"  Improvement potential: +{conservative_optimal - current_rl_avg:.1f}pp")
        
        self.findings['optimal_postponement'] = {
            'current_avg': current_rl_avg,
            'aca_avg': aca_avg,
            'rl_best': rl_best,
            'theoretical_optimal': conservative_optimal,
            'improvement_potential': conservative_optimal - current_rl_avg
        }

    def investigate_training_evaluation_mismatch(self):
        """Investigate potential mismatches between training and evaluation environments."""
        print("\n" + "="*60)
        print("TRAINING-EVALUATION MISMATCH INVESTIGATION")
        print("="*60)
        
        # Analyze evaluation data characteristics
        rl_data = self.df[self.df['method'] == 'rl_aca']
        
        # Key characteristics that might differ from training
        print(f"--- Evaluation Environment Characteristics ---")
        print(f"  Districts analyzed: {rl_data['district'].nunique()}")
        print(f"  Date range: {rl_data['date'].min().strftime('%Y-%m-%d')} to {rl_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Weekday/Weekend split: {rl_data['day_type'].value_counts().to_dict()}")
        
        # Performance variance analysis
        performance_variance = rl_data['on_time_delivery_rate'].std()
        postponement_variance = rl_data['postponement_rate'].std()
        
        print(f"\n--- Performance Variability ---")
        print(f"  On-time rate std dev: {performance_variance:.1f}%")
        print(f"  Postponement rate std dev: {postponement_variance:.1f}%")
        print(f"  Performance range: {rl_data['on_time_delivery_rate'].min():.1f}% - {rl_data['on_time_delivery_rate'].max():.1f}%")
        
        # Hypothesis: High variance suggests environment diversity not captured in training
        if performance_variance > 5:
            print(f"  🚨 High performance variance suggests diverse operational conditions")
            print(f"     Training may not have covered this range of scenarios")
        
        if postponement_variance > 3:
            print(f"  🚨 High postponement variance suggests inconsistent learned policy")
            print(f"     May indicate training instability or environment mismatch")
        
        self.findings['training_mismatch'] = {
            'performance_variance': performance_variance,
            'postponement_variance': postponement_variance,
            'performance_range': (rl_data['on_time_delivery_rate'].min(), rl_data['on_time_delivery_rate'].max()),
            'high_variance_flags': {
                'performance': performance_variance > 5,
                'postponement': postponement_variance > 3
            }
        }

    def generate_optimization_recommendations(self):
        """Generate specific recommendations for improving RL-ACA performance."""
        print("\n" + "="*60)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Recommendation 1: Postponement Strategy
        if self.findings.get('zero_postponement_potential', {}).get('potential_improvement', 0) > 2:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Postponement Strategy',
                'issue': 'Current postponement decisions are counterproductive',
                'recommendation': 'Implement selective postponement: only postpone orders with high bundling potential (>2 orders from same restaurant)',
                'expected_benefit': f"+{self.findings['zero_postponement_potential']['potential_improvement']:.1f}pp on-time rate"
            })
        
        # Recommendation 2: Feature Engineering
        correlations = self.findings.get('postponement_correlations', {})
        if any(abs(corr) < 0.1 for corr in correlations.values()):
            recommendations.append({
                'priority': 'HIGH', 
                'category': 'Feature Engineering',
                'issue': 'Weak correlation between postponement and performance suggests poor state representation',
                'recommendation': 'Add spatial-temporal features: customer density, historical bundling success, real-time traffic',
                'expected_benefit': 'Improved postponement decision quality'
            })
        
        # Recommendation 3: Reward Function
        if self.findings.get('postponement_correlations', {}).get('total_delay', 0) > 0.2:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Reward Function',
                'issue': 'Postponement increases total delay - reward misalignment',
                'recommendation': 'Redesign reward: penalize postponement unless bundling achieved, add delay penalty',
                'expected_benefit': 'Better alignment between training and evaluation objectives'
            })
        
        # Recommendation 4: Training Curriculum
        if self.findings.get('training_mismatch', {}).get('high_variance_flags', {}).get('performance', False):
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Training Curriculum', 
                'issue': 'High performance variance suggests inadequate training diversity',
                'recommendation': 'Expand curriculum: include more diverse scenarios, longer training in realistic environments',
                'expected_benefit': 'More robust policy across operational contexts'
            })
        
        # Recommendation 5: Hybrid Approach
        recommendations.append({
            'priority': 'LOW',
            'category': 'Hybrid Architecture',
            'issue': 'RL shows promise but underperforms heuristics',
            'recommendation': 'Implement hybrid: use ACA for assignment, RL only for high-confidence postponement decisions',
            'expected_benefit': 'Combine ACA efficiency with strategic RL postponement'
        })
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['category']}")
            print(f"   Issue: {rec['issue']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Expected Benefit: {rec['expected_benefit']}")
        
        self.findings['recommendations'] = recommendations
        
        return recommendations

    def save_investigation_results(self):
        """Save all investigation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"postponement_investigation_{timestamp}.json"
        
        # Convert any non-serializable keys to strings
        def clean_dict(obj):
            if isinstance(obj, dict):
                return {str(k): clean_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_dict(item) for item in obj]
            else:
                return obj
        
        cleaned_findings = clean_dict(self.findings)
        
        with open(output_file, 'w') as f:
            json.dump(cleaned_findings, f, indent=2, default=str)
        
        print(f"\n💾 Investigation results saved: {output_file}")
        return output_file

    def run_full_investigation(self):
        """Run the complete postponement investigation."""
        print("🔬 DEEP INVESTIGATION: RL-ACA POSTPONEMENT STRATEGY")
        print("="*70)
        
        # Run all analyses
        self.analyze_postponement_patterns()
        self.simulate_counterfactual_strategies()
        self.investigate_training_evaluation_mismatch()
        recommendations = self.generate_optimization_recommendations()
        
        # Save results
        results_file = self.save_investigation_results()
        
        # Summary
        print("\n" + "="*70)
        print("🎯 INVESTIGATION SUMMARY")
        print("="*70)
        
        postponement_stats = self.findings.get('postponement_stats', {})
        zero_postpone = self.findings.get('zero_postponement_potential', {})
        
        print(f"📊 Current State:")
        print(f"   RL-ACA postpones {postponement_stats.get('mean', 0):.1f}% of orders")
        print(f"   Performance: {zero_postpone.get('current_ontime', 0):.1f}% on-time rate")
        
        print(f"\n🚀 Improvement Potential:")
        print(f"   Zero postponement: +{zero_postpone.get('potential_improvement', 0):.1f}pp")
        print(f"   Optimal strategy: +{self.findings.get('optimal_postponement', {}).get('improvement_potential', 0):.1f}pp")
        
        print(f"\n🎯 Top Priority Actions:")
        high_priority = [r for r in recommendations if r['priority'] == 'HIGH']
        for rec in high_priority[:3]:
            print(f"   • {rec['category']}: {rec['recommendation']}")
        
        print(f"\n📁 Detailed results: {results_file}")
        
        return self.findings

def main():
    """Main investigation function."""
    investigator = PostponementInvestigator()
    results = investigator.run_full_investigation()
    return results

if __name__ == "__main__":
    main()