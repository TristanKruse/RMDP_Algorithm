#!/usr/bin/env python3
"""
Feature Importance Analysis for RL-ACA Postponement Strategy

This script analyzes which state features drive RL-ACA's postponement decisions
and identifies patterns that distinguish successful vs unsuccessful postponement strategies.

Based on investigation findings:
- Good districts (18, 8, 19): 17-18% postponement, 80-83% on-time
- Poor districts (20, 16): 12-13% postponement, 78-79% on-time
- Need to understand what makes postponement successful in some contexts but not others
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FeatureImportanceAnalyzer:
    """Analyze which features drive successful vs unsuccessful postponement decisions."""
    
    def __init__(self, results_dir: str = "data/simulation_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "postponement_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.df = self._load_data()
        
        # Categorize districts by postponement effectiveness
        self.district_categories = self._categorize_districts()
        
        # Analysis results storage
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
    
    def _categorize_districts(self) -> Dict[str, List[int]]:
        """Categorize districts by postponement effectiveness."""
        rl_data = self.df[self.df['method'] == 'rl_aca']
        
        # Calculate district characteristics
        district_stats = rl_data.groupby('district').agg({
            'postponement_rate': 'mean',
            'on_time_delivery_rate': 'mean',
            'total_delay': 'mean'
        })
        
        # Define categories based on investigation findings
        # Good: High postponement (>15%) AND high performance (>80%)
        # Poor: Low postponement (<15%) OR low performance (<78%)
        # Medium: Everything else
        
        good_districts = district_stats[
            (district_stats['postponement_rate'] > 15) & 
            (district_stats['on_time_delivery_rate'] > 80)
        ].index.tolist()
        
        poor_districts = district_stats[
            (district_stats['postponement_rate'] < 15) | 
            (district_stats['on_time_delivery_rate'] < 78)
        ].index.tolist()
        
        medium_districts = district_stats[
            ~district_stats.index.isin(good_districts + poor_districts)
        ].index.tolist()
        
        categories = {
            'good': good_districts,
            'medium': medium_districts,
            'poor': poor_districts
        }
        
        print(f"District Categories:")
        print(f"  Good (high postponement + high performance): {good_districts}")
        print(f"  Medium: {medium_districts}")
        print(f"  Poor (low postponement or low performance): {poor_districts}")
        
        return categories
    
    def analyze_postponement_patterns_by_category(self):
        """Analyze postponement patterns across district categories."""
        print("\n" + "="*60)
        print("POSTPONEMENT PATTERNS BY DISTRICT CATEGORY")
        print("="*60)
        
        rl_data = self.df[self.df['method'] == 'rl_aca']
        
        # Add district category to data
        def get_category(district):
            for category, districts in self.district_categories.items():
                if district in districts:
                    return category
            return 'unknown'
        
        rl_data = rl_data.copy()
        rl_data['district_category'] = rl_data['district'].apply(get_category)
        
        # Analyze patterns by category
        category_analysis = rl_data.groupby('district_category').agg({
            'postponement_rate': ['mean', 'std', 'min', 'max'],
            'on_time_delivery_rate': ['mean', 'std'],
            'total_delay': ['mean', 'std'],
            'avg_distance_per_order': ['mean', 'std'],
            'active_period_idle_rate': ['mean', 'std']
        }).round(3)
        
        print("Performance by District Category:")
        print(category_analysis)
        
        # Store findings
        self.findings['category_analysis'] = {
            'district_categories': self.district_categories,
            'performance_by_category': category_analysis.to_dict()
        }
        
        return rl_data
    
    def simulate_state_features(self, rl_data: pd.DataFrame):
        """Simulate the state features that would drive postponement decisions."""
        print("\n" + "="*60)
        print("STATE FEATURE SIMULATION AND ANALYSIS")
        print("="*60)
        
        # Based on the RL-ACA algorithm, simulate the key state features
        # Note: We don't have direct access to the actual state features used during simulation
        # But we can approximate them based on the dataset and domain knowledge
        
        print("Simulating state features based on RL-ACA algorithm specification...")
        
        # Simulate key features from the algorithm description
        features_data = []
        
        for _, row in rl_data.iterrows():
            # Extract date components for time-based features
            date = row['date']
            hour = 10 + (date.weekday() * 2)  # Approximate operational hour
            
            # 1. Time of day (normalized)
            time_feature = (hour % 24) / 24
            
            # 2. System utilization (approximate based on idle rate)
            # Higher idle rate = lower utilization
            system_utilization = 1 - row['active_period_idle_rate']
            
            # 3. Unassigned ratio (approximate based on postponement rate)
            # Higher postponement suggests more unassigned orders
            unassigned_ratio = row['postponement_rate'] / 100
            
            # 4. Order urgency (approximate based on delay patterns)
            # Higher avg delay suggests less urgent ordering
            order_urgency = max(0, min(1, 1 - (row['avg_delay_late_orders'] / 10)))
            
            # 5. Bundling potential (approximate based on restaurant density)
            # More restaurants per area = higher bundling potential
            # Use inverse of avg distance as proxy
            bundling_potential = max(0, min(5, 15 / row['avg_distance_per_order']))
            
            # 6. Restaurant congestion (approximate based on total delay)
            # Higher total delay suggests congestion
            restaurant_congestion = min(1, row['total_delay'] / 1000)
            
            features_data.append({
                'district': row['district'],
                'day': row['day'],
                'district_category': row['district_category'],
                'postponement_rate': row['postponement_rate'],
                'on_time_delivery_rate': row['on_time_delivery_rate'],
                'time_of_day': time_feature,
                'system_utilization': system_utilization,
                'unassigned_ratio': unassigned_ratio,
                'order_urgency': order_urgency,
                'bundling_potential': bundling_potential,
                'restaurant_congestion': restaurant_congestion
            })
        
        features_df = pd.DataFrame(features_data)
        
        print(f"Simulated features for {len(features_df)} datasets")
        print("\nFeature Summary by District Category:")
        
        feature_cols = ['time_of_day', 'system_utilization', 'unassigned_ratio', 
                       'order_urgency', 'bundling_potential', 'restaurant_congestion']
        
        category_features = features_df.groupby('district_category')[feature_cols].mean().round(3)
        print(category_features)
        
        return features_df
    
    def analyze_feature_importance(self, features_df: pd.DataFrame):
        """Analyze which features correlate with successful postponement."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        feature_cols = ['time_of_day', 'system_utilization', 'unassigned_ratio', 
                       'order_urgency', 'bundling_potential', 'restaurant_congestion']
        
        # Correlation with postponement rate
        print("--- Correlation with Postponement Rate ---")
        postponement_correlations = {}
        for feature in feature_cols:
            corr = features_df['postponement_rate'].corr(features_df[feature])
            postponement_correlations[feature] = corr
            print(f"  {feature}: {corr:.3f}")
        
        # Correlation with performance
        print("\n--- Correlation with On-Time Performance ---")
        performance_correlations = {}
        for feature in feature_cols:
            corr = features_df['on_time_delivery_rate'].corr(features_df[feature])
            performance_correlations[feature] = corr
            print(f"  {feature}: {corr:.3f}")
        
        # Feature differences between good and poor districts
        print("\n--- Feature Differences: Good vs Poor Districts ---")
        good_features = features_df[features_df['district_category'] == 'good'][feature_cols].mean()
        poor_features = features_df[features_df['district_category'] == 'poor'][feature_cols].mean()
        
        feature_differences = good_features - poor_features
        print("Feature values (Good - Poor):")
        for feature in feature_cols:
            diff = feature_differences[feature]
            direction = "higher" if diff > 0 else "lower"
            print(f"  {feature}: {diff:.3f} (good districts have {direction} values)")
        
        # Store findings
        self.findings['feature_importance'] = {
            'postponement_correlations': postponement_correlations,
            'performance_correlations': performance_correlations,
            'good_vs_poor_differences': feature_differences.to_dict()
        }
        
        return features_df
    
    def identify_successful_postponement_conditions(self, features_df: pd.DataFrame):
        """Identify conditions where postponement is most successful."""
        print("\n" + "="*60)
        print("SUCCESSFUL POSTPONEMENT CONDITIONS")
        print("="*60)
        
        # Define successful postponement: high postponement rate + high performance
        features_df['postponement_success_score'] = (
            (features_df['postponement_rate'] / features_df['postponement_rate'].max()) * 0.5 +
            (features_df['on_time_delivery_rate'] / features_df['on_time_delivery_rate'].max()) * 0.5
        )
        
        # Find top 25% most successful postponement scenarios
        top_quartile = features_df.nlargest(int(len(features_df) * 0.25), 'postponement_success_score')
        bottom_quartile = features_df.nsmallest(int(len(features_df) * 0.25), 'postponement_success_score')
        
        print("Characteristics of Most Successful Postponement (Top 25%):")
        feature_cols = ['time_of_day', 'system_utilization', 'unassigned_ratio', 
                       'order_urgency', 'bundling_potential', 'restaurant_congestion']
        
        top_means = top_quartile[feature_cols].mean()
        bottom_means = bottom_quartile[feature_cols].mean()
        
        for feature in feature_cols:
            top_val = top_means[feature]
            bottom_val = bottom_means[feature]
            diff = top_val - bottom_val
            print(f"  {feature}: {top_val:.3f} (vs {bottom_val:.3f} in worst, diff: {diff:+.3f})")
        
        print(f"\nSuccessful scenarios average:")
        print(f"  Postponement rate: {top_quartile['postponement_rate'].mean():.1f}%")
        print(f"  On-time rate: {top_quartile['on_time_delivery_rate'].mean():.1f}%")
        
        print(f"\nUnsuccessful scenarios average:")
        print(f"  Postponement rate: {bottom_quartile['postponement_rate'].mean():.1f}%")
        print(f"  On-time rate: {bottom_quartile['on_time_delivery_rate'].mean():.1f}%")
        
        # Store findings
        self.findings['success_conditions'] = {
            'top_quartile_features': top_means.to_dict(),
            'bottom_quartile_features': bottom_means.to_dict(),
            'success_metrics': {
                'top_postponement': top_quartile['postponement_rate'].mean(),
                'top_performance': top_quartile['on_time_delivery_rate'].mean(),
                'bottom_postponement': bottom_quartile['postponement_rate'].mean(),
                'bottom_performance': bottom_quartile['on_time_delivery_rate'].mean()
            }
        }
        
        return top_quartile, bottom_quartile
    
    def generate_feature_recommendations(self):
        """Generate recommendations for improved feature engineering."""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Based on analysis findings
        feature_importance = self.findings.get('feature_importance', {})
        success_conditions = self.findings.get('success_conditions', {})
        
        # Recommendation 1: Bundling potential enhancement
        bundling_corr = feature_importance.get('postponement_correlations', {}).get('bundling_potential', 0)
        if abs(bundling_corr) > 0.3:
            recommendations.append({
                'priority': 'HIGH',
                'feature': 'Enhanced Bundling Potential',
                'current_issue': f'Current bundling feature has correlation {bundling_corr:.3f} with postponement',
                'recommendation': 'Add real-time restaurant queue length, pending orders from same restaurant, historical bundling success rate',
                'expected_benefit': 'More accurate bundling opportunity assessment'
            })
        
        # Recommendation 2: Temporal features
        time_corr = feature_importance.get('performance_correlations', {}).get('time_of_day', 0)
        recommendations.append({
            'priority': 'MEDIUM',
            'feature': 'Advanced Temporal Features',
            'current_issue': f'Simple time feature has limited predictive power ({time_corr:.3f})',
            'recommendation': 'Add rush hour indicators, demand forecast, seasonal patterns, day-of-week effects',
            'expected_benefit': 'Better timing of postponement decisions'
        })
        
        # Recommendation 3: Spatial features
        recommendations.append({
            'priority': 'MEDIUM',
            'feature': 'Spatial Context Features',
            'current_issue': 'No spatial features in current state representation',
            'recommendation': 'Add customer density, distance to cluster centers, geographic constraints',
            'expected_benefit': 'Location-aware postponement decisions'
        })
        
        # Recommendation 4: System state features
        util_corr = feature_importance.get('performance_correlations', {}).get('system_utilization', 0)
        if abs(util_corr) > 0.2:
            recommendations.append({
                'priority': 'HIGH',
                'feature': 'Enhanced System State',
                'current_issue': f'System utilization shows {util_corr:.3f} correlation with performance',
                'recommendation': 'Add vehicle availability forecast, load balancing metrics, capacity constraints',
                'expected_benefit': 'Better understanding of system constraints'
            })
        
        # Recommendation 5: Historical features
        recommendations.append({
            'priority': 'LOW',
            'feature': 'Historical Context',
            'current_issue': 'No historical information in state representation',
            'recommendation': 'Add recent postponement outcomes, customer patience levels, restaurant performance history',
            'expected_benefit': 'Learn from past postponement success/failure'
        })
        
        # Print recommendations
        print("Recommended Feature Enhancements:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['feature']}")
            print(f"   Issue: {rec['current_issue']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Expected Benefit: {rec['expected_benefit']}")
        
        self.findings['feature_recommendations'] = recommendations
        return recommendations
    
    def create_feature_visualizations(self, features_df: pd.DataFrame):
        """Create visualizations of feature importance and patterns."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Importance Analysis for RL-ACA Postponement Strategy', 
                    fontsize=16, fontweight='bold')
        
        feature_cols = ['system_utilization', 'bundling_potential', 'order_urgency', 
                       'restaurant_congestion', 'unassigned_ratio', 'time_of_day']
        
        # Create scatter plots for each feature vs performance
        for i, feature in enumerate(feature_cols):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # Color by district category
            colors = {'good': 'green', 'medium': 'orange', 'poor': 'red'}
            for category in ['good', 'medium', 'poor']:
                category_data = features_df[features_df['district_category'] == category]
                if len(category_data) > 0:
                    ax.scatter(category_data[feature], category_data['on_time_delivery_rate'], 
                             color=colors[category], alpha=0.6, label=f'{category.title()} districts')
            
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('On-Time Delivery Rate (%)')
            ax.set_title(f'{feature.replace("_", " ").title()} vs Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "feature_importance_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Feature importance plot saved: {output_file}")
        plt.close()
    
    def save_analysis_results(self):
        """Save all analysis results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"feature_importance_analysis_{timestamp}.json"
        
        # Clean dict for JSON serialization
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
        
        print(f"\n💾 Feature analysis results saved: {output_file}")
        return output_file
    
    def run_complete_analysis(self):
        """Run the complete feature importance analysis."""
        print("🔬 FEATURE IMPORTANCE ANALYSIS FOR RL-ACA POSTPONEMENT")
        print("="*70)
        
        # Run all analyses
        rl_data = self.analyze_postponement_patterns_by_category()
        features_df = self.simulate_state_features(rl_data)
        features_df = self.analyze_feature_importance(features_df)
        top_quartile, bottom_quartile = self.identify_successful_postponement_conditions(features_df)
        recommendations = self.generate_feature_recommendations()
        self.create_feature_visualizations(features_df)
        
        # Save results
        results_file = self.save_analysis_results()
        
        # Summary
        print("\n" + "="*70)
        print("🎯 FEATURE ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"📊 District Categories:")
        for category, districts in self.district_categories.items():
            print(f"   {category.title()}: {len(districts)} districts {districts}")
        
        print(f"\n🔍 Key Feature Insights:")
        feature_importance = self.findings.get('feature_importance', {})
        perf_corrs = feature_importance.get('performance_correlations', {})
        
        # Find strongest correlations
        strong_features = {k: v for k, v in perf_corrs.items() if abs(v) > 0.3}
        for feature, corr in strong_features.items():
            direction = "improves" if corr > 0 else "hurts"
            print(f"   {feature}: {direction} performance (r={corr:.3f})")
        
        print(f"\n🚀 Top Improvement Opportunities:")
        high_priority = [r for r in recommendations if r['priority'] == 'HIGH']
        for rec in high_priority:
            print(f"   • {rec['feature']}: {rec['recommendation']}")
        
        print(f"\n📁 Detailed results: {results_file}")
        
        return self.findings

def main():
    """Main analysis function."""
    analyzer = FeatureImportanceAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()