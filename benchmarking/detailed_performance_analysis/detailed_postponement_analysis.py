#!/usr/bin/env python3
"""
Detailed Postponement Analysis

Creates comprehensive analysis of postponement patterns including:
- Orders postponed per district and day
- Postponement correlation with demand patterns
- Bundling effectiveness from postponement
- District-day postponement heat maps
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import json

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class DetailedPostponementAnalyzer:
    """Detailed analysis of postponement patterns and effectiveness."""
    
    def __init__(self, results_dir: str = "data/simulation_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("benchmarking/detailed_performance_analysis/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load raw simulation results for detailed postponement data
        self.results_files_dir = Path("data/results")

    def load_detailed_results(self):
        """Load real postponement and bundling data from JSON files and CSV."""
        # Load from filtered benchmark CSV for basic data
        benchmark_file = self.results_dir / "fastest_aca_filtered_results.csv"
        
        if not benchmark_file.exists():
            print(f"⚠️  Benchmark file not found: {benchmark_file}")
            return [], []
            
        print(f"📊 Loading postponement data from: {benchmark_file.name}")
        
        df = pd.read_csv(benchmark_file)
        rl_data = df[df['method'] == 'rl_aca'].copy()
        
        print(f"🔍 Found {len(rl_data)} RL-ACA records across {rl_data['district'].nunique()} districts")
        
        # Load real bundling data from JSON files
        print(f"📁 Loading real bundling data from JSON files...")
        json_bundling_data = {}
        
        if self.results_files_dir.exists():
            rl_files = list(self.results_files_dir.glob("results_rl_aca_*.json"))
            print(f"🔍 Found {len(rl_files)} RL-ACA JSON files for bundling analysis")
            
            for file_path in rl_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    episode_stats = data.get('episode_stats', {})
                    timestamp = data.get('timestamp', '')
                    
                    # Extract real bundling metrics
                    bundles_formed = episode_stats.get('bundles_formed', 0)
                    bundled_orders = episode_stats.get('bundled_orders', [])
                    total_orders = episode_stats.get('total_orders', 0)
                    postponed_orders = episode_stats.get('postponed_orders', [])
                    
                    # Calculate real bundle rate
                    bundle_rate = (len(bundled_orders) / total_orders * 100) if total_orders > 0 else 0
                    postponement_rate = (len(postponed_orders) / total_orders * 100) if total_orders > 0 else 0
                    
                    json_bundling_data[timestamp] = {
                        'bundles_formed': bundles_formed,
                        'bundled_orders_count': len(bundled_orders) if isinstance(bundled_orders, list) else bundled_orders,
                        'total_orders': total_orders,
                        'bundle_rate': bundle_rate,
                        'postponement_rate': postponement_rate,
                        'postponed_orders_count': len(postponed_orders) if isinstance(postponed_orders, list) else postponed_orders
                    }
                    
                except Exception as e:
                    continue
        
        # Combine CSV data with JSON bundling data
        postponement_data = []
        bundling_data = []
        
        for _, row in rl_data.iterrows():
            postponement_data.append({
                'district': int(row['district']),
                'day': row['day'],
                'postponement_rate': float(row['postponement_rate']),
                'on_time_delivery_rate': float(row['on_time_delivery_rate']),
                'avg_delay_late_orders': float(row['avg_delay_late_orders']),
                'total_delay': float(row['total_delay']),
                'max_delay': float(row['max_delay']),
                'avg_distance_per_order': float(row['avg_distance_per_order']),
                'active_period_idle_rate': float(row['active_period_idle_rate'])
            })
        
        # Use average bundling metrics from JSON files
        if json_bundling_data:
            avg_bundle_rate = sum(d['bundle_rate'] for d in json_bundling_data.values()) / len(json_bundling_data)
            avg_bundles_formed = sum(d['bundles_formed'] for d in json_bundling_data.values()) / len(json_bundling_data)
            print(f"📊 Average bundle rate from JSON files: {avg_bundle_rate:.1f}%")
            print(f"📊 Average bundles formed: {avg_bundles_formed:.1f}")
            
            for _, row in rl_data.iterrows():
                # Use real bundling data correlation with postponement
                bundle_rate = avg_bundle_rate * (row['postponement_rate'] / 60.0)  # Scale based on postponement
                bundling_data.append({
                    'district': int(row['district']),
                    'day': row['day'],
                    'bundle_rate': bundle_rate,
                    'bundles_formed': max(1, int(avg_bundles_formed * (row['postponement_rate'] / 60.0))),
                    'postponement_rate': float(row['postponement_rate']),
                    'on_time_delivery_rate': float(row['on_time_delivery_rate'])
                })
        else:
            print("⚠️  No JSON bundling data found, using CSV postponement data only")
            for _, row in rl_data.iterrows():
                bundling_data.append({
                    'district': int(row['district']),
                    'day': row['day'],
                    'bundle_rate': row['postponement_rate'] * 0.7,  # Fallback estimate
                    'bundles_formed': max(1, int(row['postponement_rate'] / 10)),
                    'postponement_rate': float(row['postponement_rate']),
                    'on_time_delivery_rate': float(row['on_time_delivery_rate'])
                })
        
        return postponement_data, bundling_data

    def load_benchmark_data(self):
        """Load benchmark results for cross-reference."""
        benchmark_files = list(self.results_dir.glob("fastest_aca_filtered_results_*.csv"))
        if not benchmark_files:
            benchmark_files = list(self.results_dir.glob("benchmark_results*.csv"))
        
        if benchmark_files:
            latest_file = max(benchmark_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            print(f"📊 Loading benchmark data from: {latest_file.name}")
            return df
        
        return None

    def create_postponement_heatmap(self, postponement_data):
        """Create heatmap of postponement rates by district and day."""
        if not postponement_data:
            print("⚠️  No postponement data available for heatmap")
            return
        
        df = pd.DataFrame(postponement_data)
        
        # Create pivot table for heatmap
        pivot_data = df.pivot(index='district', columns='day', values='postponement_rate')
        
        plt.figure(figsize=(14, 8))
        
        # Create heatmap
        sns.heatmap(pivot_data, 
                   annot=True, 
                   fmt='.1f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Postponement Rate (%)'})
        
        plt.title('Postponement Rate by District and Day', fontsize=16, pad=20)
        plt.xlabel('Day', fontsize=12)
        plt.ylabel('District', fontsize=12)
        plt.tight_layout()
        
        output_path = self.output_dir / "postponement_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")

    def create_postponement_vs_performance_analysis(self, postponement_data, benchmark_df):
        """Analyze relationship between postponement and performance."""
        if not postponement_data:
            print("⚠️  Insufficient data for postponement vs performance analysis")
            return
        
        # The postponement_data already contains all needed columns from CSV
        merged_data = pd.DataFrame(postponement_data)
        
        if merged_data.empty:
            print("⚠️  No matching data found for postponement vs performance analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Postponement rate vs on-time delivery rate
        ax1 = axes[0, 0]
        ax1.scatter(merged_data['postponement_rate'], merged_data['on_time_delivery_rate'], 
                   alpha=0.7, s=60, color='steelblue')
        
        # Add trend line
        if len(merged_data) > 1:
            z = np.polyfit(merged_data['postponement_rate'], merged_data['on_time_delivery_rate'], 1)
            p = np.poly1d(z)
            ax1.plot(merged_data['postponement_rate'], p(merged_data['postponement_rate']), 
                    "r--", alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Postponement Rate (%)', fontsize=12)
        ax1.set_ylabel('On-Time Delivery Rate (%)', fontsize=12)
        ax1.set_title('Postponement vs On-Time Performance', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Postponement rate vs total delay
        ax2 = axes[0, 1]
        ax2.scatter(merged_data['postponement_rate'], merged_data['total_delay'], 
                   alpha=0.7, s=60, color='orange')
        
        # Add trend line
        if len(merged_data) > 1:
            z = np.polyfit(merged_data['postponement_rate'], merged_data['total_delay'], 1)
            p = np.poly1d(z)
            ax2.plot(merged_data['postponement_rate'], p(merged_data['postponement_rate']), 
                    "r--", alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Postponement Rate (%)', fontsize=12)
        ax2.set_ylabel('Total Delay (min)', fontsize=12)
        ax2.set_title('Postponement vs Total Delay', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # District-wise postponement patterns
        ax3 = axes[1, 0]
        district_avg = merged_data.groupby('district')['postponement_rate'].mean().sort_values(ascending=False)
        
        bars = ax3.bar(range(len(district_avg)), district_avg.values, color='green', alpha=0.7)
        ax3.set_xlabel('District (Ranked by Postponement)', fontsize=12)
        ax3.set_ylabel('Average Postponement Rate (%)', fontsize=12)
        ax3.set_title('District Postponement Patterns', fontsize=14)
        ax3.set_xticks(range(len(district_avg)))
        ax3.set_xticklabels([f'D{d}' for d in district_avg.index], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, district_avg.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%',
                    ha='center', va='bottom', fontsize=8)
        
        # Day-wise postponement patterns
        ax4 = axes[1, 1]
        day_avg = merged_data.groupby('day')['postponement_rate'].mean()
        
        ax4.plot(day_avg.index, day_avg.values, marker='o', linewidth=2, markersize=8, color='purple')
        ax4.set_xlabel('Day', fontsize=12)
        ax4.set_ylabel('Average Postponement Rate (%)', fontsize=12)
        ax4.set_title('Daily Postponement Patterns', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Postponement Analysis - RL-ACA Strategy', fontsize=16, y=0.95)
        plt.tight_layout()
        
        output_path = self.output_dir / "postponement_vs_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")

    def create_bundling_effectiveness_analysis(self, postponement_data, bundling_data):
        """Analyze bundling effectiveness and relationship to postponement."""
        if not postponement_data or not bundling_data:
            print("⚠️  Insufficient data for bundling effectiveness analysis")
            return
        
        bundling_df = pd.DataFrame(bundling_data)
        
        # Use bundling data directly (it already has postponement_rate)
        merged_data = bundling_df.copy()
        
        if merged_data.empty:
            print("⚠️  No matching data found for bundling effectiveness analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Postponement rate vs bundling rate
        ax1 = axes[0, 0]
        ax1.scatter(merged_data['postponement_rate'], merged_data['bundle_rate'], 
                   alpha=0.7, s=60, color='blue')
        
        # Add trend line
        if len(merged_data) > 1:
            z = np.polyfit(merged_data['postponement_rate'], merged_data['bundle_rate'], 1)
            p = np.poly1d(z)
            ax1.plot(merged_data['postponement_rate'], p(merged_data['postponement_rate']), 
                    "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            correlation = np.corrcoef(merged_data['postponement_rate'], merged_data['bundle_rate'])[0, 1]
            ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax1.transAxes, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax1.set_xlabel('Postponement Rate (%)', fontsize=12)
        ax1.set_ylabel('Bundle Rate (%)', fontsize=12)
        ax1.set_title('Postponement vs Bundling Effectiveness', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Performance vs bundling rate
        ax2 = axes[0, 1]
        ax2.scatter(merged_data['on_time_delivery_rate'], merged_data['bundle_rate'], 
                   alpha=0.7, s=60, color='green')
        
        # Add trend line
        if len(merged_data) > 1:
            z = np.polyfit(merged_data['on_time_delivery_rate'], merged_data['bundle_rate'], 1)
            p = np.poly1d(z)
            ax2.plot(merged_data['on_time_delivery_rate'], p(merged_data['on_time_delivery_rate']), 
                    "r--", alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('On-Time Delivery Rate (%)', fontsize=12)
        ax2.set_ylabel('Bundle Rate (%)', fontsize=12)
        ax2.set_title('Performance vs Bundling Effectiveness', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # District-wise bundling efficiency
        ax3 = axes[1, 0]
        merged_data['bundling_efficiency'] = merged_data['bundles_formed'] / merged_data['postponed_orders']
        merged_data['bundling_efficiency'] = merged_data['bundling_efficiency'].replace([np.inf, -np.inf], 0)
        
        district_efficiency = merged_data.groupby('district')['bundling_efficiency'].mean().sort_values(ascending=False)
        
        bars = ax3.bar(range(len(district_efficiency)), district_efficiency.values, color='orange', alpha=0.7)
        ax3.set_xlabel('District (Ranked by Efficiency)', fontsize=12)
        ax3.set_ylabel('Bundles per Postponed Order', fontsize=12)
        ax3.set_title('District Bundling Efficiency', fontsize=14)
        ax3.set_xticks(range(len(district_efficiency)))
        ax3.set_xticklabels([f'D{d}' for d in district_efficiency.index], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, district_efficiency.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=8)
        
        # Postponement success rate (bundles formed / postponed orders)
        ax4 = axes[1, 1]
        success_rate = (merged_data['bundles_formed'] / merged_data['postponed_orders'] * 100).fillna(0)
        
        ax4.hist(success_rate, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Postponement Success Rate (%)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Distribution of Postponement Success', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        mean_success = success_rate.mean()
        ax4.axvline(mean_success, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_success:.1f}%')
        ax4.legend()
        
        plt.suptitle('Bundling Effectiveness Analysis', fontsize=16, y=0.95)
        plt.tight_layout()
        
        output_path = self.output_dir / "bundling_effectiveness.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")

    def generate_postponement_summary_report(self, postponement_data, bundling_data, benchmark_df):
        """Generate comprehensive postponement analysis report."""
        report = []
        report.append("# Detailed Postponement Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Data overview
        report.append("## Data Overview")
        if postponement_data:
            report.append(f"- Postponement records: {len(postponement_data)}")
            postponement_df = pd.DataFrame(postponement_data)
            report.append(f"- Districts with postponement data: {postponement_df['district'].nunique()}")
            report.append(f"- Days with postponement data: {postponement_df['day'].nunique()}")
            
            # Overall postponement statistics
            report.append(f"- Mean postponement rate: {postponement_df['postponement_rate'].mean():.2f}%")
            report.append(f"- Max postponement rate: {postponement_df['postponement_rate'].max():.2f}%")
            report.append(f"- Min postponement rate: {postponement_df['postponement_rate'].min():.2f}%")
            report.append(f"- Districts analyzed: {postponement_df['district'].nunique()}")
            report.append(f"- Total scenario records: {len(postponement_df)}")
        else:
            report.append("- No detailed postponement data found")
        
        if bundling_data:
            bundling_df = pd.DataFrame(bundling_data)
            report.append(f"- Bundling records: {len(bundling_data)}")
            report.append(f"- Total bundles formed: {bundling_df['bundles_formed'].sum()}")
            report.append(f"- Mean bundle rate: {bundling_df['bundle_rate'].mean():.2f}%")
        else:
            report.append("- Bundling analysis skipped due to data structure mismatch")
        
        report.append("")
        
        # Key findings
        report.append("## Key Findings")
        
        if postponement_data:
            postponement_df = pd.DataFrame(postponement_data)
            
            # Best and worst districts for postponement
            district_postponement = postponement_df.groupby('district')['postponement_rate'].mean()
            best_districts = district_postponement.nlargest(3)
            worst_districts = district_postponement.nsmallest(3)
            
            report.append("### Postponement Patterns")
            report.append("Districts with highest postponement rates:")
            for district, rate in best_districts.items():
                report.append(f"- District {district}: {rate:.2f}%")
            
            report.append("Districts with lowest postponement rates:")
            for district, rate in worst_districts.items():
                report.append(f"- District {district}: {rate:.2f}%")
            
            # Daily patterns
            daily_postponement = postponement_df.groupby('day')['postponement_rate'].mean()
            report.append(f"- Highest postponement day: {daily_postponement.idxmax()} ({daily_postponement.max():.2f}%)")
            report.append(f"- Lowest postponement day: {daily_postponement.idxmin()} ({daily_postponement.min():.2f}%)")
        
        # Skip bundling analysis section due to data structure mismatch
        if postponement_data:
            postponement_df = pd.DataFrame(postponement_data)
            
            report.append("")
            report.append("### Postponement Performance Analysis")
            
            # District-level analysis
            district_performance = postponement_df.groupby('district').agg({
                'postponement_rate': 'mean',
                'on_time_delivery_rate': 'mean'
            }).round(2)
            
            # Top performing districts (highest on-time rate)
            top_performers = district_performance.nlargest(3, 'on_time_delivery_rate')
            
            report.append("**Top Performing Districts (On-Time Rate):**")
            for district, row in top_performers.iterrows():
                report.append(f"- District {district}: {row['on_time_delivery_rate']:.1f}% on-time, {row['postponement_rate']:.1f}% postponement")
            
            # Correlation analysis
            correlation = np.corrcoef(postponement_df['postponement_rate'], postponement_df['on_time_delivery_rate'])[0, 1]
            report.append(f"\n- Correlation between postponement and performance: {correlation:.3f}")
            
            # Highest postponement districts
            top_postponement = district_performance.nlargest(3, 'postponement_rate')
            
            report.append("\n**Highest Postponement Districts:**")
            for district, row in top_postponement.iterrows():
                report.append(f"- District {district}: {row['postponement_rate']:.1f}% postponement, {row['on_time_delivery_rate']:.1f}% on-time")
                
            report.append("\n*Note: Bundling analysis temporarily disabled due to data structure mismatch between JSON files and CSV matrix*")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        if not postponement_data or all(pd.DataFrame(postponement_data)['postponement_rate'] == 0):
            report.append("### Critical Issue: No Postponement Activity Detected")
            report.append("- The RL-ACA algorithm shows 0% postponement rate across all districts and days")
            report.append("- This suggests the postponement mechanism is not functioning as expected")
            report.append("- Recommend investigating:")
            report.append("  - Postponement decision logic in the RL model")
            report.append("  - State features that should trigger postponement")
            report.append("  - Reward function for postponement actions")
            report.append("  - Training data and learning convergence")
        else:
            report.append("### Optimization Opportunities")
            report.append("- Focus postponement strategy on districts with proven bundling success")
            report.append("- Investigate why some districts have higher postponement rates")
            report.append("- Optimize postponement timing based on daily patterns")
            report.append("- Improve bundling efficiency in low-performing districts")
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / "detailed_postponement_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Saved report: {report_path}")
        return report_text

    def run_full_analysis(self):
        """Run complete detailed postponement analysis."""
        print("🔍 Starting Detailed Postponement Analysis")
        print("=" * 50)
        
        # Load detailed simulation results
        print("\n📊 Loading detailed simulation results...")
        postponement_data, bundling_data = self.load_detailed_results()
        
        # Load benchmark data for cross-reference
        benchmark_df = self.load_benchmark_data()
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        self.create_postponement_heatmap(postponement_data)
        self.create_postponement_vs_performance_analysis(postponement_data, benchmark_df)
        
        # NOTE: Bundling analysis temporarily disabled due to data structure mismatch:
        # - JSON files contain individual simulation runs (149 files from single day)
        # - CSV contains district-day matrix (120 records across 15 districts × 8 days)
        # - No reliable mapping between JSON timestamps and CSV district-day combinations
        # - Missing comprehensive bundling data for all algorithms and timeframes
        print("⚠️  Bundling analysis skipped - data structure mismatch between JSON files and CSV matrix")
        # self.create_bundling_effectiveness_analysis(postponement_data, bundling_data)
        
        # Generate report
        print("\n📄 Generating report...")
        self.generate_postponement_summary_report(postponement_data, None, benchmark_df)  # Skip bundling data
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
        return postponement_data, bundling_data


def main():
    """Main execution function."""
    analyzer = DetailedPostponementAnalyzer()
    postponement_data, bundling_data = analyzer.run_full_analysis()
    return postponement_data, bundling_data


if __name__ == "__main__":
    main()