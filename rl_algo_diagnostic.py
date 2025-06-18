#!/usr/bin/env python3
"""
RL Algorithm Diagnostic Tool

Analyzes RL-ACA decision patterns to identify:
1. Postponement rate patterns
2. Scenarios where RL gets "stuck"
3. Training coverage gaps
4. Decision confidence levels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLDiagnosticAnalyzer:
    """
    Diagnostic tool for analyzing RL algorithm behavior patterns.
    """
    
    def __init__(self, results_dir: str = "data/simulation_results"):
        self.results_dir = Path(results_dir)
        self.viz_dir = self.results_dir / "rl_diagnostics"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def load_filtered_data(self) -> pd.DataFrame:
        """Load the filtered benchmark data."""
        filtered_files = list(self.results_dir.glob("fastest_aca_filtered_results_*.csv"))
        
        if not filtered_files:
            raise FileNotFoundError("No filtered benchmark results found!")
        
        latest_file = max(filtered_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading filtered data from: {latest_file}")
        
        return pd.read_csv(latest_file)
    
    def analyze_postponement_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze RL postponement patterns to identify problematic scenarios.
        """
        logger.info("Analyzing RL postponement patterns...")
        
        # Filter RL data
        rl_data = df[df['method'] == 'rl_aca'].copy()
        
        if len(rl_data) == 0:
            logger.warning("No RL data found!")
            return {}
        
        analysis = {}
        
        # 1. Identify extreme postponement scenarios
        # Proxy: Very high delay suggests excessive postponement
        delay_threshold = rl_data['total_delay'].quantile(0.9)  # Top 10% worst delays
        
        extreme_delay_scenarios = rl_data[rl_data['total_delay'] >= delay_threshold]
        
        analysis['extreme_scenarios'] = {
            'count': len(extreme_delay_scenarios),
            'districts': extreme_delay_scenarios['district'].unique().tolist(),
            'avg_delay': extreme_delay_scenarios['total_delay'].mean(),
            'avg_on_time': extreme_delay_scenarios['on_time_delivery_rate'].mean()
        }
        
        # 2. Compare RL performance vs baselines by scenario complexity
        scenario_analysis = []
        
        for (district, day), group in df.groupby(['district', 'day']):
            scenario_data = {
                'district': district,
                'day': day
            }
            
            # Get performance for each method
            for method in group['method'].unique():
                method_perf = group[group['method'] == method].iloc[0]
                scenario_data[f'{method}_delay'] = method_perf['total_delay']
                scenario_data[f'{method}_on_time'] = method_perf['on_time_delivery_rate']
            
            # Calculate RL performance gap
            if 'rl_aca_delay' in scenario_data and 'fastest_aca_delay' in scenario_data:
                baseline_delay = scenario_data['fastest_aca_delay']
                rl_delay = scenario_data['rl_aca_delay']
                
                if baseline_delay > 0:
                    scenario_data['rl_delay_ratio'] = rl_delay / baseline_delay
                    scenario_data['rl_performance_gap'] = rl_delay - baseline_delay
                
                # Classify scenario difficulty
                if baseline_delay < 20:
                    scenario_data['difficulty'] = 'easy'
                elif baseline_delay < 50:
                    scenario_data['difficulty'] = 'medium'  
                else:
                    scenario_data['difficulty'] = 'hard'
            
            scenario_analysis.append(scenario_data)
        
        scenario_df = pd.DataFrame(scenario_analysis)
        analysis['scenario_breakdown'] = scenario_df
        
        # 3. Identify "stuck" scenarios (RL much worse than baseline)
        if 'rl_delay_ratio' in scenario_df.columns:
            stuck_threshold = 3.0  # RL delay 3x worse than baseline
            stuck_scenarios = scenario_df[scenario_df['rl_delay_ratio'] >= stuck_threshold]
            
            analysis['stuck_scenarios'] = {
                'count': len(stuck_scenarios),
                'percentage': (len(stuck_scenarios) / len(scenario_df)) * 100,
                'districts': stuck_scenarios['district'].unique().tolist() if len(stuck_scenarios) > 0 else [],
                'avg_ratio': stuck_scenarios['rl_delay_ratio'].mean() if len(stuck_scenarios) > 0 else 0
            }
        
        return analysis
    
    def identify_training_gaps(self, df: pd.DataFrame) -> Dict:
        """
        Identify potential training coverage gaps.
        """
        logger.info("Identifying potential training gaps...")
        
        rl_data = df[df['method'] == 'rl_aca'].copy()
        baseline_data = df[df['method'] == 'fastest_aca'].copy()
        
        gaps = {}
        
        # 1. District-specific performance analysis
        district_performance = []
        
        for district in rl_data['district'].unique():
            rl_district = rl_data[rl_data['district'] == district]
            baseline_district = baseline_data[baseline_data['district'] == district]
            
            if len(rl_district) > 0 and len(baseline_district) > 0:
                rl_avg = rl_district['on_time_delivery_rate'].mean()
                baseline_avg = baseline_district['on_time_delivery_rate'].mean()
                gap = baseline_avg - rl_avg
                
                district_performance.append({
                    'district': district,
                    'rl_performance': rl_avg,
                    'baseline_performance': baseline_avg,
                    'performance_gap': gap,
                    'rl_variability': rl_district['on_time_delivery_rate'].std()
                })
        
        district_df = pd.DataFrame(district_performance)
        
        # Identify worst-performing districts (potential training gaps)
        if len(district_df) > 0:
            worst_districts = district_df.nlargest(5, 'performance_gap')
            gaps['worst_districts'] = worst_districts.to_dict('records')
        
        # 2. Day-pattern analysis
        day_performance = []
        
        for day in rl_data['day'].unique():
            rl_day = rl_data[rl_data['day'] == day]
            baseline_day = baseline_data[baseline_data['day'] == day]
            
            if len(rl_day) > 0 and len(baseline_day) > 0:
                day_performance.append({
                    'day': day,
                    'rl_performance': rl_day['on_time_delivery_rate'].mean(),
                    'baseline_performance': baseline_day['on_time_delivery_rate'].mean(),
                    'performance_gap': baseline_day['on_time_delivery_rate'].mean() - rl_day['on_time_delivery_rate'].mean()
                })
        
        day_df = pd.DataFrame(day_performance)
        if len(day_df) > 0:
            gaps['day_patterns'] = day_df.to_dict('records')
        
        return gaps
    
    def create_diagnostic_visualizations(self, df: pd.DataFrame, analysis: Dict):
        """Create diagnostic visualizations."""
        logger.info("Creating diagnostic visualizations...")
        
        # 1. RL Performance Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RL Algorithm Diagnostic Analysis', fontsize=16, fontweight='bold')
        
        # Subplot 1: Performance by method
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='method', y='on_time_delivery_rate', ax=ax1)
        ax1.set_title('On-Time Rate Distribution by Method')
        ax1.tick_params(axis='x', rotation=45)
        
        # Subplot 2: RL delay vs baseline delay scatter
        ax2 = axes[0, 1]
        if 'scenario_breakdown' in analysis:
            scenario_df = analysis['scenario_breakdown']
            if 'rl_aca_delay' in scenario_df.columns and 'fastest_aca_delay' in scenario_df.columns:
                ax2.scatter(scenario_df['fastest_aca_delay'], scenario_df['rl_aca_delay'], alpha=0.6)
                
                # Add diagonal line (perfect performance)
                max_delay = max(scenario_df['fastest_aca_delay'].max(), scenario_df['rl_aca_delay'].max())
                ax2.plot([0, max_delay], [0, max_delay], 'r--', alpha=0.5, label='Perfect Performance')
                
                ax2.set_xlabel('Baseline Delay (minutes)')
                ax2.set_ylabel('RL Delay (minutes)')
                ax2.set_title('RL vs Baseline Performance\n(Points above line = RL worse)')
                ax2.legend()
        
        # Subplot 3: District performance gaps
        ax3 = axes[1, 0]
        rl_by_district = df[df['method'] == 'rl_aca'].groupby('district')['on_time_delivery_rate'].mean()
        baseline_by_district = df[df['method'] == 'fastest_aca'].groupby('district')['on_time_delivery_rate'].mean()
        
        district_gaps = baseline_by_district - rl_by_district
        district_gaps.plot(kind='bar', ax=ax3, color='orangered', alpha=0.7)
        ax3.set_title('Performance Gap by District\n(Higher = RL worse)')
        ax3.set_ylabel('Gap (percentage points)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Subplot 4: Scenario difficulty analysis
        ax4 = axes[1, 1]
        if 'scenario_breakdown' in analysis:
            scenario_df = analysis['scenario_breakdown']
            if 'difficulty' in scenario_df.columns and 'rl_delay_ratio' in scenario_df.columns:
                difficulty_performance = scenario_df.groupby('difficulty')['rl_delay_ratio'].mean()
                difficulty_performance.plot(kind='bar', ax=ax4, color=['green', 'orange', 'red'], alpha=0.7)
                ax4.set_title('RL Performance by Scenario Difficulty\n(Lower = Better)')
                ax4.set_ylabel('RL/Baseline Delay Ratio')
                ax4.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "rl_diagnostic_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed postponement pattern analysis
        self._create_postponement_analysis_chart(df, analysis)
    
    def _create_postponement_analysis_chart(self, df: pd.DataFrame, analysis: Dict):
        """Create detailed postponement pattern analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Postponement Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Get RL data
        rl_data = df[df['method'] == 'rl_aca']
        
        # 1. Delay distribution with extreme scenarios highlighted
        ax1 = axes[0, 0]
        ax1.hist(rl_data['total_delay'], bins=20, alpha=0.7, color='orange', label='All RL scenarios')
        
        if 'extreme_scenarios' in analysis and analysis['extreme_scenarios']['count'] > 0:
            extreme_delays = rl_data[rl_data['total_delay'] >= rl_data['total_delay'].quantile(0.9)]['total_delay']
            ax1.hist(extreme_delays, bins=10, alpha=0.8, color='red', label='Extreme delays (top 10%)')
        
        ax1.set_xlabel('Total Delay (minutes)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('RL Delay Distribution')
        ax1.legend()
        
        # 2. Performance by district with stuck scenarios highlighted
        ax2 = axes[0, 1]
        district_performance = rl_data.groupby('district')['on_time_delivery_rate'].mean()
        bars = ax2.bar(district_performance.index, district_performance.values, alpha=0.7, color='blue')
        
        # Highlight stuck scenarios
        if 'stuck_scenarios' in analysis and len(analysis['stuck_scenarios']['districts']) > 0:
            stuck_districts = analysis['stuck_scenarios']['districts']
            for i, district in enumerate(district_performance.index):
                if district in stuck_districts:
                    bars[i].set_color('red')
                    bars[i].set_alpha(0.8)
        
        ax2.set_xlabel('District')
        ax2.set_ylabel('On-Time Rate (%)')
        ax2.set_title('RL Performance by District\n(Red = Stuck scenarios)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Scenario difficulty vs RL performance
        ax3 = axes[1, 0]
        if 'scenario_breakdown' in analysis:
            scenario_df = analysis['scenario_breakdown']
            if 'fastest_aca_delay' in scenario_df.columns and 'rl_aca_on_time' in scenario_df.columns:
                ax3.scatter(scenario_df['fastest_aca_delay'], scenario_df['rl_aca_on_time'], 
                           alpha=0.6, c=scenario_df['fastest_aca_delay'], cmap='RdYlBu_r')
                ax3.set_xlabel('Baseline Delay (Scenario Difficulty)')
                ax3.set_ylabel('RL On-Time Rate (%)')
                ax3.set_title('RL Performance vs Scenario Difficulty')
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = "RL DIAGNOSTIC SUMMARY\n\n"
        
        if 'extreme_scenarios' in analysis:
            extreme = analysis['extreme_scenarios']
            summary_text += f"Extreme Delay Scenarios: {extreme['count']}\n"
            summary_text += f"Avg Delay: {extreme['avg_delay']:.1f} min\n"
            summary_text += f"Avg On-Time: {extreme['avg_on_time']:.1f}%\n\n"
        
        if 'stuck_scenarios' in analysis:
            stuck = analysis['stuck_scenarios']
            summary_text += f"Stuck Scenarios: {stuck['count']}\n"
            summary_text += f"Percentage: {stuck['percentage']:.1f}%\n"
            summary_text += f"Avg Delay Ratio: {stuck['avg_ratio']:.1f}x\n\n"
        
        summary_text += "RECOMMENDATIONS:\n"
        summary_text += "1. Implement safety fallback\n"
        summary_text += "2. Retrain on difficult scenarios\n"
        summary_text += "3. Add confidence thresholds\n"
        summary_text += "4. Use hybrid approach"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "rl_postponement_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_diagnostic_report(self, analysis: Dict, gaps: Dict) -> str:
        """Generate comprehensive diagnostic report."""
        
        report = f"""# RL Algorithm Diagnostic Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This diagnostic analysis examines RL-ACA decision patterns to identify scenarios where the algorithm gets "stuck" in suboptimal behavior, particularly excessive postponement.

## Key Findings

### Extreme Performance Scenarios
"""
        
        if 'extreme_scenarios' in analysis:
            extreme = analysis['extreme_scenarios']
            report += f"""
- **Extreme delay scenarios**: {extreme['count']} identified
- **Affected districts**: {extreme['districts']}
- **Average delay in extreme cases**: {extreme['avg_delay']:.1f} minutes
- **Average on-time rate**: {extreme['avg_on_time']:.1f}%
"""
        
        if 'stuck_scenarios' in analysis:
            stuck = analysis['stuck_scenarios']
            report += f"""
### "Stuck" Scenarios Analysis
- **Scenarios where RL ≥3x worse than baseline**: {stuck['count']}
- **Percentage of all scenarios**: {stuck['percentage']:.1f}%
- **Affected districts**: {stuck['districts']}
- **Average performance degradation**: {stuck['avg_ratio']:.1f}x worse than baseline
"""
        
        report += """
### Potential Root Causes

1. **Training Coverage Gaps**: RL may not have seen similar scenarios during training
2. **Exploration-Exploitation Issues**: Getting stuck in local optima (excessive postponement)
3. **State Representation**: Current state may not capture scenario complexity adequately
4. **Reward Function**: May inadvertently reward postponement in edge cases

## Training Gap Analysis
"""
        
        if 'worst_districts' in gaps:
            report += "\n### Worst Performing Districts (Potential Training Gaps)\n"
            for district_info in gaps['worst_districts'][:3]:  # Top 3
                report += f"- **District {district_info['district']}**: {district_info['performance_gap']:.1f}pp gap, {district_info['rl_variability']:.1f}% variability\n"
        
        report += """
## Recommended Solutions

### 1. Safety Fallback Mechanism
```python
# Implement confidence-based fallback
if model_confidence < threshold or recent_performance < minimum:
    return NO_POSTPONEMENT  # Conservative default
```

### 2. Scenario-Specific Training
- Identify problematic district-day combinations
- Generate synthetic training data for edge cases  
- Use curriculum learning (easy → hard scenarios)

### 3. Hybrid Decision Making
- Combine RL with heuristic fallbacks
- Use ensemble methods for robustness
- Implement maximum postponement limits

### 4. Enhanced State Representation
- Include scenario difficulty indicators
- Add historical performance context
- Incorporate time-of-day and district characteristics

## Implementation Priority

1. **Immediate**: Safety fallback for problematic districts
2. **Short-term**: Retrain with focus on identified gap scenarios
3. **Medium-term**: Implement hybrid RL-heuristic approach
4. **Long-term**: Enhanced state representation and training curriculum
"""
        
        return report
    
    def run_full_diagnostic(self):
        """Run complete diagnostic analysis."""
        logger.info("Starting RL diagnostic analysis...")
        
        # Load data
        df = self.load_filtered_data()
        logger.info(f"Loaded {len(df)} records for analysis")
        
        # Run analyses
        postponement_analysis = self.analyze_postponement_patterns(df)
        training_gaps = self.identify_training_gaps(df)
        
        # Create visualizations
        self.create_diagnostic_visualizations(df, postponement_analysis)
        
        # Generate report
        report = self.generate_diagnostic_report(postponement_analysis, training_gaps)
        
        # Save report
        report_path = self.viz_dir / f"rl_diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Diagnostic analysis complete!")
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Visualizations saved to: {self.viz_dir}")
        
        return postponement_analysis, training_gaps


def main():
    """Main function to run RL diagnostics."""
    analyzer = RLDiagnosticAnalyzer()
    
    try:
        analysis, gaps = analyzer.run_full_diagnostic()
        
        print("\n" + "="*60)
        print("RL DIAGNOSTIC ANALYSIS COMPLETE")
        print("="*60)
        
        # Print key findings
        if 'stuck_scenarios' in analysis:
            stuck = analysis['stuck_scenarios']
            print(f"🔍 Found {stuck['count']} 'stuck' scenarios ({stuck['percentage']:.1f}%)")
            print(f"📍 Affected districts: {stuck['districts']}")
        
        if 'extreme_scenarios' in analysis:
            extreme = analysis['extreme_scenarios']
            print(f"⚠️  {extreme['count']} extreme delay scenarios identified")
        
        print(f"\n📊 Diagnostic visualizations: data/simulation_results/rl_diagnostics/")
        print(f"📋 Detailed report generated with specific recommendations")
        
    except Exception as e:
        logger.error(f"Diagnostic analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()