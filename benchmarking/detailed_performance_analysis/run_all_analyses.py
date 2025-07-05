#!/usr/bin/env python3
"""
Run All Performance Analyses

Master script to execute all detailed performance analyses including:
- District-level performance analysis
- Demand-based performance analysis  
- Detailed postponement analysis
"""

import sys
from pathlib import Path
import traceback

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from district_performance_analysis import DistrictPerformanceAnalyzer
from demand_performance_analysis import DemandPerformanceAnalyzer
from detailed_postponement_analysis import DetailedPostponementAnalyzer


def run_all_analyses():
    """Execute all performance analyses in sequence."""
    print("🚀 Starting Comprehensive Performance Analysis Suite")
    print("=" * 60)
    
    analyses_results = {}
    
    # 1. District Performance Analysis
    print("\n" + "=" * 60)
    print("1️⃣  DISTRICT PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    try:
        district_analyzer = DistrictPerformanceAnalyzer()
        df, district_stats = district_analyzer.run_full_analysis()
        analyses_results['district_analysis'] = {
            'status': 'success',
            'data': df,
            'stats': district_stats
        }
        print("✅ District analysis completed successfully")
    except Exception as e:
        print(f"❌ District analysis failed: {e}")
        traceback.print_exc()
        analyses_results['district_analysis'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # 2. Demand-Based Performance Analysis
    print("\n" + "=" * 60)
    print("2️⃣  DEMAND-BASED PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    try:
        demand_analyzer = DemandPerformanceAnalyzer()
        demand_df = demand_analyzer.run_full_analysis()
        analyses_results['demand_analysis'] = {
            'status': 'success',
            'data': demand_df
        }
        print("✅ Demand analysis completed successfully")
    except Exception as e:
        print(f"❌ Demand analysis failed: {e}")
        traceback.print_exc()
        analyses_results['demand_analysis'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # 3. Detailed Postponement Analysis
    print("\n" + "=" * 60)
    print("3️⃣  DETAILED POSTPONEMENT ANALYSIS")
    print("=" * 60)
    
    try:
        postponement_analyzer = DetailedPostponementAnalyzer()
        postponement_data, bundling_data = postponement_analyzer.run_full_analysis()
        analyses_results['postponement_analysis'] = {
            'status': 'success',
            'postponement_data': postponement_data,
            'bundling_data': bundling_data
        }
        print("✅ Postponement analysis completed successfully")
    except Exception as e:
        print(f"❌ Postponement analysis failed: {e}")
        traceback.print_exc()
        analyses_results['postponement_analysis'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    total_analyses = len(analyses_results)
    successful_analyses = sum(1 for result in analyses_results.values() if result['status'] == 'success')
    
    print(f"Total analyses run: {total_analyses}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {total_analyses - successful_analyses}")
    
    if successful_analyses == total_analyses:
        print("\n🎉 All analyses completed successfully!")
        print("\n📁 Results saved to: benchmarking/detailed_performance_analysis/outputs/")
        print("\n📋 Generated files:")
        
        output_dir = Path("benchmarking/detailed_performance_analysis/outputs")
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            for file in sorted(files):
                print(f"   - {file.name}")
    else:
        print("\n⚠️  Some analyses failed. Check the error messages above.")
        
        # Show failed analyses
        failed_analyses = [name for name, result in analyses_results.items() if result['status'] == 'failed']
        print(f"\nFailed analyses: {', '.join(failed_analyses)}")
    
    return analyses_results


def main():
    """Main execution function."""
    return run_all_analyses()


if __name__ == "__main__":
    main()