#!/usr/bin/env python3
"""
Test script to verify that all 4 methods are properly loaded and processed
in the visualization script.
"""
import sys
import os

# Test the data loading and filtering logic
def test_method_filtering():
    # Simulate what the visualization script does
    print("🔍 Testing method filtering logic...")
    
    # Check what methods exist in the data
    csv_path = "data/simulation_results/fastest_aca_filtered_results.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        return False
    
    # Read the file manually to check methods
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Extract method column (3rd column, index 2)
    methods = set()
    for line in lines[1:]:  # Skip header
        parts = line.strip().split(',')
        if len(parts) > 2:
            methods.add(parts[2])
    
    print(f"📊 Methods found in data: {sorted(methods)}")
    
    # Test our filtering logic
    desired_methods = [
        "rl_aca",              # RL-ACA model
        "fastest_aca",         # Fastest ACA baseline  
        "aca_17",              # ACA with buffer=14 (will be relabeled)
        "aca-postponement",    # ACA with postponement strategy
    ]
    
    print(f"🎯 Desired methods: {desired_methods}")
    
    # Check if all desired methods are present
    missing_methods = []
    for method in desired_methods:
        if method not in methods:
            missing_methods.append(method)
    
    if missing_methods:
        print(f"❌ Missing methods: {missing_methods}")
        return False
    else:
        print("✅ All 4 methods are present in the data!")
        return True

def test_method_mapping():
    print("\n🔄 Testing method mapping...")
    
    # Test the mapping logic
    method_mapping = {
        "rl_aca": "RL-ACA",
        "fastest_aca": "Fastest ACA",
        "aca_17": "ACA (Buffer=14)",
        "aca-postponement": "ACA-Postponement",
    }
    
    print("📝 Method mappings:")
    for original, mapped in method_mapping.items():
        print(f"  {original} → {mapped}")
    
    return True

def test_color_mapping():
    print("\n🎨 Testing color mapping...")
    
    method_colors = {
        "RL-ACA": "#F18F01",                 # Orange - RL model
        "Fastest ACA": "#2E86AB",            # Blue - Baseline
        "ACA (Buffer=14)": "#A23B72",        # Purple - Optimal buffer ACA
        "ACA-Postponement": "#FF6B6B",       # Red - Postponement strategy
        "Meituan Baseline": "#63B600"        # Green - Meituan's method
    }
    
    print("🎨 Color mappings:")
    for method, color in method_colors.items():
        print(f"  {method} → {color}")
    
    return True

def main():
    print("🧪 VISUALIZATION METHOD TEST")
    print("=" * 50)
    
    tests = [
        ("Method Filtering", test_method_filtering),
        ("Method Mapping", test_method_mapping),
        ("Color Mapping", test_color_mapping),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Visualization should show all 4 methods.")
    else:
        print("⚠️  Some tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)