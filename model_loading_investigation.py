#!/usr/bin/env python3
"""
Model Loading Investigation Script

This script investigates how your current system loads RL models
and identifies what needs to be changed for proper multi-model support.
"""

import os
import sys
from pathlib import Path
import importlib.util
import inspect


def investigate_model_loading():
    """Investigate current model loading system."""

    print("🔍 MODEL LOADING INVESTIGATION")
    print("=" * 50)

    # 1. Check models directory
    models_dir = Path("data/models")
    print(f"\n📁 Models Directory: {models_dir}")

    if models_dir.exists():
        model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
        print(f"Found {len(model_files)} model files:")

        for model_file in sorted(model_files):
            stat = model_file.stat()
            print(f"  📄 {model_file.name} ({stat.st_size/1024/1024:.1f} MB, modified: {stat.st_mtime})")

        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            print(f"\n🔄 Latest model (current behavior): {latest_model.name}")
    else:
        print("❌ Models directory not found!")

    # 2. Check training/train.py
    print(f"\n🔧 Training System Analysis:")
    train_file = Path("training/train.py")

    if train_file.exists():
        print(f"✅ Found training/train.py")

        # Try to import and analyze
        try:
            spec = importlib.util.spec_from_file_location("train_module", train_file)
            train_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_module)

            # Check run_test_episode function
            if hasattr(train_module, "run_test_episode"):
                func = train_module.run_test_episode
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                print(f"📋 run_test_episode parameters: {params}")

                # Check for model path related parameters
                model_params = [p for p in params if "model" in p.lower() or "path" in p.lower()]
                if model_params:
                    print(f"🎯 Model-related parameters: {model_params}")
                else:
                    print("❌ No obvious model path parameters found")

            else:
                print("❌ run_test_episode function not found")

            # Check for compare_models function
            if hasattr(train_module, "compare_models"):
                func = train_module.compare_models
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                print(f"📋 compare_models parameters: {params}")

        except Exception as e:
            print(f"⚠️  Could not analyze training/train.py: {e}")
    else:
        print("❌ training/train.py not found!")

    # 3. Check algorithm_benchmarking.py
    print(f"\n📊 Current Benchmarking Analysis:")
    benchmark_file = Path("algorithm_benchmarking.py")

    if benchmark_file.exists():
        with open(benchmark_file, "r") as f:
            content = f.read()

        # Look for method definitions
        if "methods = [" in content:
            import re

            methods_match = re.search(r"methods\s*=\s*\[(.*?)\]", content, re.DOTALL)
            if methods_match:
                methods = methods_match.group(1)
                print(f"📋 Current methods: {methods}")

        # Look for run_test_episode calls
        if "run_test_episode(" in content:
            print("✅ Uses run_test_episode")

            # Look for model path parameters
            if "model_path" in content or "rl_model_path" in content:
                print("✅ Already has model path support")
            else:
                print("❌ No model path parameters found")

    else:
        print("❌ algorithm_benchmarking.py not found!")


def check_current_benchmark_usage():
    """Check how run_test_episode is currently used."""

    print(f"\n🔬 CURRENT USAGE ANALYSIS")
    print("=" * 30)

    # Look at actual function calls in algorithm_benchmarking.py
    benchmark_file = Path("algorithm_benchmarking.py")

    if benchmark_file.exists():
        with open(benchmark_file, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "run_test_episode(" in line:
                print(f"Line {i+1}: {line.strip()}")

                # Show context (next few lines)
                for j in range(1, 4):
                    if i + j < len(lines) and lines[i + j].strip():
                        print(f"Line {i+j+1}: {lines[i+j].strip()}")
                break


def generate_integration_recommendations():
    """Generate recommendations for proper model integration."""

    print(f"\n💡 INTEGRATION RECOMMENDATIONS")
    print("=" * 40)

    models_dir = Path("data/models")
    model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth")) if models_dir.exists() else []

    print("1. **Model File Organization**:")
    if len(model_files) > 1:
        print("   ✅ Multiple models found - good for comparison")
        print("   📝 Recommend naming convention:")
        print("      - rl_aca_phase2_final.pt")
        print("      - rl_aca_phase3_final.pt")
        print("      - rl_aca_latest.pt")
    else:
        print("   ⚠️  Only one model found - need phase 2 and phase 3 models")

    print("\n2. **Code Integration Steps**:")
    print("   a) Check if run_test_episode accepts rl_model_path parameter")
    print("   b) If not, modify it to support specific model loading")
    print("   c) Update algorithm_benchmarking.py to use model-specific configs")
    print("   d) Use incremental benchmarking to test different models")

    print("\n3. **Testing Strategy**:")
    print("   a) First test: run same model twice to verify consistency")
    print("   b) Second test: compare phase 2 vs phase 3 models")
    print("   c) Analyze stuck rate differences")


def main():
    """Main investigation function."""

    investigate_model_loading()
    check_current_benchmark_usage()
    generate_integration_recommendations()

    print(f"\n🎯 NEXT STEPS")
    print("=" * 15)
    print("1. Run this script to understand current system")
    print("2. Check run_test_episode function signature")
    print("3. Rename your models with specific names (phase2, phase3)")
    print("4. Implement proper model path support")
    print("5. Test incremental benchmarking")


if __name__ == "__main__":
    main()
