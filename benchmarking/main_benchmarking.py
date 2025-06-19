#!/usr/bin/env python3
"""
Main Benchmarking Analysis Runner with Incremental Model Support

This script orchestrates the complete benchmarking analysis pipeline:
0. Incremental model benchmarking (NEW - detects and benchmarks new RL models)
1. Data filtering (remove problematic districts)
2. Statistical analysis and reporting
3. Advanced visualization generation

Usage:
    python main_benchmarking.py [options]

New Options:
    --model-name [name]          # Give current model a custom name
    --start-fresh                # Start benchmarking from scratch
    --skip-model-benchmark       # Skip model benchmarking, only run analysis
"""

import os
import sys
import subprocess
import time
import logging
import glob
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
from typing import Optional, List, Dict

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class IncrementalModelBenchmarker:
    """Handle incremental model detection and benchmarking."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.models_dir = Path("data/models")

        # Define benchmarking parameters
        self.districts = list(range(1, 23))  # Districts 1-22
        self.days = [f"202210{day:02d}" for day in range(17, 25)]  # Oct 17-24, 2022
        self.kpis = [
            "on_time_delivery_rate",
            "active_period_idle_rate",
            "avg_delay_late_orders",
            "max_delay",
            "avg_distance_per_order",
            "total_delay",
        ]

        # Import here to avoid circular imports
        try:
            from training.train import run_test_episode, MeituanDataConfig

            self.run_test_episode = run_test_episode
            self.MeituanDataConfig = MeituanDataConfig
        except ImportError as e:
            logger.error(f"Failed to import training modules: {e}")
            raise

    def get_current_model_info(self) -> tuple[Optional[str], Optional[str]]:
        """Get information about the current model in data/models/."""
        if not self.models_dir.exists():
            return None, None

        model_files = list(self.models_dir.glob("*.pt"))
        if not model_files:
            return None, None

        # Get the most recent model (should be only one based on your setup)
        latest_model = max(model_files, key=os.path.getmtime)
        model_name = latest_model.stem
        model_time = os.path.getmtime(latest_model)

        logger.info(f"Found current model: {model_name} (modified: {datetime.fromtimestamp(model_time)})")
        return model_name, str(latest_model)

    def find_existing_results(self) -> Optional[Path]:
        """Find the most recent benchmark results file."""
        # Look for different result file patterns
        patterns = ["benchmark_results_*.csv", "combined_with_baseline_*.csv", "fastest_aca_filtered_results_*.csv"]

        all_files = []
        for pattern in patterns:
            all_files.extend(self.results_dir.glob(pattern))

        if not all_files:
            return None

        # Return the most recent file
        latest_file = max(all_files, key=os.path.getmtime)
        logger.info(f"Found existing results: {latest_file.name}")
        return latest_file

    def check_if_model_benchmarked(self, results_file: Path, model_name: str) -> bool:
        """Check if the model is already benchmarked."""
        try:
            df = pd.read_csv(results_file)
            existing_methods = df["method"].unique()
            logger.info(f"Existing methods in results: {list(existing_methods)}")

            if model_name in existing_methods:
                logger.info(f"Model '{model_name}' already benchmarked")
                return True
            else:
                logger.info(f"Model '{model_name}' not found in existing results")
                return False

        except Exception as e:
            logger.error(f"Error reading results file: {e}")
            return False

    def benchmark_new_model(self, model_name: str) -> str:
        """Benchmark the new model and return the output file path. Runs 1 episode per dataset."""
        logger.info(f"Starting benchmark for new model: {model_name}")
        logger.info(f"Benchmarking {model_name} on {len(self.districts) * len(self.days)} datasets...")

        results = []
        total_datasets = len(self.districts) * len(self.days)
        completed = 0

        for district in self.districts:
            for day in self.days:
                completed += 1
                logger.info(f"Progress: {completed}/{total_datasets} - District {district}, Day {day}")

                # Configure data for this dataset
                meituan_config = self.MeituanDataConfig(
                    district_id=district,
                    day=day,
                    use_restaurant_positions=True,  # Use real restaurant positions
                    use_vehicle_count=True,  # Use real vehicle counts
                    use_vehicle_positions=True,  # Use real vehicle positions
                    use_service_area=True,  # Use real geographic boundaries
                    use_deadlines=True,  # Use real order deadlines
                    order_generation_mode="replay",  # Use REAL order data instead of artificial patterns
                    temporal_pattern=None,  # Not needed for replay mode
                    simulation_start_hour=10,  # Start at 10 AM (adjust as needed)
                    simulation_duration_hours=12,  # Run for 12 hours (adjust as needed)
                )

                # Run single episode for this dataset
                seed = (district * 1000) + (int(day[-2:]) * 10000)

                try:
                    stats = self.run_test_episode(
                        solver_name="rl_aca",  # Always use rl_aca for new models
                        meituan_config=meituan_config,
                        seed=seed,
                        reposition_idle_vehicles=False,
                        visualize=False,
                        warmup_duration=0,
                        save_results_to_disk=False,  # Disable disk saving to avoid to_dict error
                        aca_buffer=17,
                        exploration_rate=0,  # For RL-ACA, set exploration rate to 0 for evaluation
                    )

                    # Store results
                    result = {
                        "district": district,
                        "day": day,
                        "method": model_name,
                    }

                    for kpi in self.kpis:
                        result[kpi] = stats.get(kpi, 0)

                    results.append(result)

                except Exception as e:
                    logger.error(f"Error in District {district}, Day {day}: {e}")
                    continue

        # Save new results
        new_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_results_file = self.results_dir / f"new_model_benchmark_{model_name}_{timestamp}.csv"
        new_df.to_csv(new_results_file, index=False)

        logger.info(f"Completed benchmarking for {model_name}: {len(results)} results")
        logger.info(f"Saved new model results to: {new_results_file}")

        return str(new_results_file)

    def combine_with_existing_results(
        self, new_results_file: str, existing_results_file: Optional[Path], model_name: str
    ) -> str:
        """Combine new model results with existing results."""
        new_df = pd.read_csv(new_results_file)

        if existing_results_file and existing_results_file.exists():
            existing_df = pd.read_csv(existing_results_file)

            # Remove any existing data for this model (in case of replacement)
            existing_df = existing_df[existing_df["method"] != model_name]

            # Combine datasets
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            logger.info(f"Combined {len(existing_df)} existing + {len(new_df)} new = {len(combined_df)} total records")
        else:
            combined_df = new_df
            logger.info(f"No existing results to combine with, using {len(new_df)} new records")

        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = self.results_dir / f"benchmark_results_{timestamp}.csv"
        combined_df.to_csv(combined_file, index=False)

        logger.info(f"Saved combined results to: {combined_file}")
        return str(combined_file)


class BenchmarkingRunner:
    """
    Main class to orchestrate the complete benchmarking analysis pipeline.
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.benchmarking_dir = self.base_dir / "benchmarking"
        self.results_dir = self.base_dir / "data" / "simulation_results"

        # Initialize incremental benchmarker
        self.incremental_benchmarker = IncrementalModelBenchmarker(self.results_dir)

        # Define the pipeline steps (original steps)
        self.pipeline_steps = [
            {
                "name": "Data Filtering",
                "script": "filter_districts_negative_ontime.py",
                "description": "Remove districts with negative fastest_aca performance",
                "required": True,
                "output_pattern": "fastest_aca_filtered_results_*.csv",
            },
            {
                "name": "Statistical Analysis",
                "script": "benchmarking_pipeline.py",
                "description": "Generate performance reports and statistical analysis",
                "required": True,
                "output_pattern": "performance_report_*.md",
            },
            {
                "name": "Advanced Visualizations",
                "script": "advanced_comparative_visualizations.py",
                "description": "Create comprehensive visualization suite",
                "required": True,
                "output_pattern": "advanced_visualizations/",
            },
        ]

    def handle_incremental_model_benchmarking(
        self, start_fresh: bool = False, custom_model_name: Optional[str] = None
    ) -> bool:
        """Handle the incremental model benchmarking step."""
        logger.info("\n🤖 STEP 0: INCREMENTAL MODEL BENCHMARKING")
        logger.info("Detecting and benchmarking new RL models from data/models/")

        # 1. Get current model info
        model_name, model_path = self.incremental_benchmarker.get_current_model_info()
        if not model_name:
            logger.warning("No model found in data/models/ directory")
            logger.info("Skipping incremental benchmarking - proceeding with existing results")
            return True

        # Use custom name if provided
        if custom_model_name:
            model_name = custom_model_name
            logger.info(f"Using custom model name: {model_name}")

        # 2. Check existing results
        existing_results_file = None if start_fresh else self.incremental_benchmarker.find_existing_results()

        # 3. Check if model needs benchmarking
        if not start_fresh and existing_results_file:
            if self.incremental_benchmarker.check_if_model_benchmarked(existing_results_file, model_name):
                logger.info(f"✅ Model '{model_name}' already benchmarked - skipping")
                return True

        # 4. Run benchmarking for new model
        logger.info(f"🚀 Benchmarking new model: {model_name}")
        try:
            new_results_file = self.incremental_benchmarker.benchmark_new_model(model_name)

            # 5. Combine with existing results
            combined_file = self.incremental_benchmarker.combine_with_existing_results(
                new_results_file, existing_results_file, model_name
            )

            logger.info(f"✅ Successfully benchmarked and integrated model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to benchmark model {model_name}: {e}")
            return False

    def check_prerequisites(self):
        """Check if all required files and data exist."""
        logger.info("Checking prerequisites...")

        # Check if benchmarking directory exists
        if not self.benchmarking_dir.exists():
            raise FileNotFoundError(f"Benchmarking directory not found: {self.benchmarking_dir}")

        # Check if required scripts exist
        missing_scripts = []
        for step in self.pipeline_steps:
            script_path = self.benchmarking_dir / step["script"]
            if not script_path.exists():
                missing_scripts.append(step["script"])

        if missing_scripts:
            raise FileNotFoundError(f"Missing required scripts: {missing_scripts}")

        # Check if benchmark data exists (more flexible now with incremental benchmarking)
        data_files = list(self.results_dir.glob("*results_*.csv"))
        if not data_files:
            logger.warning("No existing benchmark data found. Will rely on incremental benchmarking.")
            return True  # Changed to True since we can create data with incremental benchmarking

        logger.info("✅ All prerequisites satisfied")
        return True

    def run_script(self, script_name: str, step_name: str) -> bool:
        """Run a single script and return success status."""
        script_path = self.benchmarking_dir / script_name

        logger.info(f"🚀 Starting: {step_name}")
        logger.info(f"   Script: {script_name}")

        start_time = time.time()

        try:
            # Run the script using Python with UTF-8 encoding for both input and output
            env = os.environ.copy()
            env.update({"PYTHONIOENCODING": "utf-8", "PYTHONLEGACYWINDOWSSTDIO": "0"})  # Force UTF-8 on Windows

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env,
                encoding="utf-8",  # Explicitly set encoding
                errors="replace",  # Replace problematic characters instead of crashing
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"✅ {step_name} completed successfully in {duration:.1f}s")

                # Print key output lines (last few lines usually contain summary)
                if result.stdout:
                    output_lines = result.stdout.strip().split("\n")
                    summary_lines = [line for line in output_lines[-10:] if line.strip()]
                    if summary_lines:
                        logger.info("   Key outputs:")
                        for line in summary_lines[-3:]:  # Last 3 meaningful lines
                            logger.info(f"   {line.strip()}")

                return True
            else:
                logger.error(f"❌ {step_name} failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"   Error: {result.stderr.strip()}")
                if result.stdout:
                    logger.error(f"   Output: {result.stdout.strip()}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ {step_name} timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"❌ {step_name} failed with exception: {e}")
            return False

    def check_step_output(self, step: dict) -> bool:
        """Check if a pipeline step produced expected output."""
        output_pattern = step["output_pattern"]

        if output_pattern.endswith("/"):
            # Directory check
            output_dir = self.results_dir / output_pattern.rstrip("/")
            return output_dir.exists() and any(output_dir.iterdir())
        else:
            # File pattern check
            files = list(self.results_dir.glob(output_pattern))
            return len(files) > 0

    def run_full_pipeline(
        self,
        skip_existing: bool = False,
        benchmark_new_model: bool = True,
        start_fresh: bool = False,
        custom_model_name: Optional[str] = None,
    ):
        """Run the complete benchmarking analysis pipeline."""

        logger.info("🔬 STARTING FULL BENCHMARKING ANALYSIS PIPELINE")
        logger.info("=" * 60)

        start_time = time.time()
        completed_steps = 0
        failed_steps = []

        # Step 0: Incremental Model Benchmarking (NEW)
        if benchmark_new_model:
            success = self.handle_incremental_model_benchmarking(start_fresh, custom_model_name)
            if not success:
                logger.error("❌ Incremental model benchmarking failed")
                failed_steps.append("Incremental Model Benchmarking")
            else:
                completed_steps += 1

        # Original pipeline steps
        for i, step in enumerate(self.pipeline_steps, 1):
            step_name = step["name"]
            script = step["script"]

            logger.info(f"\n📋 STEP {i}/{len(self.pipeline_steps)}: {step_name}")
            logger.info(f"   {step['description']}")

            # Check if output already exists and skip if requested
            if skip_existing and self.check_step_output(step):
                logger.info(f"⏭️  Skipping {step_name} - output already exists")
                completed_steps += 1
                continue

            # Run the step
            success = self.run_script(script, step_name)

            if success:
                # Verify output was created
                if self.check_step_output(step):
                    completed_steps += 1
                    logger.info(f"✅ {step_name} - Output verified")
                else:
                    logger.warning(f"⚠️  {step_name} - Script succeeded but expected output not found")
                    failed_steps.append(step_name)
            else:
                failed_steps.append(step_name)
                if step["required"]:
                    logger.error(f"❌ Pipeline halted due to failed required step: {step_name}")
                    break

        # Final summary
        total_time = time.time() - start_time
        total_possible_steps = len(self.pipeline_steps) + (1 if benchmark_new_model else 0)

        logger.info("\n" + "=" * 60)
        logger.info("🏁 BENCHMARKING ANALYSIS PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total runtime: {total_time:.1f} seconds")
        logger.info(f"✅ Completed steps: {completed_steps}/{total_possible_steps}")

        if failed_steps:
            logger.info(f"❌ Failed steps: {', '.join(failed_steps)}")

        # Output locations
        logger.info(f"\n📁 OUTPUT LOCATIONS:")
        logger.info(f"   📊 Benchmark Data: {self.results_dir}/benchmark_results_*.csv")
        logger.info(f"   📊 Filtered Data: {self.results_dir}/fastest_aca_filtered_results_*.csv")
        logger.info(f"   📋 Analysis Report: {self.results_dir}/performance_report_*.md")
        logger.info(f"   📈 Visualizations: {self.results_dir}/advanced_visualizations/")

        return len(failed_steps) == 0

    def print_file_status(self):
        """Print status of files in the benchmarking directory."""
        logger.info("\n📁 BENCHMARKING FOLDER FILE STATUS")
        logger.info("=" * 50)

        required_files = {
            "filter_districts_negative_ontime.py": "✅ REQUIRED - Filter problematic districts",
            "benchmarking_pipeline.py": "✅ REQUIRED - Statistical analysis",
            "advanced_comparative_visualizations.py": "✅ REQUIRED - Generate visualizations",
            "main_benchmarking.py": "✅ REQUIRED - This main runner script (with incremental support)",
        }

        optional_files = {
            "algorithm_benchmarking.py": "⚪ OPTIONAL - Original benchmarking (superseded by incremental)",
            "enhanced_pipeline.py": "⚪ OPTIONAL - Original enhanced pipeline (superseded)",
            "advanced_visualizations.py": "⚪ OPTIONAL - Original visualizations (superseded)",
            "filter_benchmark_data.py": "⚪ OPTIONAL - Comprehensive filtering (superseded)",
        }

        # Check which files exist
        existing_files = [f.name for f in self.benchmarking_dir.glob("*.py")]

        logger.info("Required files:")
        for filename, description in required_files.items():
            status = "✅ EXISTS" if filename in existing_files else "❌ MISSING"
            logger.info(f"  {status} - {filename}")
            logger.info(f"           {description}")

        logger.info("\nOptional files:")
        for filename, description in optional_files.items():
            status = "📁 EXISTS" if filename in existing_files else "⚪ NOT FOUND"
            logger.info(f"  {status} - {filename}")
            logger.info(f"           {description}")

        # Show any extra files
        all_known = set(required_files.keys()) | set(optional_files.keys())
        extra_files = [f for f in existing_files if f not in all_known]

        if extra_files:
            logger.info(f"\nOther files found:")
            for filename in extra_files:
                logger.info(f"  📄 {filename}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Run complete benchmarking analysis pipeline with incremental model support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_benchmarking.py                                      # DEFAULT: Detect and benchmark new model, then run analysis
  python main_benchmarking.py --model-name phase2_final           # Benchmark with custom model name
  python main_benchmarking.py --start-fresh                       # Start fresh with new model
  python main_benchmarking.py --skip-model-benchmark              # Skip model benchmarking, only run analysis
  python main_benchmarking.py --skip-existing                     # Skip analysis steps with existing output
  python main_benchmarking.py --check-files                       # Just check file status
        """,
    )

    parser.add_argument(
        "--model-name", type=str, help="Custom name for the current model (default: auto-detect from filename)"
    )
    parser.add_argument(
        "--start-fresh", action="store_true", help="Start benchmarking from scratch, ignore existing results"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip analysis steps that already have output files"
    )
    parser.add_argument(
        "--skip-model-benchmark",
        action="store_true",
        help="Skip model benchmarking, only run analysis on existing data",
    )
    parser.add_argument("--check-files", action="store_true", help="Only check file status, don't run pipeline")

    args = parser.parse_args()

    # By default, we benchmark new model unless explicitly skipped
    benchmark_new_model = not args.skip_model_benchmark
    custom_model_name = args.model_name

    # Create runner instance
    runner = BenchmarkingRunner()

    try:
        if args.check_files:
            # Just check file status
            runner.print_file_status()
            return

        # Check prerequisites
        if not runner.check_prerequisites():
            logger.error("Prerequisites not met. Please ensure required scripts exist.")
            return

        # Run the pipeline
        success = runner.run_full_pipeline(
            skip_existing=args.skip_existing,
            benchmark_new_model=benchmark_new_model,
            start_fresh=args.start_fresh,
            custom_model_name=custom_model_name,
        )

        if success:
            logger.info("🎉 All analysis steps completed successfully!")
            logger.info("Check the output locations above for results.")
        else:
            logger.error("⚠️  Pipeline completed with some failures.")
            logger.info("Check the logs above for details.")

    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
