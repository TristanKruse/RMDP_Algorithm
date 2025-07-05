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
import numpy as np
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
            "postponement_rate",
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

    def benchmark_all_methods(self, model_name: str = None) -> str:
        """Benchmark all methods: fastest_aca, aca_17, and rl_aca."""

        # Define all methods to benchmark with exact parameters
        methods = [
            {
                "name": "fastest_aca",
                "solver": "aca",
                "needs_model": False,
                "buffer": 999,  # Very high buffer to allow maximum flexibility
                "max_postponements": 0,  # No postponements allowed
                "max_postpone_time": 0,  # No postponement time allowed
                "postponement_method": "heuristic",
            },
            {
                "name": "aca_17",
                "solver": "aca",
                "needs_model": False,
                "buffer": 17,  # 17-minute postponement buffer
                "max_postponements": 0,  # Controlled by buffer, not count
                "max_postpone_time": 0,  # Controlled by buffer
                "postponement_method": "heuristic",
            },
            {
                "name": "rl_aca",
                "solver": "rl_aca",
                "needs_model": True,
                "buffer": 17,  # Same buffer as standard ACA
                "max_postponements": 0,  # Controlled by RL decisions
                "max_postpone_time": 0,  # Controlled by RL decisions
                "postponement_method": "rl-aca",
            },
        ]

        # Keep RL method name as "rl_aca" regardless of model file name
        # (model_name is only used for file path, not method name)

        logger.info(f"Starting complete benchmark for all methods")
        logger.info(f"Methods: {[m['name'] for m in methods]}")
        logger.info(f"Datasets per method: {len(self.districts) * len(self.days)}")

        all_results = []
        total_combinations = len(methods) * len(self.districts) * len(self.days)
        completed = 0

        for method in methods:
            logger.info(f"\n🚀 Starting benchmarking for method: {method['name']}")

            for district in self.districts:
                for day in self.days:
                    completed += 1
                    logger.info(
                        f"Progress: {completed}/{total_combinations} - {method['name']} District {district}, Day {day}"
                    )

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
                        # Configure method-specific parameters
                        run_params = {
                            "solver_name": method["solver"],
                            "meituan_config": meituan_config,
                            "seed": seed,
                            "reposition_idle_vehicles": False,
                            "visualize": False,
                            "warmup_duration": 0,
                            "save_results_to_disk": False,
                            "aca_buffer": method.get("buffer", 17),
                            "exploration_rate": 0,
                            "training_mode": False,
                            "save_rl_model": False,
                        }

                        # Add model path for RL method
                        if method["needs_model"]:
                            # For RL method, we need either a provided model_name or detect from models dir
                            if model_name:
                                model_path = str(self.models_dir / f"{model_name}.pt")
                            else:
                                # Try to find the most recent model
                                current_model_name, current_model_path = self.get_current_model_info()
                                if not current_model_name:
                                    logger.warning(f"Skipping {method['name']} - no RL model found")
                                    continue
                                model_path = current_model_path
                            run_params["rl_model_path"] = model_path

                        stats = self.run_test_episode(**run_params)

                        # Calculate KPIs using EXACT same logic as train_rl.py compare_models
                        total_orders = stats.get("total_orders", 1)
                        orders_delivered = stats.get("orders_delivered", 0)
                        late_orders = stats.get("late_orders", set())
                        delay_values = stats.get("delay_values", [])
                        total_distance = stats.get("total_distance", 0)
                        postponed_orders = stats.get("postponed_orders", set())

                        # 1. total_delay: same as compare_models
                        total_delay = sum(delay_values) if delay_values else 0

                        # 2. on_time_rate: EXACT same logic as compare_models
                        total_orders_calc = max(1, orders_delivered)  # compare_models uses orders_delivered
                        late_count = len(late_orders)
                        on_time_rate = ((total_orders_calc - late_count) / total_orders_calc) * 100

                        # 3. avg_delay_late_orders: average of delay_values (same as compare_models)
                        avg_delay_late = sum(delay_values) / len(delay_values) if delay_values else 0

                        # 4. postponement_rate: EXACT same logic as compare_models
                        postponement_rate = len(postponed_orders) / max(1, total_orders) * 100

                        # 5. avg_distance_per_order: Use idle-rate method for realistic estimates
                        from training.core.stats import calculate_idle_rate_distance

                        simulation_duration = 720  # 12 hours in minutes
                        total_productive_distance = calculate_idle_rate_distance(stats, simulation_duration)
                        avg_distance_per_order = total_productive_distance / max(1, orders_delivered)

                        # 6. vehicle_utilization: same as compare_models
                        idle_rates = stats.get("active_period_idle_rates_by_vehicle", {})
                        if idle_rates:
                            vehicle_utilizations = [1 - np.mean(rates) for rates in idle_rates.values() if rates]
                            active_period_idle_rate = np.mean(vehicle_utilizations) if vehicle_utilizations else 0
                        else:
                            active_period_idle_rate = 0

                        # Store results with calculated KPIs (using compare_models logic)
                        result = {
                            "district": district,
                            "day": day,
                            "method": method["name"],  # Use actual method name
                            "on_time_delivery_rate": on_time_rate,
                            "avg_delay_late_orders": avg_delay_late,
                            "avg_distance_per_order": avg_distance_per_order,
                            "total_delay": total_delay,  # Calculated from delay_values
                            "max_delay": stats.get("max_delay", 0),
                            "active_period_idle_rate": active_period_idle_rate,  # Calculated from vehicle idle rates
                            "postponement_rate": postponement_rate,  # Added postponement KPI
                        }

                        # Log detailed results for debugging
                        logger.info(f"   📊 Results for {method['name']} - District {district}, Day {day}:")
                        logger.info(
                            f"      Total/Delivered: {total_orders}/{orders_delivered} ({orders_delivered/total_orders*100:.1f}%)"
                        )
                        logger.info(
                            f"      On-time rate: {on_time_rate:.1f}% ({total_orders_calc-late_count}/{total_orders_calc} on-time)"
                        )
                        logger.info(
                            f"      Postponement rate: {postponement_rate:.1f}% ({len(postponed_orders)}/{total_orders} postponed)"
                        )
                        logger.info(f"      Avg delay (late): {avg_delay_late:.1f} min")
                        logger.info(f"      Avg distance per delivered order: {avg_distance_per_order:.1f} km")
                        logger.info(f"      Total delay (calculated): {total_delay:.1f} min (from delay_values)")
                        logger.info(
                            f"      Active idle rate (calculated): {active_period_idle_rate:.3f} (from {len(idle_rates)} vehicles)"
                        )

                        all_results.append(result)

                    except Exception as e:
                        logger.error(f"Error in {method['name']} District {district}, Day {day}: {e}")
                        continue

        # Save complete benchmark results with both timestamped and fixed names
        all_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Timestamped version for history
        timestamped_file = self.results_dir / f"benchmark_results_{timestamp}.csv"
        all_df.to_csv(timestamped_file, index=False)
        
        # Fixed name version for current use
        fixed_file = self.results_dir / "benchmark_results.csv"
        all_df.to_csv(fixed_file, index=False)

        logger.info(f"Completed benchmarking for all methods: {len(all_results)} total results")
        logger.info(f"Saved timestamped results to: {timestamped_file}")
        logger.info(f"Saved current results to: {fixed_file}")

        # Print summary by method
        if all_results:
            summary_df = all_df.groupby("method").size().reset_index(name="count")
            logger.info(f"\n📊 Results summary by method:")
            for _, row in summary_df.iterrows():
                logger.info(f"   {row['method']}: {row['count']} datasets")

        return str(timestamped_file)

    def combine_with_existing_results(
        self, new_results_file: str, existing_results_file: Optional[Path], model_name: str
    ) -> str:
        """Combine new model results with existing results using fixed file names."""
        new_df = pd.read_csv(new_results_file)

        # Use fixed file name for benchmark results
        benchmark_results_file = self.results_dir / "benchmark_results.csv"

        if benchmark_results_file.exists():
            existing_df = pd.read_csv(benchmark_results_file)

            # Remove any existing data for this model (in case of replacement)
            existing_df = existing_df[existing_df["method"] != model_name]

            # Combine datasets
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            logger.info(
                f"Updated benchmark_results.csv: {len(existing_df)} existing + {len(new_df)} new = {len(combined_df)} total records"
            )
        else:
            combined_df = new_df
            logger.info(f"Created new benchmark_results.csv with {len(new_df)} records")

        # Save to fixed file name
        combined_df.to_csv(benchmark_results_file, index=False)
        logger.info(f"Saved combined results to: benchmark_results.csv")

        # Clean up temporary file
        import os

        os.remove(new_results_file)

        return str(benchmark_results_file)


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

        # 4. Run complete benchmarking for all methods
        logger.info(f"🚀 Running complete benchmarking for all methods")
        try:
            new_results_file = self.incremental_benchmarker.benchmark_all_methods(model_name)

            # 5. Combine with existing results
            combined_file = self.incremental_benchmarker.combine_with_existing_results(
                new_results_file, existing_results_file, model_name
            )

            logger.info(f"✅ Successfully benchmarked and integrated model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to benchmark model {model_name}: {e}")
            return False

    def run_complete_benchmarking(self, custom_model_name: Optional[str] = None) -> bool:
        """Run complete benchmarking for all three methods from scratch."""
        logger.info("\n🚀 COMPLETE BENCHMARKING MODE")
        logger.info("Running ALL methods from scratch: fastest_aca, aca_17, rl_aca")
        logger.info("=" * 60)

        # Use the modified benchmark_all_methods that now handles all three methods
        try:
            results_file = self.incremental_benchmarker.benchmark_all_methods(custom_model_name)

            if results_file and os.path.exists(results_file):
                logger.info(f"✅ Complete benchmarking successful!")
                logger.info(f"Results saved to: {results_file}")
                return True
            else:
                logger.error("❌ Complete benchmarking failed - no results file created")
                return False

        except Exception as e:
            logger.error(f"❌ Complete benchmarking failed: {e}")
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
        complete_benchmark: bool = False,
    ):
        """Run the complete benchmarking analysis pipeline."""

        logger.info("🔬 STARTING FULL BENCHMARKING ANALYSIS PIPELINE")
        logger.info("=" * 60)

        start_time = time.time()
        completed_steps = 0
        failed_steps = []

        # Step 0: Benchmarking (either complete or incremental)
        if complete_benchmark:
            # Complete benchmarking mode - run all methods from scratch
            success = self.run_complete_benchmarking(custom_model_name)
            step_name = "Complete Benchmarking"
            if not success:
                logger.error("❌ Complete benchmarking failed")
                failed_steps.append(step_name)
            else:
                completed_steps += 1
        elif benchmark_new_model:
            # Incremental mode - only benchmark new RL models
            success = self.handle_incremental_model_benchmarking(start_fresh, custom_model_name)
            step_name = "Incremental Model Benchmarking"
            if not success:
                logger.error("❌ Incremental model benchmarking failed")
                failed_steps.append(step_name)
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
  python benchmarking/main_benchmarking.py                                     # DEFAULT: Detect and benchmark new model, then run analysis
  python benchmarking/main_benchmarking.py --complete-benchmark                # NEW: Run ALL methods from scratch (fastest_aca, aca_17, rl_aca)
  python benchmarking/main_benchmarking.py --complete-benchmark --model-name my_model  # Complete benchmark with custom RL model name
  python benchmarking/main_benchmarking.py --model-name phase2_final           # Benchmark with custom model name (incremental)
  python benchmarking/main_benchmarking.py --start-fresh                       # Start fresh with new model (incremental)
  python benchmarking/main_benchmarking.py --skip-model-benchmark              # Skip model benchmarking, only run analysis
  python benchmarking/main_benchmarking.py --skip-existing                     # Skip analysis steps with existing output
  python benchmarking/main_benchmarking.py --check-files                       # Just check file status
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
    parser.add_argument(
        "--complete-benchmark",
        action="store_true",
        help="Run complete benchmarking for all methods from scratch (fastest_aca, aca_17, rl_aca)",
    )

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
            complete_benchmark=args.complete_benchmark,
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
