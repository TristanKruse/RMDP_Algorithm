#!/usr/bin/env python3
"""
Main Benchmarking Analysis Runner

This script orchestrates the complete benchmarking analysis pipeline:
1. Data filtering (remove problematic districts)
2. Statistical analysis and reporting
3. Advanced visualization generation

Usage:
    python benchmarking/run_full_analysis.py [options]

Required Files in benchmarking/ folder:
- algorithm_benchmarking.py         # Generate raw benchmark data
- filter_districts_negative_ontime.py  # Filter problematic districts
- benchmarking_pipeline.py          # Statistical analysis
- advanced_comparative_visualizations.py  # Generate visualizations
- run_full_analysis.py (this file)  # Main runner

Optional/Utility Files:
- enhanced_pipeline.py              # Original enhanced pipeline (integrated into benchmarking_pipeline.py)
- advanced_visualizations.py        # Original visualizations (updated as advanced_comparative_visualizations.py)
- filter_benchmark_data.py         # Comprehensive filtering (superseded by district filtering)
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class BenchmarkingRunner:
    """
    Main class to orchestrate the complete benchmarking analysis pipeline.
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.benchmarking_dir = self.base_dir / "benchmarking"
        self.results_dir = self.base_dir / "data" / "simulation_results"

        # Define the pipeline steps
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

        # Check if benchmark data exists
        data_files = list(self.results_dir.glob("combined_with_baseline_*.csv"))
        if not data_files:
            logger.warning("No combined baseline data found. You may need to run algorithm_benchmarking.py first.")
            return False

        logger.info("✅ All prerequisites satisfied")
        return True

    def run_script(self, script_name: str, step_name: str) -> bool:
        """Run a single script and return success status."""
        script_path = self.benchmarking_dir / script_name

        logger.info(f"🚀 Starting: {step_name}")
        logger.info(f"   Script: {script_name}")

        start_time = time.time()

        try:
            # Run the script using Python
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
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

    def run_full_pipeline(self, skip_existing: bool = False):
        """Run the complete benchmarking analysis pipeline."""

        logger.info("🔬 STARTING FULL BENCHMARKING ANALYSIS PIPELINE")
        logger.info("=" * 60)

        start_time = time.time()
        completed_steps = 0
        failed_steps = []

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

        logger.info("\n" + "=" * 60)
        logger.info("🏁 BENCHMARKING ANALYSIS PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total runtime: {total_time:.1f} seconds")
        logger.info(f"✅ Completed steps: {completed_steps}/{len(self.pipeline_steps)}")

        if failed_steps:
            logger.info(f"❌ Failed steps: {', '.join(failed_steps)}")

        # Output locations
        logger.info(f"\n📁 OUTPUT LOCATIONS:")
        logger.info(f"   📊 Filtered Data: {self.results_dir}/fastest_aca_filtered_results_*.csv")
        logger.info(f"   📋 Analysis Report: {self.results_dir}/performance_report_*.md")
        logger.info(f"   📈 Visualizations: {self.results_dir}/advanced_visualizations/")

        return len(failed_steps) == 0

    def print_file_status(self):
        """Print status of files in the benchmarking directory."""
        logger.info("\n📁 BENCHMARKING FOLDER FILE STATUS")
        logger.info("=" * 50)

        required_files = {
            "algorithm_benchmarking.py": "✅ REQUIRED - Generate raw benchmark data",
            "filter_districts_negative_ontime.py": "✅ REQUIRED - Filter problematic districts",
            "benchmarking_pipeline.py": "✅ REQUIRED - Statistical analysis",
            "advanced_comparative_visualizations.py": "✅ REQUIRED - Generate visualizations",
            "run_full_analysis.py": "✅ REQUIRED - This main runner script",
        }

        optional_files = {
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
        description="Run complete benchmarking analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarking/run_full_analysis.py                    # Run full pipeline
  python benchmarking/run_full_analysis.py --skip-existing    # Skip steps with existing output
  python benchmarking/run_full_analysis.py --check-files      # Just check file status
        """,
    )

    parser.add_argument("--skip-existing", action="store_true", help="Skip steps that already have output files")

    parser.add_argument("--check-files", action="store_true", help="Only check file status, don't run pipeline")

    args = parser.parse_args()

    # Create runner instance
    runner = BenchmarkingRunner()

    try:
        if args.check_files:
            # Just check file status
            runner.print_file_status()
            return

        # Check prerequisites
        if not runner.check_prerequisites():
            logger.error("Prerequisites not met. Please ensure benchmark data exists.")
            logger.info("You may need to run: python algorithm_benchmarking.py")
            return

        # Run the pipeline
        success = runner.run_full_pipeline(skip_existing=args.skip_existing)

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
