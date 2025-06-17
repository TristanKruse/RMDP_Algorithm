#!/usr/bin/env python3
"""
Meituan Benchmark Data Extraction Script

This script processes the raw Meituan order data to extract ground truth performance metrics
for each district-day combination. These benchmarks will be used to compare algorithmic
performance against real-world Meituan delivery performance.

Extracts:
- On-time delivery rate
- Average delay for late orders
- Total orders processed
- Orders delivered successfully
- Average distance per order
- Maximum delay experienced
- Total delay across all orders

Author: Generated for delivery algorithm benchmarking
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_and_clean_order_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean Meituan order data.

    Args:
        file_path: Path to the order CSV file

    Returns:
        Cleaned DataFrame with proper timestamp conversions
    """
    logger.info(f"Loading order data from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Order data file not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} order records")

    # Convert timestamp columns to datetime - these are already in proper datetime string format
    timestamp_cols = [
        "platform_order_time",  # When order was created
        "estimate_meal_prepare_time",  # Estimated meal preparation completion
        "order_push_time",  # When order was pushed to dispatch system
        "dispatch_time",  # When order was dispatched to courier
        "grab_time",  # When courier accepted/grabbed the order
        "fetch_time",  # When courier picked up the order from restaurant
        "arrive_time",  # When courier delivered the order to customer
        "estimate_arrived_time",  # Estimated arrival time
    ]

    for col in timestamp_cols:
        if col in df.columns:
            # These are already datetime strings, just convert directly
            df[col] = pd.to_datetime(df[col], errors="coerce")
            # No need to handle UNIX timestamps - the data is already in proper format

    logger.info(f"Processed timestamps for {len(df)} orders")
    return df


def calculate_geographic_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate approximate distance between two geographic points using Haversine formula.

    Args:
        lat1, lng1: Latitude and longitude of first point
        lat2, lng2: Latitude and longitude of second point

    Returns:
        Distance in kilometers
    """
    # Convert coordinates from Meituan format (scaled by 1,000,000) to decimal degrees
    if abs(lat1) > 1000:  # Check if scaling is needed
        lat1, lng1, lat2, lng2 = lat1 / 1000000, lng1 / 1000000, lat2 / 1000000, lng2 / 1000000

    # Haversine formula
    R = 6371  # Earth's radius in km

    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)

    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance


def extract_day_from_order_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract day information from order data and add as a column.

    Args:
        df: Order DataFrame

    Returns:
        DataFrame with added 'day' column
    """
    # Debug: Check what columns are available
    logger.info(f"Available columns in order data: {list(df.columns)}")

    # Check if there's already an 'order_date' column (this is the best option)
    if "order_date" in df.columns:
        # Use the existing order_date column
        df["day"] = df["order_date"].astype(str)
        logger.info(f"Using existing 'order_date' column")
        logger.info(f"Sample order_date values: {df['order_date'].head(3).tolist()}")

    # Use platform_order_time as the primary timestamp for day determination
    elif "platform_order_time" in df.columns:
        # Check for valid timestamps
        valid_timestamps = df[df["platform_order_time"].notna()]
        if len(valid_timestamps) > 0:
            # Extract date in YYYYMMDD format from platform_order_time
            df["day"] = df["platform_order_time"].dt.strftime("%Y%m%d").astype(str)
            logger.info(f"Sample timestamps: {df['platform_order_time'].dropna().head(3).tolist()}")
            logger.info(f"Sample extracted days: {df['day'].dropna().head(3).tolist()}")
        else:
            logger.warning("No valid platform_order_time timestamps found")
            # Fallback to 'dt' column if available
            if "dt" in df.columns:
                df["day"] = df["dt"].astype(str)
                logger.info(f"Fallback: Using 'dt' column, sample values: {df['dt'].head(3).tolist()}")
            else:
                raise ValueError("No suitable date column found and no valid timestamps")
    elif "dt" in df.columns:
        # Fallback to 'dt' column if available
        df["day"] = df["dt"].astype(str)
        logger.info(f"Using 'dt' column, sample values: {df['dt'].head(3).tolist()}")
    else:
        raise ValueError("No suitable date column found in order data")

    # Filter out rows where day couldn't be determined (but don't filter NaT timestamps)
    df = df[df["day"].notna() & (df["day"] != "NaT")]
    logger.info(f"Extracted day information for {len(df)} orders")
    logger.info(f"Unique days found: {sorted(df['day'].unique())}")

    return df


def calculate_delivery_performance_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate delivery performance metrics from order data.

    Args:
        df: Order DataFrame for a specific district-day combination

    Returns:
        Dictionary containing calculated metrics
    """
    metrics = {}

    # Total orders processed
    metrics["total_orders"] = len(df)

    # Orders with valid delivery times (successfully delivered)
    delivered_orders = df[df["arrive_time"].notna() & df["grab_time"].notna()]
    metrics["orders_delivered"] = len(delivered_orders)

    # Orders that were never delivered
    metrics["undelivered_orders"] = metrics["total_orders"] - metrics["orders_delivered"]

    if len(delivered_orders) == 0:
        # No delivered orders - set defaults
        metrics.update(
            {
                "on_time_delivery_rate": -100.0,  # Indicates complete failure
                "avg_delay_late_orders": 0.0,
                "max_delay": 0.0,
                "avg_distance_per_order": 0.0,
                "total_delay": 0.0,
                "late_orders_count": metrics["total_orders"],
            }
        )
        return metrics

    # Calculate delivery times and delays
    delivered_orders = delivered_orders.copy()

    # Estimate promised delivery time (use estimated arrival time if available, otherwise add standard window)
    if "estimate_arrived_time" in delivered_orders.columns and delivered_orders["estimate_arrived_time"].notna().any():
        # Use estimated arrival time as the promised time
        delivered_orders["promised_time"] = delivered_orders["estimate_arrived_time"]
        logger.info("Using estimate_arrived_time as promised delivery time")
    else:
        # Fallback: Add 40-minute delivery window to order creation time
        delivery_window_minutes = 40
        delivered_orders["promised_time"] = delivered_orders["platform_order_time"] + pd.Timedelta(
            minutes=delivery_window_minutes
        )
        logger.info(f"Using {delivery_window_minutes}-minute window from order time as promised delivery time")

    # Calculate actual delay (in minutes) - adjusted for simulation comparison
    delivered_orders["raw_delay_minutes"] = (
        delivered_orders["arrive_time"] - delivered_orders["promised_time"]
    ).dt.total_seconds() / 60

    # Convert to simulation-style delays: negative delays (early) become 0, positive delays remain
    delivered_orders["delay_minutes"] = delivered_orders["raw_delay_minutes"].clip(lower=0)

    # Debug logging for first few orders
    if len(delivered_orders) > 0:
        sample_raw_delays = delivered_orders["raw_delay_minutes"].head(3)
        sample_sim_delays = delivered_orders["delay_minutes"].head(3)
        logger.info(f"Sample raw delays (minutes): {sample_raw_delays.tolist()}")
        logger.info(f"Sample simulation-style delays (minutes): {sample_sim_delays.tolist()}")

        early_count = (delivered_orders["raw_delay_minutes"] < 0).sum()
        logger.info(
            f"Early deliveries converted to 0 delay: {early_count}/{len(delivered_orders)} ({early_count/len(delivered_orders)*100:.1f}%)"
        )

    # Identify late orders (positive delay)
    late_orders = delivered_orders[delivered_orders["delay_minutes"] > 0]
    on_time_orders = delivered_orders[delivered_orders["delay_minutes"] <= 0]

    metrics["late_orders_count"] = len(late_orders)

    # On-time delivery rate calculation (corrected)
    # Simply: on-time orders / total delivered orders * 100
    metrics["on_time_delivery_rate"] = len(on_time_orders) / max(1, metrics["orders_delivered"]) * 100

    # Average delay for late orders only
    if len(late_orders) > 0:
        metrics["avg_delay_late_orders"] = late_orders["delay_minutes"].mean()
        metrics["max_delay"] = late_orders["delay_minutes"].max()
    else:
        metrics["avg_delay_late_orders"] = 0.0
        metrics["max_delay"] = 0.0

    # Total delay across all orders (including on-time orders with negative delay)
    metrics["total_delay"] = delivered_orders["delay_minutes"].sum()

    # Calculate average distance per order
    distance_orders = delivered_orders[
        delivered_orders["sender_lat"].notna()
        & delivered_orders["sender_lng"].notna()
        & delivered_orders["recipient_lat"].notna()
        & delivered_orders["recipient_lng"].notna()
    ]

    if len(distance_orders) > 0:
        distances = []
        for _, order in distance_orders.iterrows():
            dist = calculate_geographic_distance(
                order["sender_lat"], order["sender_lng"], order["recipient_lat"], order["recipient_lng"]
            )
            distances.append(dist)

        metrics["avg_distance_per_order"] = np.mean(distances)
    else:
        metrics["avg_distance_per_order"] = 0.0

    # Idle rate is not calculable from order data alone (would need vehicle/courier data)
    metrics["active_period_idle_rate"] = 0  # Placeholder

    return metrics


def process_district_day_data(district_dir: str, district_id: int, day: str) -> Optional[Dict]:
    """
    Process order data for a specific district and day.

    Args:
        district_dir: Directory containing district data files
        district_id: District ID (1-22)
        day: Day in YYYYMMDD format

    Returns:
        Dictionary with metrics or None if data not available
    """
    order_file = os.path.join(district_dir, f"district_{district_id}_orders.csv")

    if not os.path.exists(order_file):
        logger.warning(f"Order file not found: {order_file}")
        return None

    try:
        # Load order data
        df = load_and_clean_order_data(order_file)

        if df.empty:
            logger.warning(f"No data in {order_file}")
            return None

        # Extract day information and filter for the specific day
        df = extract_day_from_order_data(df)

        # Debug: Show what days we found vs what we're looking for
        found_days = sorted(df["day"].unique())
        logger.info(f"Looking for day: {day}, Found days in data: {found_days}")

        day_data = df[df["day"] == day]

        if day_data.empty:
            logger.warning(f"No data for day {day} in district {district_id}. Available days: {found_days}")
            return None

        # Calculate metrics
        metrics = calculate_delivery_performance_metrics(day_data)
        metrics["district"] = district_id
        metrics["day"] = int(day)  # Convert to integer for consistency

        logger.info(
            f"District {district_id}, Day {day}: {metrics['total_orders']} orders, "
            f"{metrics['orders_delivered']} delivered, "
            f"on-time rate: {metrics['on_time_delivery_rate']:.1f}%"
        )

        return metrics

    except Exception as e:
        logger.error(f"Error processing district {district_id}, day {day}: {e}")
        return None


def extract_meituan_benchmarks(
    data_dir: str = "data/meituan_data/processed/daily_orders", output_dir: str = "data/meituan_benchmark"
) -> str:
    """
    Extract Meituan benchmark performance data from raw order files.

    Args:
        data_dir: Directory containing daily order data
        output_dir: Directory to save benchmark results

    Returns:
        Path to the created benchmark CSV file
    """
    logger.info("Starting Meituan benchmark data extraction")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define districts and days from your benchmark
    districts = list(range(1, 23))  # Districts 1 to 22
    days = [f"202210{day:02d}" for day in range(17, 25)]  # October 17 to October 24, 2022

    logger.info(f"Processing {len(districts)} districts across {len(days)} days")

    all_benchmarks = []

    # Process each day
    for day in days:
        day_dir = os.path.join(data_dir, day)

        if not os.path.exists(day_dir):
            logger.warning(f"Day directory not found: {day_dir}")
            continue

        logger.info(f"Processing day: {day}")

        # Process each district for this day
        for district_id in districts:
            metrics = process_district_day_data(day_dir, district_id, day)

            if metrics is not None:
                all_benchmarks.append(metrics)

    # Create DataFrame and save results
    if all_benchmarks:
        benchmark_df = pd.DataFrame(all_benchmarks)

        # Sort by district and day for easier reading
        benchmark_df = benchmark_df.sort_values(["district", "day"])

        # Save benchmark data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"meituan_ground_truth_performance_{timestamp}.csv")
        benchmark_df.to_csv(output_file, index=False)

        logger.info(f"Saved Meituan benchmark data to: {output_file}")
        logger.info(f"Extracted benchmarks for {len(benchmark_df)} district-day combinations")

        # Print summary statistics
        logger.info("Summary statistics:")
        logger.info(f"  Average on-time rate: {benchmark_df['on_time_delivery_rate'].mean():.2f}%")
        logger.info(f"  Average total delay: {benchmark_df['total_delay'].mean():.2f} minutes")
        logger.info(f"  Average orders per district-day: {benchmark_df['total_orders'].mean():.1f}")
        logger.info(
            f"  Average delivery success rate: {(benchmark_df['orders_delivered']/benchmark_df['total_orders']).mean()*100:.2f}%"
        )

        return output_file
    else:
        raise ValueError("No benchmark data could be extracted. Check data directory and file structure.")


def main():
    """Main function to run the benchmark extraction."""
    try:
        # Set paths - adjust these according to your directory structure
        data_dir = "data/meituan_data/processed/daily_orders"  # Where your daily order files are stored
        output_dir = "data/meituan_benchmark"  # Where to save benchmark results

        # Extract benchmarks
        output_file = extract_meituan_benchmarks(data_dir, output_dir)

        logger.info("Meituan benchmark extraction completed successfully!")
        logger.info(f"Benchmark file created: {output_file}")

    except Exception as e:
        logger.error(f"Benchmark extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()
