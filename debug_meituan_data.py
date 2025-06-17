#!/usr/bin/env python3
"""
Debug script to examine Meituan timestamp data and understand delivery calculation issues.
"""

import pandas as pd
import numpy as np
import os


def debug_timestamp_data():
    """Debug timestamp columns to understand why no deliveries are calculated."""

    # Let's examine one file to understand timestamp issues
    file_path = "data/meituan_data/processed/daily_orders/20221017/district_1_orders.csv"

    print(f"Examining file: {file_path}")
    print("=" * 60)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load the data
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")

    # Focus on timestamp columns needed for delivery calculations
    timestamp_cols = [
        "grab_time",  # When courier grabbed the order
        "arrive_time",  # When courier delivered
        "estimate_arrived_time",  # Promised delivery time
        "platform_order_time",  # Order creation time
        "dispatch_time",  # When dispatched
        "fetch_time",  # When picked up from restaurant
    ]

    print(f"\n=== TIMESTAMP ANALYSIS ===")
    for col in timestamp_cols:
        if col in df.columns:
            print(f"\n--- {col} ---")
            print(f"Data type: {df[col].dtype}")

            # Count non-null values
            non_null = df[col].notna().sum()
            print(f"Non-null values: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")

            if non_null > 0:
                # Show sample values
                sample_values = df[col].dropna().head(5).tolist()
                print(f"Sample values: {sample_values}")

                # Try to convert to see if they're timestamps
                try:
                    # Method 1: If they look like UNIX timestamps
                    if df[col].dtype in ["int64", "float64"]:
                        converted = pd.to_datetime(df[col].dropna(), unit="s", errors="coerce")
                        valid_converted = converted.dropna()
                        if len(valid_converted) > 0:
                            print(f"As timestamps: {valid_converted.head(3).tolist()}")
                        else:
                            print("Could not convert to valid timestamps")

                    # Method 2: Direct datetime conversion
                    else:
                        converted = pd.to_datetime(df[col], errors="coerce")
                        valid_converted = converted.dropna()
                        if len(valid_converted) > 0:
                            print(f"As timestamps: {valid_converted.head(3).tolist()}")
                        else:
                            print("Could not convert to valid timestamps")

                except Exception as e:
                    print(f"Conversion failed: {e}")
            else:
                print("No non-null values found!")

    # Check what percentage of orders have both grab_time and arrive_time
    if "grab_time" in df.columns and "arrive_time" in df.columns:
        both_valid = df[df["grab_time"].notna() & df["arrive_time"].notna()]
        print(f"\n=== DELIVERY COMPLETION ANALYSIS ===")
        print(
            f"Orders with both grab_time AND arrive_time: {len(both_valid)}/{len(df)} ({len(both_valid)/len(df)*100:.1f}%)"
        )

        if len(both_valid) > 0:
            print("Sample complete delivery:")
            sample = both_valid.iloc[0]
            print(f"  grab_time: {sample['grab_time']}")
            print(f"  arrive_time: {sample['arrive_time']}")

    print(f"\n" + "=" * 60)
    print("DIAGNOSIS: If most timestamp columns are empty or invalid,")
    print("that explains why all orders show 0 deliveries.")


if __name__ == "__main__":
    debug_timestamp_data()
