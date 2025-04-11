import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_meal_prep_times(df, output_dir):
    """
    Analyze and visualize meal preparation times from the Meituan dataset in the style of Figure 4.2.
    
    This function:
    1. Calculates prep time as (estimate_meal_prepare_time - platform_order_time)
    2. Filters to reasonable values (1-60 minutes)
    3. Creates a histogram styled like the distance distribution
    4. Adds a 'prep_time_minutes' column to the dataframe
    
    Args:
        df: DataFrame with order data
        output_dir: Directory to save visualizations
        
    Returns:
        DataFrame with added 'prep_time_minutes' column
    """
    print("Analyzing meal preparation times...")

    # Create output directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Check for required columns
    if 'platform_order_time' not in processed_df.columns or 'estimate_meal_prepare_time' not in processed_df.columns:
        print("Warning: Missing required columns for meal prep time calculation")
        return processed_df

    # Convert UNIX timestamps to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(processed_df['platform_order_time']):
        processed_df['platform_order_time'] = pd.to_datetime(processed_df['platform_order_time'], unit='s')
        # Convert Chinese time to UTC by adding 8 hours
        processed_df['platform_order_time'] = processed_df['platform_order_time'] + pd.Timedelta(hours=8)
    if not pd.api.types.is_datetime64_any_dtype(processed_df['estimate_meal_prepare_time']):
        processed_df['estimate_meal_prepare_time'] = pd.to_datetime(processed_df['estimate_meal_prepare_time'], unit='s')
        # Convert Chinese time to UTC by adding 8 hours
        processed_df['estimate_meal_prepare_time'] = processed_df['estimate_meal_prepare_time'] + pd.Timedelta(hours=8)

    # Calculate prep time (in minutes)
    valid_times = processed_df[processed_df['platform_order_time'].notna() & 
                              processed_df['estimate_meal_prepare_time'].notna()]
    
    if len(valid_times) == 0:
        print("Warning: No valid data for meal prep time calculation")
        return processed_df

    prep_times = (valid_times['estimate_meal_prepare_time'] - 
                 valid_times['platform_order_time']).dt.total_seconds() / 60

    # Add rounded prep time to the dataframe
    processed_df['prep_time_minutes'] = np.nan
    processed_df.loc[valid_times.index, 'prep_time_minutes'] = np.round(prep_times)

    # Filter to reasonable values (1-60 minutes)
    reasonable_mask = (processed_df['prep_time_minutes'] >= 1) & (processed_df['prep_time_minutes'] <= 60)
    unreasonable_count = len(processed_df[~reasonable_mask & processed_df['prep_time_minutes'].notna()])
    if unreasonable_count > 0:
        print(f"Found {unreasonable_count} orders with unreasonable prep times (< 1 min or > 60 min)")
        processed_df.loc[~reasonable_mask & processed_df['prep_time_minutes'].notna(), 'prep_time_minutes'] = np.nan

    # Create visualization styled like Figure 4.2
    valid_prep_times = processed_df['prep_time_minutes'].dropna()

    plt.figure(figsize=(10, 6))
    # Use the same blue color as in the distance histogram
    plt.hist(
        valid_prep_times,
        bins=30,
        alpha=0.5,
        color='blue',  # Exact same blue as in the distance histogram
        edgecolor='black'
    )
    # No KDE line, as requested
    
    plt.xlabel('Meal Preparation Time (minutes)')
    plt.ylabel('Number of Orders')
    plt.title('Distribution of Meal Preparation Times')
    plt.grid(True, alpha=0.3)

    # Add mean and median lines
    mean_val = valid_prep_times.mean()
    median_val = valid_prep_times.median()
    std_val = valid_prep_times.std()


    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f} min')
    plt.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.1f} min')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "meal_prep_time_distribution.pdf"), dpi=300, format='pdf')
    plt.close()

    # Print summary statistics
    print(f"Meal preparation time statistics:")
    print(f"  Valid data points: {len(valid_prep_times)}")
    print(f"  Mean prep time: {mean_val:.1f} minutes")
    print(f"  Median prep time: {median_val:.1f} minutes")
    print(f"  Standard deviation: {std_val:.1f} minutes")
    print(f"  Min prep time: {valid_prep_times.min():.1f} minutes")
    print(f"  Max prep time: {valid_prep_times.max():.1f} minutes")

    return processed_df


def plot_hourly_demand(df, output_dir):
    """
    Visualize the hourly demand distribution averaged over 8 days from the Meituan dataset
    and compute a normalized demand pattern for all 24 hours relative to the average.
    
    This function:
    1. Extracts order times and computes average hourly demand.
    2. Normalizes the demand relative to the average hourly rate to derive hourly rate multipliers.
    3. Creates a line plot with peak hours highlighted.
    4. Returns a dictionary mapping each hour to its rate multiplier.
    
    Args:
        df: DataFrame with order data
        output_dir: Directory to save visualizations
    
    Returns:
        dict: A dictionary mapping each hour (0-23) to its rate multiplier
    """
    print("Analyzing hourly demand distribution...")

    # Create output directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Convert UNIX timestamp to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(df['platform_order_time']):
        df['platform_order_time'] = pd.to_datetime(df['platform_order_time'], unit='s')
        # Convert Chinese time to UTC by adding 8 hours
        df['platform_order_time'] = df['platform_order_time'] + pd.Timedelta(hours=8)

    # Extract hour and compute average orders per hour over 8 days
    df['hour'] = df['platform_order_time'].dt.hour
    hourly_counts = df.groupby('hour').size() / 8  # Average over 8 days

    # Ensure all hours (0-23) are present, filling missing hours with 0
    hourly_counts = hourly_counts.reindex(range(24), fill_value=0)

    # Define the average hourly rate as the base rate
    average_rate = hourly_counts.mean()
    if average_rate == 0:
        average_rate = 1.0  # Avoid division by zero
    print(f"Average hourly orders (base rate): {average_rate:.1f} orders per hour")

    # Compute the rate multipliers relative to the average rate
    rate_multipliers = hourly_counts / average_rate

    # Create a dictionary mapping each hour to its rate multiplier
    demand_pattern = {hour: rate for hour, rate in rate_multipliers.items()}
    print("\nHourly demand pattern (rate multipliers relative to average):")
    for hour, rate in demand_pattern.items():
        print(f"Hour {hour:02d}: {rate:.2f}x")

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=hourly_counts.index, y=hourly_counts.values, marker='o', color='blue')
    plt.xlabel('Hour of Day (UTC)')
    plt.ylabel('Average Number of Orders')
    plt.title('Hourly Demand Distribution (Averaged Over 8 Days)')
    plt.grid(True, alpha=0.3)

    # Highlight peak hours (e.g., lunch 10-12, dinner 17-19)
    plt.axvspan(10, 12, color='yellow', alpha=0.2, label='Lunch Peak')
    plt.axvspan(17, 19, color='orange', alpha=0.2, label='Dinner Peak')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "hourly_demand_distribution.pdf"), dpi=300, format='pdf')
    plt.close()

    # Print peak hour insights
    peak_lunch = hourly_counts[10:13].mean()  # 10:00-12:00
    peak_dinner = hourly_counts[17:20].mean()  # 17:00-19:00
    off_peak = hourly_counts[[h for h in range(24) if h not in range(10, 13) and h not in range(17, 20)]].mean()
    print(f"\nPeak demand insights:")
    print(f"Average orders during lunch peak (10:00-12:00): {peak_lunch:.1f}")
    print(f"Average orders during dinner peak (17:00-19:00): {peak_dinner:.1f}")
    print(f"Average orders during off-peak hours: {off_peak:.1f}")
    print(f"Lunch peak increase over average: {peak_lunch / average_rate:.1f}x")
    print(f"Dinner peak increase over average: {peak_dinner / average_rate:.1f}x")
    
    # Print general hourly demand statistics
    print(f"\nHourly demand statistics:")
    print(f"  Minimum hourly orders: {hourly_counts.min():.1f} (Hour: {hourly_counts.idxmin()})")
    print(f"  Maximum hourly orders: {hourly_counts.max():.1f} (Hour: {hourly_counts.idxmax()})")
    print(f"  Average hourly orders: {hourly_counts.mean():.1f}")
    print(f"  Median hourly orders: {hourly_counts.median():.1f}")

    return demand_pattern























def plot_demand_by_district(df, output_dir):
    """
    Analyze hourly demand variations across business districts and create a summary table.
    
    This function:
    1. Extracts order times and computes average hourly demand by district
    2. Creates a table summarizing demand during peak hours (lunch: 10:00-12:00, dinner: 17:00-19:00)
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with order timestamps.
        output_dir (str): Directory to save the table.
    
    Returns:
        None
    """
    print("Analyzing demand variations by business district...")
    os.makedirs(output_dir, exist_ok=True)

    # Convert UNIX timestamp to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(df['platform_order_time']):
        df['platform_order_time'] = pd.to_datetime(df['platform_order_time'], unit='s')
        # Convert Chinese time to UTC by adding 8 hours
        df['platform_order_time'] = df['platform_order_time'] + pd.Timedelta(hours=8)
    
    # Extract hour from the timestamp
    df['hour'] = df['platform_order_time'].dt.hour
    
    # Calculate the percentage of orders between 7:00 and 20:00
    workday_hours = df[(df['hour'] >= 7) & (df['hour'] < 20)]
    workday_orders = len(workday_hours)
    total_orders = len(df)
    workday_percentage = (workday_orders / total_orders) * 100
    print(f"Percentage of orders between 7:00 and 20:00: {workday_percentage:.1f}%")
    
    # Group by da_id and hour to get order counts
    demand_by_district = df.groupby(['da_id', 'hour']).size().unstack(fill_value=0)
    
    # Average over the 8 days in the dataset
    demand_by_district = demand_by_district / 8
    
    # Calculate total orders per district
    total_orders_per_district = demand_by_district.sum(axis=1)
    
    # Normalize to get percentage of orders per hour for each district
    demand_normalized = demand_by_district.div(total_orders_per_district, axis=0) * 100
    
    # Calculate average orders during peak times for each district
    lunch_peak = demand_by_district.loc[:, 10:11].mean(axis=1)  # 10:00-12:00
    dinner_peak = demand_by_district.loc[:, 17:18].mean(axis=1)  # 17:00-19:00
    lunch_peak_percentage = (lunch_peak / total_orders_per_district) * 100
    dinner_peak_percentage = (dinner_peak / total_orders_per_district) * 100
    
    # Create a summary table
    summary_table = pd.DataFrame({
        'Total Orders': total_orders_per_district.astype(int),
        'Lunch Peak Orders (10:00-12:00)': lunch_peak.astype(int),
        'Lunch Peak Percentage (%)': lunch_peak_percentage.round(1),
        'Dinner Peak Orders (17:00-19:00)': dinner_peak.astype(int),
        'Dinner Peak Percentage (%)': dinner_peak_percentage.round(1)
    }).reset_index()
    print("\nDemand by Business District (Peak Hours):")
    print(summary_table)
    
    # Save the table as a CSV for inclusion in LaTeX
    table_path = os.path.join(output_dir, "peak_demand_by_district.csv")
    summary_table.to_csv(table_path, index=False)
    print(f"Saved peak demand table to {table_path}")


    

def main():
    """
    Main function to run the Meituan data analysis pipeline.
    Loads data, runs analysis functions, and saves results.
    """
    # Define input and output paths
    input_file = "data/meituan_data/all_waybill_info_meituan_0322.csv"
    output_dir = "data/meituan_data/abb"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run analysis functions
    analyze_meal_prep_times(df, output_dir)
    plot_hourly_demand(df, output_dir)
    plot_demand_by_district(df, output_dir)
    print("Analysis complete!")


# Run the main function when the script is executed
if __name__ == "__main__":
    main()