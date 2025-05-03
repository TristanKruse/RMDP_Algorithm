# 4_2_6and7_active_couriers_utilization_dispatch_only.py
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_and_clean_data(file_path, dataset_type="orders"):
    """
    Load the Meituan data and perform initial cleaning.
    Adapted from preprocessing_data.py.
    
    Args:
        file_path: Path to the data file
        dataset_type: Type of dataset ('orders' or 'dispatch_riders')
    """
    print(f"Loading {dataset_type} data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Original data shape: {df.shape}")
    
    # Convert timestamp columns to datetime
    if dataset_type == "orders":
        timestamp_cols = ['platform_order_time', 'grab_time', 'arrive_time']
    else:  # dispatch_riders
        timestamp_cols = ['dispatch_time']
    
    for col in timestamp_cols:
        if col in df.columns:
            # Convert UNIX timestamp to datetime and add 8 hours to convert from Chinese time to UTC
            df[col] = pd.to_datetime(df[col], unit='s') + pd.Timedelta(hours=8)
    
    # Handle missing values: Convert '0' timestamps to NaT
    for col in timestamp_cols:
        if col in df.columns:
            df.loc[df[col] == pd.Timestamp(0) + pd.Timedelta(hours=8), col] = pd.NaT
    
    print(f"Data after cleaning: {df.shape}")
    return df

def compute_active_couriers_dispatch_only(dispatch_df, order_df, output_dir="output"):
    """
    Compute the number of distinct active couriers using only dispatch data within daily timeframes,
    and calculate system utilization metrics (overall, daily, district-level).
    
    Args:
        dispatch_df: DataFrame with dispatch data (dispatch_rider_meituan.csv)
        order_df: DataFrame with order data (all_waybill_info_meituan_0322.csv) for district mapping and metrics
        output_dir: Directory to save results
    
    Returns:
        DataFrame with active couriers per day
        DataFrame with daily averages
        DataFrame with district-level averages
    """
    print("Computing distinct active couriers and system utilization (dispatch-only)...")
    
    # Filter dispatch data with valid dispatch_time
    valid_dispatch = dispatch_df[dispatch_df['dispatch_time'].notna()].copy()
    print(f"Number of dispatch records with valid dispatch time: {len(valid_dispatch)}")
    
    # Analyze dispatch data coverage
    dispatch_start = valid_dispatch['dispatch_time'].min()
    dispatch_end = valid_dispatch['dispatch_time'].max()
    dispatch_hours = (dispatch_end - dispatch_start).total_seconds() / 3600
    print(f"\nDispatch Data Coverage:")
    print(f"  Dispatch data spans from {dispatch_start} to {dispatch_end}")
    print(f"  Total hours of dispatch data: {dispatch_hours:.2f}")
    
    # Add day column for grouping
    valid_dispatch['day'] = valid_dispatch['dispatch_time'].dt.date
    
    # Count dispatch records by day and hour for diagnostic purposes
    valid_dispatch['hour'] = valid_dispatch['dispatch_time'].dt.hour
    dispatch_counts = valid_dispatch.groupby(['day', 'hour']).size().reset_index(name='record_count')
    print("\nDispatch Records by Day and Hour:")
    for day in dispatch_counts['day'].unique():
        print(f"\nDay: {day}")
        day_data = dispatch_counts[dispatch_counts['day'] == day]
        for _, row in day_data.iterrows():
            print(f"  Hour {row['hour']:02d}:00: {row['record_count']} records")
    
    # Group dispatch data by day to find daily timeframes
    daily_timeframes = valid_dispatch.groupby('day').agg({
        'dispatch_time': ['min', 'max']
    }).reset_index()
    daily_timeframes.columns = ['day', 'start_time', 'end_time']
    
    print("\nDaily Timeframes (Dispatch-Only):")
    for _, row in daily_timeframes.iterrows():
        print(f"  Day: {row['day']}, Start: {row['start_time']}, End: {row['end_time']}")
    
    # Compute relative metrics (couriers per restaurant, couriers per order)
    # 1. Count distinct restaurants (poi_id) from order data
    valid_orders = order_df[order_df['platform_order_time'].notna()].copy()
    distinct_restaurants = len(valid_orders['poi_id'].unique())
    print(f"\nNumber of distinct restaurants (poi_id): {distinct_restaurants}")
    
    # 2. Compute total orders and mean interarrival time from order data
    total_orders = len(valid_orders)
    total_orders_per_day = total_orders / 8  # Average over 8 days
    print(f"Total orders over 8 days: {total_orders}")
    print(f"Average orders per day: {total_orders_per_day:.1f}")
    
    # Compute mean interarrival time
    valid_orders = valid_orders.sort_values('platform_order_time')
    valid_orders['interarrival_time'] = valid_orders['platform_order_time'].diff().dt.total_seconds()
    mean_interarrival_time = valid_orders['interarrival_time'].mean()  # In seconds
    print(f"Mean interarrival time of orders (seconds): {mean_interarrival_time:.2f}")
    mean_interarrival_time_minutes = mean_interarrival_time / 60
    print(f"Mean interarrival time of orders (minutes): {mean_interarrival_time_minutes:.2f}")
    
    # Compute orders per district
    valid_orders['district'] = valid_orders['da_id']
    orders_per_district = valid_orders.groupby('district').size().reset_index(name='total_orders')
    orders_per_district['daily_orders'] = orders_per_district['total_orders'] / 8  # Average over 8 days
    
    # Map couriers to districts using order data (most recent order's da_id)
    valid_orders_for_mapping = order_df[order_df['grab_time'].notna() & order_df['arrive_time'].notna() & order_df['da_id'].notna()].copy()
    all_orders = valid_orders_for_mapping.sort_values('grab_time')
    courier_district_map = all_orders.groupby('courier_id').last()[['da_id']].reset_index()
    courier_district_map = courier_district_map.rename(columns={'da_id': 'district'})
    
    # Compute distinct active couriers and utilization within each daily timeframe
    active_data = []
    district_data = []
    daily_averages = []
    utilization_data = []
    
    for _, timeframe in daily_timeframes.iterrows():
        day = timeframe['day']
        start_time = timeframe['start_time']
        end_time = timeframe['end_time']
        
        # Filter dispatch records within the timeframe (active couriers)
        daily_dispatch = valid_dispatch[
            (valid_dispatch['day'] == day) &
            (valid_dispatch['dispatch_time'] >= start_time) &
            (valid_dispatch['dispatch_time'] <= end_time)
        ]
        
        # Count distinct active couriers for the entire timeframe
        active_couriers = set(daily_dispatch['courier_id'].unique())
        total_active = len(active_couriers)
        
        # Filter orders to find busy couriers within the timeframe
        busy_orders = order_df[
            (order_df['grab_time'].notna()) &
            (order_df['arrive_time'].notna()) &
            (order_df['grab_time'] <= end_time) &
            (order_df['arrive_time'] >= start_time)
        ]
        # Add 'day' column to busy_orders
        busy_orders.loc[:, 'day'] = busy_orders['grab_time'].dt.date
        busy_orders = busy_orders[busy_orders['day'] == day]
        busy_couriers = set(busy_orders['courier_id'].unique())
        
        # Intersect busy couriers with active couriers to ensure v_t <= 1.0
        busy_couriers = busy_couriers.intersection(active_couriers)
        total_busy = len(busy_couriers)
        
        # Compute utilization (v_t)
        utilization = total_busy / total_active if total_active > 0 else 0
        utilization_data.append({
            'day': str(day),  # Convert to string for consistency
            'active_couriers': total_active,
            'busy_couriers': total_busy,
            'utilization': utilization
        })
        
        active_data.append({
            'day': day,
            'total_active': total_active
        })
        
        daily_averages.append({
            'day': day,
            'distinct_active_couriers': total_active
        })
        print(f"\nDay: {day}, Distinct Active Couriers: {total_active}, Utilization: {utilization:.3f}")
        
        # District-level breakdown
        active_with_district = daily_dispatch.merge(courier_district_map, on='courier_id', how='inner')
        combined_active = active_with_district[['courier_id', 'district']].drop_duplicates(subset='courier_id')
        
        busy_with_district = busy_orders.merge(courier_district_map, on='courier_id', how='inner')
        combined_busy = busy_with_district[['courier_id', 'district']].drop_duplicates(subset='courier_id')
        
        for district in combined_active['district'].unique():
            district_active = combined_active[combined_active['district'] == district]
            total_active_district = len(set(district_active['courier_id'].unique()))
            
            district_busy = combined_busy[combined_busy['district'] == district]
            total_busy_district = len(set(district_busy['courier_id'].unique()))
            
            district_utilization = total_busy_district / total_active_district if total_active_district > 0 else 0
            
            district_data.append({
                'day': day,
                'district': district,
                'total_active': total_active_district,
                'total_busy': total_busy_district,
                'utilization': district_utilization
            })
    
    # Create DataFrames
    active_df = pd.DataFrame(active_data)
    daily_averages_df = pd.DataFrame(daily_averages)
    utilization_df = pd.DataFrame(utilization_data)
    district_df = pd.DataFrame(district_data)
    
    # Compute overall average utilization
    overall_utilization = utilization_df['utilization'].mean()
    print(f"\nOverall Average System Utilization (Dispatch-Only): {overall_utilization:.3f}")
    
    # Compute overall average (mean of daily distinct counts)
    overall_avg = daily_averages_df['distinct_active_couriers'].mean()
    print(f"Overall Average Distinct Active Couriers (Dispatch-Only): {overall_avg:.1f}")
    
    # Compute relative metrics (overall)
    # Couriers per restaurant
    total_orders = len(valid_orders)
    total_orders_per_day = total_orders / 8  # Average over 8 days
    distinct_restaurants = len(valid_orders['poi_id'].unique())
    couriers_per_restaurant = overall_avg / distinct_restaurants
    print(f"Couriers per restaurant (during 5-minute windows): {couriers_per_restaurant:.2f}")
    
    # Couriers per order (using total orders per day)
    couriers_per_order = overall_avg / total_orders_per_day
    print(f"Couriers per order (based on average daily orders): {couriers_per_order:.2f}")
    
    # Create a LaTeX table for daily comparison (active couriers, utilization)
    comparison_df = utilization_df[['day', 'active_couriers', 'utilization']]
    comparison_df.columns = ['Day', 'Active Couriers', 'Utilization']
    comparison_df['Utilization'] = comparison_df['Utilization'].round(3)
    latex_table = comparison_df.to_latex(index=False, float_format="%.3f", 
                                         caption="Daily Comparison of Active Couriers and System Utilization (Dispatch-Only, 11:25--11:30)", 
                                         label="tab:daily_utilization_comparison")
    latex_path = os.path.join(output_dir, "daily_utilization_comparison.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved daily comparison LaTeX table to: {os.path.abspath(latex_path)}")
    
    # Generate bar chart with dual axes (Active Couriers and Utilization only)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Using visualizations directory: {os.path.abspath(viz_dir)}")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # X-axis: Days
    days = comparison_df['Day']
    x = range(len(days))
    
    # Left Y-axis: Active Couriers
    ax1.bar(x, comparison_df['Active Couriers'], width=0.4, label='Active Couriers', color='purple')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Active Couriers', color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.set_xticks(x)
    ax1.set_xticklabels(days, rotation=45)
    
    # Right Y-axis: Utilization
    ax2 = ax1.twinx()
    ax2.plot(x, comparison_df['Utilization'], color='blue', marker='o', label='Utilization')
    ax2.set_ylabel('Utilization', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 1)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.title('Daily Active Couriers and Utilization (Dispatch-Only, 11:25--11:30)')
    plt.tight_layout()
    save_path = os.path.join(viz_dir, "daily_utilization_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved bar chart to: {os.path.abspath(save_path)}")
    plt.close()
    
    # Compute district-level utilization averages
    district_overall_avg = district_df.groupby('district')[['total_active', 'utilization']].mean().reset_index()
    district_overall_avg = district_overall_avg.merge(orders_per_district, on='district')
    district_overall_avg['district'] = district_overall_avg['district'].apply(lambda x: f"DA {int(x)}")
    print("\nDistrict-Level Average Utilization (Dispatch-Only):")
    for _, row in district_overall_avg.iterrows():
        print(f"  {row['district']}: Active Couriers: {row['total_active']:.1f}, Utilization: {row['utilization']:.3f}")
    
    # Save district-level utilization as a LaTeX table
    district_latex = district_overall_avg[['district', 'total_active', 'utilization']]
    district_latex.columns = ['District', 'Average Distinct Active Couriers', 'Utilization']
    district_latex['Average Distinct Active Couriers'] = district_latex['Average Distinct Active Couriers'].round(1)
    district_latex['Utilization'] = district_latex['Utilization'].round(3)
    latex_table = district_latex.to_latex(index=False, float_format="%.3f", 
                                          caption="Average Distinct Active Couriers and Utilization per District (Dispatch-Only, 11:25--11:30 Daily)", 
                                          label="tab:utilization_per_district")
    latex_path = os.path.join(output_dir, "utilization_per_district.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved district-level utilization LaTeX table to: {os.path.abspath(latex_path)}")
    
    return active_df, daily_averages_df, district_overall_avg

def main():
    # Print current working directory for debugging
    print(f"Current working directory: {os.path.abspath(os.getcwd())}")
    
    # Set input and output paths
    data_dir = "data/meituan_data"
    order_file = os.path.join(data_dir, "all_waybill_info_meituan_0322.csv")
    dispatch_file = os.path.join(data_dir, "dispatch_rider_meituan.csv")
    output_dir = os.path.join(data_dir, "abb")
    
    # Load and preprocess data
    order_df = load_and_clean_data(order_file, dataset_type="orders")
    dispatch_df = load_and_clean_data(dispatch_file, dataset_type="dispatch_riders")
    
    # Compute active couriers and utilization using dispatch data only
    active_df, daily_averages_df, district_avg = compute_active_couriers_dispatch_only(
        dispatch_df, order_df, output_dir=output_dir
    )
    
    print("Dispatch-only active couriers and utilization analysis complete!")

if __name__ == "__main__":
    main()