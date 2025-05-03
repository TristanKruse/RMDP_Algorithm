import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend for saving plots
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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
        timestamp_cols = ['platform_order_time', 'estimate_meal_prepare_time', 
                         'order_push_time', 'dispatch_time', 'grab_time', 
                         'fetch_time', 'arrive_time', 'estimate_arrived_time']
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

def preprocess_coordinates(df, scale_factor=1000000.0):
    """
    Normalize Meituan coordinates by dividing by scale_factor (default: 1,000,000).
    Adapted from preprocessing_data.py.
    """
    coordinate_columns = [
        'sender_lat', 'sender_lng', 'recipient_lat', 'recipient_lng',
        'grab_lat', 'grab_lng', 'rider_lat', 'rider_lng'
    ]
    
    processed_df = df.copy()
    
    for col in coordinate_columns:
        if col in processed_df.columns:
            # Check if scaling is needed (values > 1000)
            if processed_df[col].abs().max() > 1000:
                print(f"Scaling down {col} from Meituan format")
                processed_df[col] = processed_df[col] / scale_factor
    
    return processed_df

def compute_utilization_per_district(order_df, dispatch_df, time_resolution='10min', activity_window='1h', output_dir="output"):
    """
    Compute the system utilization (v_t) per district (using da_id) over time.
    
    Args:
        order_df: DataFrame with order data (all_waybill_info_meituan_0322.csv)
        dispatch_df: DataFrame with dispatch data (dispatch_rider_meituan.csv)
        time_resolution: Time resolution for computation (default: '10min')
        activity_window: Window to consider a courier active after last activity (default: '1h')
        output_dir: Directory to save results
    
    Returns:
        DataFrame with utilization per district statistics
    """
    print(f"Computing utilization per district (da_id) at {time_resolution} resolution...")
    
    # Create output directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    try:
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Created visualizations directory at: {os.path.abspath(viz_dir)}")
    except Exception as e:
        print(f"Error creating visualizations directory: {e}")
        raise
    
    # Create a list of all time points at the specified resolution
    start_time = min(
        order_df['grab_time'].min(),
        dispatch_df['dispatch_time'].min()
    ).floor(time_resolution)
    end_time = max(
        order_df['arrive_time'].max(),
        dispatch_df['dispatch_time'].max()
    ).ceil(time_resolution)
    all_times = pd.date_range(start=start_time, end=end_time, freq=time_resolution)
    print(f"Processing {len(all_times)} time points from {start_time} to {end_time}")
    
    # Filter orders with valid grab_time, arrive_time, and da_id
    valid_orders = order_df[order_df['grab_time'].notna() & order_df['arrive_time'].notna() & order_df['da_id'].notna()].copy()
    print(f"Number of orders with valid grab, arrive times, and da_id: {len(valid_orders)}")
    
    # Filter dispatch data with valid dispatch_time
    valid_dispatch = dispatch_df[dispatch_df['dispatch_time'].notna()].copy()
    print(f"Number of dispatch records with valid dispatch time: {len(valid_dispatch)}")
    
    # Map each courier to a district based on their most recent order's da_id (at any time)
    all_orders = valid_orders.sort_values('grab_time')
    courier_district_map = all_orders.groupby('courier_id').last()[['da_id']].reset_index()
    courier_district_map = courier_district_map.rename(columns={'da_id': 'district'})
    
    # Initialize utilization per district DataFrame
    utilization_data = []
    activity_window = pd.Timedelta(activity_window)
    
    for time in all_times:
        # Active couriers (from dispatch data)
        dispatch_window_start = time - activity_window
        dispatch_active = valid_dispatch[
            (valid_dispatch['dispatch_time'] >= dispatch_window_start) &
            (valid_dispatch['dispatch_time'] <= time)
        ]
        active_couriers = set(dispatch_active['courier_id'].unique())
        
        # Identify couriers who were recently active (within the activity window)
        recent_mask = (valid_orders['arrive_time'] > time - activity_window) & (valid_orders['arrive_time'] <= time)
        recent_orders = valid_orders[recent_mask]
        recent_orders = recent_orders.sort_values('arrive_time')
        recent_orders = recent_orders.groupby('courier_id').last().reset_index()
        recent_with_district = recent_orders.merge(courier_district_map, on='courier_id', how='inner')
        
        # Combine active and recently active couriers
        active_with_district = dispatch_active.merge(courier_district_map, on='courier_id', how='inner')
        combined_active = pd.concat([
            active_with_district[['courier_id', 'district']],
            recent_with_district[['courier_id', 'district']]
        ]).drop_duplicates(subset='courier_id')
        
        # Identify busy couriers (from orders at time t)
        busy_mask = (valid_orders['grab_time'] <= time) & (valid_orders['arrive_time'] > time)
        busy_orders = valid_orders[busy_mask]
        busy_orders = busy_orders.sort_values('grab_time')
        busy_orders = busy_orders.groupby('courier_id').last().reset_index()
        
        # Compute utilization for each district with active couriers
        for district in combined_active['district'].unique():
            # Active couriers in this district
            district_active = combined_active[combined_active['district'] == district]
            total_active = len(set(district_active['courier_id'].unique()))
            
            # Busy couriers in this district
            district_busy_orders = busy_orders[busy_orders['da_id'] == district]
            district_busy_couriers = set(district_busy_orders['courier_id'].unique())
            # Ensure busy couriers are a subset of active couriers in this district
            busy_couriers = len(district_busy_couriers.intersection(set(district_active['courier_id'])))
            
            # Compute utilization for this district
            v_t = busy_couriers / total_active if total_active > 0 else 0
            
            # Find the hour for this time point
            hour = time.hour
            
            utilization_data.append({
                'time': time,
                'hour': hour,
                'district': district,
                'busy_couriers': busy_couriers,
                'total_active': total_active,
                'v_t': v_t
            })
    
    # Create DataFrame
    utilization_df = pd.DataFrame(utilization_data)
    print(f"Utilization data collected for {len(utilization_df)} district-time points")
    
    # Compute max, min, and average utilization per district, including the hour of max and min
    if not utilization_df.empty:
        # Find the indices of max and min v_t for each district
        max_indices = utilization_df.groupby('district')['v_t'].idxmax()
        min_indices = utilization_df.groupby('district')['v_t'].idxmin()
        
        # Extract the hours for max and min
        max_hours = utilization_df.loc[max_indices, ['district', 'hour']].set_index('district')['hour']
        min_hours = utilization_df.loc[min_indices, ['district', 'hour']].set_index('district')['hour']
        
        # Aggregate statistics
        district_stats = utilization_df.groupby('district').agg({
            'v_t': ['max', 'min', 'mean']
        }).reset_index()
        
        district_stats.columns = ['District', 'Max Utilization', 'Min Utilization', 'Avg Utilization']
        district_stats['District'] = district_stats['District'].apply(lambda x: f"DA {int(x)}")
        district_stats['Max Utilization'] = district_stats['Max Utilization'].round(2)
        district_stats['Min Utilization'] = district_stats['Min Utilization'].round(2)
        district_stats['Avg Utilization'] = district_stats['Avg Utilization'].round(2)
        
        # Add the hours of max and min
        district_stats['Hour of Max'] = district_stats['District'].map(lambda x: int(max_hours.loc[int(x.split()[1])] if int(x.split()[1]) in max_hours.index else np.nan))
        district_stats['Hour of Min'] = district_stats['District'].map(lambda x: int(min_hours.loc[int(x.split()[1])] if int(x.split()[1]) in min_hours.index else np.nan))
        
        # Filter out districts with NaN values (if any)
        district_stats = district_stats.dropna(subset=['Hour of Max', 'Hour of Min'])
    else:
        print("No utilization data available for any district.")
        district_stats = pd.DataFrame(columns=['District', 'Max Utilization', 'Min Utilization', 'Avg Utilization', 'Hour of Max', 'Hour of Min'])
    
    print("\nUtilization per District Statistics:")
    print(district_stats)
    
    # Save the table as a LaTeX file
    latex_table = district_stats.to_latex(index=False, float_format="%.2f", caption="Utilization Statistics per District (DA) with Hours of Max and Min Utilization", label="tab:utilization_per_district")
    latex_path = os.path.join(output_dir, "utilization_per_district_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to: {os.path.abspath(latex_path)}")
    
    return district_stats, utilization_df

def compute_active_couriers_per_district(order_df, dispatch_df, time_resolution='10min', activity_window='1h', output_dir="output"):
    """
    Compute the number of active couriers per district (using da_id) over time.
    
    Args:
        order_df: DataFrame with order data (all_waybill_info_meituan_0322.csv)
        dispatch_df: DataFrame with dispatch data (dispatch_rider_meituan.csv)
        time_resolution: Time resolution for computation (default: '10min')
        activity_window: Window to consider a courier active after last activity (default: '1h')
        output_dir: Directory to save results
    
    Returns:
        DataFrame with active couriers per district statistics
    """
    print(f"Computing active couriers per district (da_id) at {time_resolution} resolution...")
    
    # Create a list of all time points at the specified resolution
    start_time = min(
        order_df['grab_time'].min(),
        dispatch_df['dispatch_time'].min()
    ).floor(time_resolution)
    end_time = max(
        order_df['arrive_time'].max(),
        dispatch_df['dispatch_time'].max()
    ).ceil(time_resolution)
    all_times = pd.date_range(start=start_time, end=end_time, freq=time_resolution)
    print(f"Processing {len(all_times)} time points from {start_time} to {end_time}")
    
    # Filter orders with valid grab_time, arrive_time, and da_id
    valid_orders = order_df[order_df['grab_time'].notna() & order_df['arrive_time'].notna() & order_df['da_id'].notna()].copy()
    print(f"Number of orders with valid grab, arrive times, and da_id: {len(valid_orders)}")
    
    # Filter dispatch data with valid dispatch_time
    valid_dispatch = dispatch_df[dispatch_df['dispatch_time'].notna()].copy()
    print(f"Number of dispatch records with valid dispatch time: {len(valid_dispatch)}")
    
    # Map each courier to a district based on their most recent order's da_id (at any time)
    all_orders = valid_orders.sort_values('grab_time')
    courier_district_map = all_orders.groupby('courier_id').last()[['da_id']].reset_index()
    courier_district_map = courier_district_map.rename(columns={'da_id': 'district'})
    
    # Initialize active couriers per district DataFrame
    active_data = []
    activity_window = pd.Timedelta(activity_window)
    
    for time in all_times:
        # Active couriers (from dispatch data)
        dispatch_window_start = time - activity_window
        dispatch_active = valid_dispatch[
            (valid_dispatch['dispatch_time'] >= dispatch_window_start) &
            (valid_dispatch['dispatch_time'] <= time)
        ]
        
        # Identify couriers who were recently active (within the activity window)
        recent_mask = (valid_orders['arrive_time'] > time - activity_window) & (valid_orders['arrive_time'] <= time)
        recent_orders = valid_orders[recent_mask]
        recent_orders = recent_orders.sort_values('arrive_time')
        recent_orders = recent_orders.groupby('courier_id').last().reset_index()
        recent_with_district = recent_orders.merge(courier_district_map, on='courier_id', how='inner')
        
        # Combine active and recently active couriers
        active_with_district = dispatch_active.merge(courier_district_map, on='courier_id', how='inner')
        combined_active = pd.concat([
            active_with_district[['courier_id', 'district']],
            recent_with_district[['courier_id', 'district']]
        ]).drop_duplicates(subset='courier_id')
        
        # Compute active couriers for each district
        for district in combined_active['district'].unique():
            # Active couriers in this district (at time t)
            district_active = combined_active[combined_active['district'] == district]
            total_active = len(set(district_active['courier_id'].unique()))
            
            # Find the hour for this time point
            hour = time.hour
            
            active_data.append({
                'time': time,
                'hour': hour,
                'district': district,
                'total_active': total_active
            })
    
    # Create DataFrame
    active_df = pd.DataFrame(active_data)
    print(f"Active couriers data collected for {len(active_df)} district-time points")
    
    # Compute max, min, and average active couriers per district, including the hour of max and min
    if not active_df.empty:
        # Find the indices of max and min total_active for each district
        max_indices = active_df.groupby('district')['total_active'].idxmax()
        min_indices = active_df.groupby('district')['total_active'].idxmin()
        
        # Extract the hours for max and min
        max_hours = active_df.loc[max_indices, ['district', 'hour']].set_index('district')['hour']
        min_hours = active_df.loc[min_indices, ['district', 'hour']].set_index('district')['hour']
        
        # Aggregate statistics
        district_stats = active_df.groupby('district').agg({
            'total_active': ['max', 'min', 'mean']
        }).reset_index()
        
        district_stats.columns = ['District', 'Max Active Couriers', 'Min Active Couriers', 'Avg Active Couriers']
        district_stats['District'] = district_stats['District'].apply(lambda x: f"DA {int(x)}")
        district_stats['Max Active Couriers'] = district_stats['Max Active Couriers'].round(2)
        district_stats['Min Active Couriers'] = district_stats['Min Active Couriers'].round(2)
        district_stats['Avg Active Couriers'] = district_stats['Avg Active Couriers'].round(2)
        
        # Add the hours of max and min
        district_stats['Hour of Max'] = district_stats['District'].map(lambda x: int(max_hours.loc[int(x.split()[1])] if int(x.split()[1]) in max_hours.index else np.nan))
        district_stats['Hour of Min'] = district_stats['District'].map(lambda x: int(min_hours.loc[int(x.split()[1])] if int(x.split()[1]) in min_hours.index else np.nan))
        
        # Filter out districts with NaN values (if any)
        district_stats = district_stats.dropna(subset=['Hour of Max', 'Hour of Min'])
    else:
        print("No active couriers data available for any district.")
        district_stats = pd.DataFrame(columns=['District', 'Max Active Couriers', 'Min Active Couriers', 'Avg Active Couriers', 'Hour of Max', 'Hour of Min'])
    
    print("\nActive Couriers per District Statistics:")
    print(district_stats)
    
    # Save the table as a LaTeX file
    latex_table = district_stats.to_latex(index=False, float_format="%.2f", caption="Active Couriers per District (DA) with Hours of Max and Min", label="tab:active_couriers_per_district")
    latex_path = os.path.join(output_dir, "active_couriers_per_district_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to: {os.path.abspath(latex_path)}")
    
    return district_stats

def compute_active_couriers_and_utilization(order_df, dispatch_df, time_resolution='10min', activity_window='1h', output_dir="output"):
    """
    Compute the number of active couriers and system utilization (v_t) over time at specified resolution.
    
    Args:
        order_df: DataFrame with order data (all_waybill_info_meituan_0322.csv)
        dispatch_df: DataFrame with dispatch data (dispatch_rider_meituan.csv)
        time_resolution: Time resolution for computation (default: '10min')
        activity_window: Window to consider a courier active after last activity (default: '1h')
        output_dir: Directory to save visualizations
    
    Returns:
        DataFrame with active couriers and utilization over time
        DataFrame with hourly averages
    """
    print(f"Computing active couriers and utilization at {time_resolution} resolution...")
    
    # Create output directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    try:
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Created visualizations directory at: {os.path.abspath(viz_dir)}")
    except Exception as e:
        print(f"Error creating visualizations directory: {e}")
        raise
    
    # Create a list of all time points at the specified resolution
    start_time = min(
        order_df['grab_time'].min(),
        dispatch_df['dispatch_time'].min()
    ).floor(time_resolution)
    end_time = max(
        order_df['arrive_time'].max(),
        dispatch_df['dispatch_time'].max()
    ).ceil(time_resolution)
    all_times = pd.date_range(start=start_time, end=end_time, freq=time_resolution)
    
    # Filter orders with valid grab_time and arrive_time
    valid_orders = order_df[order_df['grab_time'].notna() & order_df['arrive_time'].notna()].copy()
    print(f"Number of orders with valid grab and arrive times: {len(valid_orders)}")
    
    # Filter dispatch data with valid dispatch_time
    valid_dispatch = dispatch_df[dispatch_df['dispatch_time'].notna()].copy()
    print(f"Number of dispatch records with valid dispatch time: {len(valid_dispatch)}")
    
    # Initialize active couriers DataFrame
    active_data = []
    
    # For each time point, determine active couriers
    activity_window = pd.Timedelta(activity_window)
    
    for time in all_times:
        # Active couriers (from dispatch data)
        dispatch_window_start = time - activity_window
        dispatch_active = valid_dispatch[
            (valid_dispatch['dispatch_time'] >= dispatch_window_start) &
            (valid_dispatch['dispatch_time'] <= time)
        ]
        
        # Identify couriers who were recently active (within the activity window)
        recent_mask = (valid_orders['arrive_time'] > time - activity_window) & (valid_orders['arrive_time'] <= time)
        recent_orders = valid_orders[recent_mask]
        recent_orders = recent_orders.sort_values('arrive_time')
        recent_orders = recent_orders.groupby('courier_id').last().reset_index()
        
        # Combine active and recently active couriers
        active_couriers_df = pd.concat([
            dispatch_active[['courier_id']],
            recent_orders[['courier_id']]
        ]).drop_duplicates(subset='courier_id')
        active_couriers = set(active_couriers_df['courier_id'])
        total_active = len(active_couriers)
        
        # Busy couriers (only those currently handling an order)
        busy_mask = (valid_orders['grab_time'] <= time) & (valid_orders['arrive_time'] > time)
        busy_orders = valid_orders[busy_mask]
        busy_orders = busy_orders.sort_values('grab_time')
        busy_orders = busy_orders.groupby('courier_id').last().reset_index()
        busy_couriers_set = set(busy_orders['courier_id'].unique())
        
        # Ensure busy couriers are a subset of active couriers
        busy_couriers = len(busy_couriers_set.intersection(active_couriers))
        
        # Compute utilization
        v_t = busy_couriers / total_active if total_active > 0 else 0
        
        active_data.append({
            'time': time,
            'busy_couriers': busy_couriers,
            'total_active': total_active,
            'v_t': v_t
        })
    
    # Create DataFrame
    active_df = pd.DataFrame(active_data)
    
    # Average by hour of day (across all days)
    active_df['hour_of_day'] = active_df['time'].dt.hour + (active_df['time'].dt.minute / 60)
    hourly_active = active_df.groupby(active_df['time'].dt.hour).agg({
        'busy_couriers': 'mean',
        'total_active': 'mean',
        'v_t': 'mean'
    }).reset_index()
    

    # Compute hourly rate multipliers for active couriers
    mean_active = hourly_active['total_active'].mean()  # Mean across all hours
    hourly_active['rate_multiplier'] = (hourly_active['total_active'] / mean_active).round(2)  # Rate multiplier

    # Create a LaTeX table for hourly rate multipliers
    # Split into two columns: 00:00-11:00 and 12:00-23:00
    latex_table = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{c c | c c}\n\\toprule\n"
    latex_table += "\\textbf{Hour} & \\textbf{Rate Multiplier} & \\textbf{Hour} & \\textbf{Rate Multiplier} \\\\\n\\midrule\n"
    
    for i in range(12):
        hour_1 = i  # 00:00 to 11:00
        hour_2 = i + 12  # 12:00 to 23:00
        rate_1 = hourly_active.loc[hourly_active['time'] == hour_1, 'rate_multiplier'].iloc[0]
        rate_2 = hourly_active.loc[hourly_active['time'] == hour_2, 'rate_multiplier'].iloc[0]
        latex_table += f"{hour_1:02d}:00 & {rate_1:.2f} & {hour_2:02d}:00 & {rate_2:.2f} \\\\\n"
    
    latex_table += "\\bottomrule\n\\end{tabular}\n"
    latex_table += "\\caption{Hourly Rate Multipliers for Active Couriers in the RMDP Simulation}\n"
    latex_table += "\\label{tab:hourly_active_courier_multipliers}\n\\end{table}"

    # Save the LaTeX table to a file
    latex_path = os.path.join(output_dir, "hourly_active_courier_multipliers.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to: {os.path.abspath(latex_path)}")


    # Compute demand distribution (orders per hour)
    order_df['hour'] = order_df['platform_order_time'].dt.hour
    hourly_counts = order_df.groupby('hour').size() / 8  # Average over 8 days
    hourly_counts = hourly_counts.reindex(range(24), fill_value=0)
    
    # Compute Pearson correlation between hourly system utilization and active couriers
    correlation, p_value = pearsonr(hourly_active['v_t'].values, hourly_active['total_active'].values)
    print(f"Pearson correlation between hourly system utilization and active couriers: {correlation:.2f} (p-value: {p_value:.3f})")
    
    # Plot active couriers and utilization
    print("Generating plot: hourly_utilization_active.png")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(hourly_active['time'], hourly_active['v_t'], marker='o', color='blue', label='Utilization (v_t)')
    ax1.set_xlabel('Hour of Day (UTC)')
    ax1.set_ylabel('System Utilization (v_t)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(hourly_active['time'], hourly_active['total_active'], marker='s', color='green', label='Active Couriers')
    # Plot demand on the same axis as active couriers (ax2)
    ax2.plot(hourly_counts.index, hourly_counts.values, marker='o', color='orange', label='Order Demand')
    ax2.set_ylabel('Number of Active Couriers / Order Demand', color='black')  # Update label to reflect both
    ax2.tick_params(axis='y', labelcolor='black')  # Adjust color for clarity
    
    # Update title to include demand
    plt.title('Average Hourly System Utilization, Active Couriers, and Demand (Averaged Over Days)')
    plt.xticks(ticks=range(0, 24, 2), labels=[f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45, fontsize=8)
    fig.tight_layout()
    # Update legend to include all three lines
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    save_path = os.path.join(viz_dir, "hourly_utilization_active.png")
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {os.path.abspath(save_path)}")
        if os.path.exists(save_path):
            print(f"Confirmed: Plot file exists at {os.path.abspath(save_path)}")
        else:
            print(f"Warning: Plot file not found at {os.path.abspath(save_path)} after saving")
    except Exception as e:
        print(f"Error saving plot hourly_utilization_active.png: {e}")
    plt.close()

    
    print("\nHourly statistics:")
    print(f"  Mean v_t: {hourly_active['v_t'].mean():.3f}")
    print(f"  Max v_t: {hourly_active['v_t'].max():.3f} at hour {hourly_active['time'][hourly_active['v_t'].idxmax()]}")
    print(f"  Min v_t: {hourly_active['v_t'].min():.3f} at hour {hourly_active['time'][hourly_active['v_t'].idxmin()]}")
    print(f"  Mean active couriers: {hourly_active['total_active'].mean():.1f}")
    print(f"  Max active couriers: {hourly_active['total_active'].max():.1f} at hour {hourly_active['time'][hourly_active['total_active'].idxmax()]}")
    print(f"  Min active couriers: {hourly_active['total_active'].min():.1f} at hour {hourly_active['time'][hourly_active['total_active'].idxmin()]}")
    
    return active_df, hourly_active




def plot_active_couriers_with_demand(order_df, hourly_active, output_dir="output"):
    """
    Plot active couriers overlaid with demand distribution and compute their correlation.
    
    Args:
        order_df: DataFrame with order data
        hourly_active: DataFrame with hourly active couriers and utilization
        output_dir: Directory to save visualizations
    """
    print("Plotting active couriers with demand distribution...")
    
    # Create output directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Using visualizations directory: {os.path.abspath(viz_dir)}")
    
    # Compute demand distribution (orders per hour)
    order_df['hour'] = order_df['platform_order_time'].dt.hour
    hourly_counts = order_df.groupby('hour').size() / 8  # Average over 8 days
    hourly_counts = hourly_counts.reindex(range(24), fill_value=0)
    
    # Compute Pearson correlation between hourly demand and active couriers
    correlation, p_value = pearsonr(hourly_counts.values, hourly_active['total_active'].values)
    print(f"\nPearson correlation between hourly demand and active couriers: {correlation:.2f} (p-value: {p_value:.3f})")
    
    # Compute Pearson correlation between hourly demand and system utilization (v_t)
    correlation_vt_demand, p_value_vt_demand = pearsonr(hourly_counts.values, hourly_active['v_t'].values)
    print(f"Pearson correlation between hourly demand and system utilization (v_t): {correlation_vt_demand:.2f} (p-value: {p_value_vt_demand:.3f})")


    # Plot active couriers and demand
    print("Generating plot: active_couriers_demand.png")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(hourly_active['time'], hourly_active['total_active'], marker='s', color='green', label='Active Couriers')
    ax1.set_xlabel('Hour of Day (UTC)')
    ax1.set_ylabel('Number of Active Couriers', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(hourly_counts.index, hourly_counts.values, marker='o', color='orange', label='Order Demand')
    ax2.set_ylabel('Average Number of Orders', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Highlight peak hours
    ax1.axvspan(10, 12, color='yellow', alpha=0.2, label='Lunch Peak')
    ax1.axvspan(17, 19, color='orange', alpha=0.2, label='Dinner Peak')
    
    plt.title('Active Couriers and Order Demand Distribution (Averaged Over 8 Days)')
    plt.xticks(ticks=range(0, 24, 2), labels=[f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45, fontsize=8)
    fig.tight_layout()
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    save_path = os.path.join(viz_dir, "active_couriers_demand.png")
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {os.path.abspath(save_path)}")
        if os.path.exists(save_path):
            print(f"Confirmed: Plot file exists at {os.path.abspath(save_path)}")
        else:
            print(f"Warning: Plot file not found at {os.path.abspath(save_path)} after saving")
    except Exception as e:
        print(f"Error saving plot active_couriers_demand.png: {e}")
    plt.close()

def correlate_utilization_with_delays(order_df, active_df, time_resolution='10min', output_dir="output"):
    """
    Correlate system utilization (v_t) with delivery delays.
    
    Args:
        order_df: DataFrame with order data
        active_df: DataFrame with utilization and active couriers data
        time_resolution: Time resolution for matching utilization (default: '10min')
        output_dir: Directory to save visualizations
    
    Returns:
        dict: Statistics on delay vs. utilization
    """
    print("Correlating system utilization with delivery delays...")
    
    # Create output directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Using visualizations directory: {os.path.abspath(viz_dir)}")
    
    # Compute delays for each order
    order_df['delay_minutes'] = np.nan
    valid_orders = order_df[order_df['arrive_time'].notna() & order_df['estimate_arrived_time'].notna()].copy()
    delay_mask = valid_orders['arrive_time'] > valid_orders['estimate_arrived_time']
    valid_orders.loc[delay_mask, 'delay_minutes'] = (valid_orders['arrive_time'] - valid_orders['estimate_arrived_time']).dt.total_seconds() / 60
    order_df.loc[valid_orders.index, 'delay_minutes'] = valid_orders['delay_minutes']
    
    late_orders = order_df[order_df['delay_minutes'].notna() & (order_df['delay_minutes'] > 0)].copy()
    print(f"Number of late orders: {len(late_orders)}")
    
    # Assign utilization to each late order based on order_push_time
    late_orders['push_time'] = late_orders['order_push_time'].dt.floor(time_resolution)
    late_orders = late_orders.merge(
        active_df[['time', 'v_t']],
        left_on='push_time',
        right_on='time',
        how='left'
    )
    
    # Drop orders where utilization couldn't be matched
    late_orders = late_orders[late_orders['v_t'].notna()]
    print(f"Number of late orders with utilization data: {len(late_orders)}")
    
    # Bin utilization into ranges
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['≤0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '>0.8']
    late_orders['v_t_bin'] = pd.cut(late_orders['v_t'], bins=bins, labels=labels, include_lowest=True)
    
    # Compute average delay per bin
    delay_by_utilization = late_orders.groupby('v_t_bin', observed=True)['delay_minutes'].agg(['mean', 'count']).reset_index()
    delay_by_utilization['mean'] = delay_by_utilization['mean'].round(2)
    
    # Overall average delay for late orders
    overall_avg_delay = late_orders['delay_minutes'].mean()
    
    # Percentage increase when v_t > 0.8
    high_util_delay = delay_by_utilization[delay_by_utilization['v_t_bin'] == '>0.8']['mean'].iloc[0] if '>0.8' in delay_by_utilization['v_t_bin'].values else np.nan
    if not np.isnan(high_util_delay):
        percentage_increase = ((high_util_delay - overall_avg_delay) / overall_avg_delay) * 100
    else:
        percentage_increase = np.nan
    
    # Compute correlation between v_t and delay
    correlation, p_value = pearsonr(late_orders['v_t'], late_orders['delay_minutes'])
    
    # Plot delay vs. utilization
    print("Generating plot: delay_vs_utilization.png")
    plt.figure(figsize=(10, 6))
    plt.bar(delay_by_utilization['v_t_bin'], delay_by_utilization['mean'])
    plt.axhline(y=overall_avg_delay, color='r', linestyle='--', label=f'Overall Mean: {overall_avg_delay:.2f} min')
    plt.xlabel('System Utilization (v_t)')
    plt.ylabel('Average Delay (minutes)')
    plt.title('Average Delay for Late Orders by System Utilization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(viz_dir, "delay_vs_utilization.png")
    try:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to: {os.path.abspath(save_path)}")
        if os.path.exists(save_path):
            print(f"Confirmed: Plot file exists at {os.path.abspath(save_path)}")
        else:
            print(f"Warning: Plot file not found at {os.path.abspath(save_path)} after saving")
    except Exception as e:
        print(f"Error saving plot delay_vs_utilization.png: {e}")
    plt.close()
    
    print("\nDelay vs. Utilization Statistics:")
    print(delay_by_utilization)
    print(f"\nOverall average delay for late orders: {overall_avg_delay:.2f} minutes")
    print(f"Average delay when v_t > 0.8: {high_util_delay if not np.isnan(high_util_delay) else 'N/A'} minutes")
    print(f"Percentage increase when v_t > 0.8: {percentage_increase if not np.isnan(percentage_increase) else 'N/A'}%")
    print(f"Pearson correlation between v_t and delay: {correlation:.2f} (p-value: {p_value:.3f})")
    
    return {
        'delay_by_utilization': delay_by_utilization,
        'overall_avg_delay': overall_avg_delay,
        'high_util_delay': high_util_delay,
        'percentage_increase': percentage_increase,
        'correlation': correlation,
        'p_value': p_value
    }


def compute_restaurant_congestion(order_df, time_resolution='10min', output_dir="output"):
    """
    Compute the restaurant congestion (c_{D,t}) over time at specified resolution.
    
    Args:
        order_df: DataFrame with order data (all_waybill_info_meituan_0322.csv)
        time_resolution: Time resolution for computation (default: '10min')
        output_dir: Directory to save results
    
    Returns:
        DataFrame with restaurant congestion over time
        DataFrame with hourly averages
    """
    print(f"Computing restaurant congestion at {time_resolution} resolution...")
    
    # Create a list of all time points at the specified resolution
    start_time = order_df['grab_time'].min().floor(time_resolution)
    end_time = order_df['arrive_time'].max().ceil(time_resolution)
    all_times = pd.date_range(start=start_time, end=end_time, freq=time_resolution)
    
    # Filter orders with valid grab_time and arrive_time
    valid_orders = order_df[order_df['grab_time'].notna() & order_df['arrive_time'].notna()].copy()
    print(f"Number of orders with valid grab and arrive times: {len(valid_orders)}")
    
    # Initialize congestion DataFrame
    congestion_data = []
    
    for time in all_times:
        # Active orders at time t (orders that have been placed and not yet delivered)
        active_orders = valid_orders[
            (valid_orders['platform_order_time'] <= time) &
            (valid_orders['arrive_time'] > time)
        ].copy()
        
        # For each order D, compute c_{D,t}
        for _, order in active_orders.iterrows():
            poi_id = order['poi_id']  # R^D
            order_id = order['order_id']  # D
            
            # Compute O_{R^D}: Number of assigned orders from R^D
            # Assigned orders: grab_time <= time < arrive_time
            assigned_orders = active_orders[
                (active_orders['poi_id'] == poi_id) &
                (active_orders['grab_time'] <= time)
            ]
            O_RD = len(assigned_orders)
            
            # Compute V_{R^D}: Number of vehicles heading to R^D
            # Vehicles heading to R^D: assigned (grab_time <= time) but not yet picked up (grab_time > time)
            vehicles_heading = active_orders[
                (active_orders['poi_id'] == poi_id) &
                (active_orders['grab_time'] > time)
            ]
            V_RD = len(vehicles_heading['courier_id'].unique()) if not vehicles_heading.empty else 0
            
            # Compute c_{D,t}
            if V_RD > 0:
                c_Dt = min(1, (O_RD / V_RD) / 5)
            else:
                c_Dt = 0
            
            congestion_data.append({
                'time': time,
                'order_id': order_id,
                'poi_id': poi_id,
                'O_RD': O_RD,
                'V_RD': V_RD,
                'c_Dt': c_Dt
            })
    
    # Create DataFrame
    congestion_df = pd.DataFrame(congestion_data)
    
    # Aggregate by hour of day (across all days)
    congestion_df['hour_of_day'] = congestion_df['time'].dt.hour + (congestion_df['time'].dt.minute / 60)
    hourly_congestion = congestion_df.groupby(congestion_df['time'].dt.hour).agg({
        'c_Dt': 'mean'
    }).reset_index()
    
    # Compute average c_{D,t} for peak (10:00-12:00, 17:00-19:00) and off-peak (00:00-05:00) hours
    peak_hours = [10, 11, 17, 18]  # 10:00-12:00 and 17:00-19:00
    off_peak_hours = [0, 1, 2, 3, 4]  # 00:00-05:00
    peak_avg = hourly_congestion[hourly_congestion['time'].isin(peak_hours)]['c_Dt'].mean()
    off_peak_avg = hourly_congestion[hourly_congestion['time'].isin(off_peak_hours)]['c_Dt'].mean()
    
    print("\nRestaurant Congestion Statistics:")
    print(f"Average c_{{D,t}} during peak hours (10:00-12:00, 17:00-19:00): {peak_avg:.2f}")
    print(f"Average c_{{D,t}} during off-peak hours (00:00-05:00): {off_peak_avg:.2f}")
    
    return congestion_df, hourly_congestion


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
    order_df = preprocess_coordinates(order_df)
    dispatch_df = preprocess_coordinates(dispatch_df)
    
    # Compute utilization per district (using da_id)
    district_util_stats, utilization_df = compute_utilization_per_district(
        order_df, dispatch_df, time_resolution='10min', activity_window='1h', output_dir=output_dir
    )
    
    # Compute active couriers per district (using da_id)
    district_active_stats = compute_active_couriers_per_district(
        order_df, dispatch_df, time_resolution='10min', activity_window='1h', output_dir=output_dir
    )
    
    # Compute active couriers and utilization (overall)
    active_df, hourly_active = compute_active_couriers_and_utilization(
        order_df, dispatch_df, time_resolution='10min', activity_window='1h', output_dir=output_dir
    )
    
    # Plot active couriers with demand
    plot_active_couriers_with_demand(order_df, hourly_active, output_dir=output_dir)
    
    # Correlate utilization with delays
    delay_stats = correlate_utilization_with_delays(order_df, active_df, time_resolution='10min', output_dir=output_dir)
    

    ### Runtime too long
    # Compute restaurant congestion (c_{D,t})
    #congestion_df, hourly_congestion = compute_restaurant_congestion(
    #    order_df, time_resolution='10min', output_dir=output_dir
    # )

    print("Active couriers, utilization, and delay analysis complete!")

if __name__ == "__main__":
    main()