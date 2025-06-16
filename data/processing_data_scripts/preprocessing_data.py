import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def load_and_clean_data(file_path):
    """
    Load the Meituan data and perform initial cleaning
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Basic cleaning
    print(f"Original data shape: {df.shape}")
    
    # Convert timestamp columns to datetime
    timestamp_cols = ['platform_order_time', 'estimate_meal_prepare_time', 
                     'order_push_time', 'dispatch_time', 'grab_time', 
                     'fetch_time', 'arrive_time', 'estimate_arrived_time']
    
    for col in timestamp_cols:
        if col in df.columns:
            # Convert UNIX timestamp to datetime and add 8 hours to convert from Chinese time to UTC
            df[col] = pd.to_datetime(df[col], unit='s') + pd.Timedelta(hours=8)
    
    # Handle missing values
    # Convert '0' timestamps to NaT
    for col in timestamp_cols:
        if col in df.columns:
            df.loc[df[col] == pd.Timestamp(0) + pd.Timedelta(hours=8), col] = pd.NaT
    
    print(f"Data after cleaning: {df.shape}")
    return df


def analyze_business_districts(df):
    """
    Analyze and visualize the business districts in the data
    """
    print("Analyzing business districts...")
    
    # Get unique business districts
    districts = df['da_id'].unique()
    print(f"Found {len(districts)} unique business districts")
    
    # Create summary statistics for each district
    district_stats = {}
    for district in districts:
        district_data = df[df['da_id'] == district]
        
        # Calculate key metrics
        n_orders = district_data.shape[0]
        n_restaurants = district_data['poi_id'].nunique()
        n_couriers = district_data['courier_id'].nunique()
        
        # Get geographic boundaries
        if 'sender_lat' in district_data.columns and 'sender_lng' in district_data.columns:
            # Use coordinates directly (already scaled)
            sender_lat_scaled = district_data['sender_lat']
            sender_lng_scaled = district_data['sender_lng']
            
            lat_min, lat_max = sender_lat_scaled.min(), sender_lat_scaled.max()
            lng_min, lng_max = sender_lng_scaled.min(), sender_lng_scaled.max()
            
            # Calculate approximate district size in km (very rough approximation)
            # 1 degree of latitude is approximately 111 km
            lat_range_km = (lat_max - lat_min) * 111
            # 1 degree of longitude varies based on latitude, using an average
            lng_range_km = (lng_max - lng_min) * 111 * np.cos(np.radians((lat_min + lat_max) / 2))
            
            district_stats[district] = {
                'n_orders': n_orders,
                'n_restaurants': n_restaurants,
                'n_couriers': n_couriers,
                'lat_range': (lat_min, lat_max),
                'lng_range': (lng_min, lng_max),
                'approx_size_km': (lat_range_km, lng_range_km)
            }
    
    # Print summary statistics
    for district, stats in district_stats.items():
        print(f"\nDistrict {district}:")
        print(f"  Orders: {stats['n_orders']}")
        print(f"  Restaurants: {stats['n_restaurants']}")
        print(f"  Couriers: {stats['n_couriers']}")
        if 'approx_size_km' in stats:
            print(f"  Approximate size: {stats['approx_size_km'][0]:.2f} km × {stats['approx_size_km'][1]:.2f} km")
    
    return district_stats


def create_district_datasets(df, district_stats, output_dir):
    """
    Split data by business district and save to separate files
    """
    print(f"Creating district datasets in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    for district, stats in district_stats.items():
        district_data = df[df['da_id'] == district]
        
        # Save district data
        district_file = os.path.join(output_dir, f"district_{district}_data.csv")
        district_data.to_csv(district_file, index=False)
        print(f"Saved district {district} data ({district_data.shape[0]} rows) to {district_file}")
        
        # Create a summary file with district parameters
        with open(os.path.join(output_dir, f"district_{district}_summary.txt"), 'w') as f:
            f.write(f"District {district} Summary\n")
            f.write(f"Total Orders: {stats['n_orders']}\n")
            f.write(f"Unique Restaurants: {stats['n_restaurants']}\n")
            f.write(f"Unique Couriers: {stats['n_couriers']}\n")
            f.write(f"Geographic Range: {stats['approx_size_km'][0]:.2f} km × {stats['approx_size_km'][1]:.2f} km\n")
            
            # Add more statistics as needed for simulation parameters
            if district_data['order_push_time'].notna().any():
                f.write("\nOrder Timing (Local Time):\n")
                hourly_orders = district_data.groupby(district_data['order_push_time'].dt.hour).size()
                f.write(f"Peak Hour: {hourly_orders.idxmax()}:00 with {hourly_orders.max()} orders\n")
                
            # Courier activity patterns
            if 'grab_time' in district_data.columns and district_data['grab_time'].notna().any():
                courier_activity = district_data.groupby('courier_id').size().describe()
                f.write(f"\nCourier Orders per Day:\n")
                f.write(f"Min: {courier_activity['min']:.1f}\n")
                f.write(f"Mean: {courier_activity['mean']:.1f}\n")
                f.write(f"Max: {courier_activity['max']:.1f}\n")
                
            # Order preparation and delivery times
            if 'grab_time' in district_data.columns and 'arrive_time' in district_data.columns:
                valid_times = district_data[(district_data['grab_time'].notna()) & (district_data['arrive_time'].notna())]
                if not valid_times.empty:
                    delivery_times = (valid_times['arrive_time'] - valid_times['grab_time']).dt.total_seconds() / 60
                    f.write("\nDelivery Times (minutes):\n")
                    f.write(f"Min: {delivery_times.min():.1f}\n")
                    f.write(f"Mean: {delivery_times.mean():.1f}\n")
                    f.write(f"Max: {delivery_times.max():.1f}\n")
                    
            # Preparation times
            if 'order_push_time' in district_data.columns and 'fetch_time' in district_data.columns:
                valid_prep = district_data[(district_data['order_push_time'].notna()) & (district_data['fetch_time'].notna())]
                if not valid_prep.empty:
                    prep_times = (valid_prep['fetch_time'] - valid_prep['order_push_time']).dt.total_seconds() / 60
                    f.write("\nPreparation Times (minutes):\n")
                    f.write(f"Min: {prep_times.min():.1f}\n")
                    f.write(f"Mean: {prep_times.mean():.1f}\n")
                    f.write(f"Max: {prep_times.max():.1f}\n")


def generate_simulation_configs(district_stats, output_dir):
    """
    Generate simulation configuration files for each district
    """
    print("Generating simulation configuration files...")
    os.makedirs(os.path.join(output_dir, "configs"), exist_ok=True)
    
    for district, stats in district_stats.items():
        # Convert NumPy types to Python native types to make them JSON serializable
        config = {
            "district_id": int(district) if isinstance(district, np.integer) else district,
            "simulation_params": {
                "num_restaurants": int(stats['n_restaurants']),
                "num_vehicles": int(stats['n_couriers']),
                "service_area_dimensions": [
                    float(stats['approx_size_km'][0]), 
                    float(stats['approx_size_km'][1])
                ],
                "movement_per_step": 0.4,  # Assuming this is a reasonable default
                "mean_prep_time": 10.0,  # Default from your simulation
                "delivery_window": 40.0,  # Default from your simulation
                "downtown_concentration": 0.7,  # Default from your simulation
            },
            "restaurant_locations": f"district_{district}_restaurants.csv",
            "vehicle_start_positions": f"district_{district}_vehicles.csv"
        }
        
        # Save configuration to JSON
        import json
        with open(os.path.join(output_dir, "configs", f"district_{district}_config.json"), 'w') as f:
            json.dump(config, f, indent=2)


def extract_restaurant_and_vehicle_data(df, district_stats, output_dir):
    """
    Extract restaurant locations and initial vehicle positions for each district
    """
    print("Extracting restaurant and vehicle position data...")
    
    for district, stats in district_stats.items():
        district_data = df[df['da_id'] == district]
        
    # Extract unique restaurant locations
    if 'sender_lat' in district_data.columns and 'sender_lng' in district_data.columns:
        restaurants = district_data[['poi_id', 'sender_lat', 'sender_lng']].drop_duplicates()
        # No need to scale - already done
        restaurants.to_csv(os.path.join(output_dir, f"district_{district}_restaurants.csv"), index=False)
    
    # Extract vehicle positions
    if 'courier_id' in district_data.columns and 'grab_lat' in district_data.columns:
        courier_positions = district_data.groupby('courier_id').agg({
            'grab_lat': 'first',
            'grab_lng': 'first'
        }).reset_index()
        
        # No need to scale - already done
        courier_positions.to_csv(os.path.join(output_dir, f"district_{district}_vehicles.csv"), index=False)


def visualize_districts(df, district_stats, output_dir):
    """
    Create visualizations for each district
    """
    print("Creating district visualizations...")
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    for district, stats in district_stats.items():
        district_data = df[df['da_id'] == district]
        
        # Create scatter plot of restaurant and delivery locations
        plt.figure(figsize=(10, 10))
        
        # Plot restaurants
        restaurants = district_data[['poi_id', 'sender_lat', 'sender_lng']].drop_duplicates()
        # Scale coordinates
        sender_lat_scaled = restaurants['sender_lat']  
        sender_lng_scaled = restaurants['sender_lng']
        
        plt.scatter(sender_lng_scaled, sender_lat_scaled, 
                   alpha=0.7, s=30, label='Restaurants', c='blue')
        
        # Plot delivery locations (sample for clarity)
        delivery_sample = district_data[['recipient_lat', 'recipient_lng']].sample(
            min(1000, district_data.shape[0])
        )
        # Scale coordinates
        recipient_lat_scaled = delivery_sample['recipient_lat']
        recipient_lng_scaled = delivery_sample['recipient_lng']
        
        plt.scatter(recipient_lng_scaled, recipient_lat_scaled, 
                   alpha=0.3, s=10, label='Delivery Locations', c='green')
        
        plt.title(f"District {district} - Restaurants and Delivery Locations")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "visualizations", f"district_{district}_map.png"), dpi=300)
        plt.close()
        
        # Create time distribution of orders
        if 'order_push_time' in district_data.columns and district_data['order_push_time'].notna().any():
            plt.figure(figsize=(12, 6))
            
            # Use the hour in local time (already adjusted by +8 hours)
            hourly_orders = district_data.groupby(district_data['order_push_time'].dt.hour).size()
            
            # Create a nicer x-axis with formatted hour labels
            plt.bar(hourly_orders.index, hourly_orders.values)
            plt.xticks(range(0, 24), [f"{h:02d}:00" for h in range(0, 24)], rotation=45)
            
            plt.title(f"District {district} - Hourly Order Distribution (Local Time)")
            plt.xlabel("Hour of Day (Local Time)")
            plt.ylabel("Number of Orders")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "visualizations", f"district_{district}_hourly_orders.png"), dpi=300)
            plt.close()


def extract_daily_orders(df, district_stats, output_dir):
    """
    Extract orders by day (based on order_push_time) and district
    """
    print("Extracting daily orders by district...")
    
    # Create directory for daily order data
    daily_order_dir = os.path.join(output_dir, "daily_orders")
    os.makedirs(daily_order_dir, exist_ok=True)
    
    # Make sure order_push_time is datetime
    if 'order_push_time' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['order_push_time']):
            df['order_push_time'] = pd.to_datetime(df['order_push_time'])
    else:
        print("WARNING: order_push_time column not found, falling back to dt")
        df['date'] = df['dt']
        unique_days = df['date'].unique()
        # Continue with the rest of the function using 'dt'...
        return
        
    # Extract date part from order_push_time
    df['order_date'] = df['order_push_time'].dt.strftime('%Y%m%d')
    
    # Get unique days from order_push_time
    unique_days = sorted(df['order_date'].unique())
    print(f"Found {len(unique_days)} unique days based on order_push_time")
    
    # Get unique districts
    unique_districts = df['da_id'].unique()
    
    for day in unique_days:
        day_data = df[df['order_date'] == day]
        
        # Create directory for this day
        day_dir = os.path.join(daily_order_dir, str(day))
        os.makedirs(day_dir, exist_ok=True)
        
        print(f"Processing orders for day {day}...")
        
        for district in unique_districts:
            district_day_data = day_data[day_data['da_id'] == district]
            
            # Only save if there's data
            if not district_day_data.empty:
                # Save order data for this day and district
                district_day_data.to_csv(
                    os.path.join(day_dir, f"district_{district}_orders.csv"), 
                    index=False
                )
                print(f"  Saved {len(district_day_data)} orders for district {district}")
                
                # Create summary with time analysis
                with open(os.path.join(day_dir, f"district_{district}_summary.txt"), 'w') as f:
                    f.write(f"Day: {day}, District: {district}\n")
                    f.write(f"Total Orders: {len(district_day_data)}\n")
                    f.write(f"Unique Restaurants: {district_day_data['poi_id'].nunique()}\n")
                    f.write(f"Unique Couriers: {district_day_data['courier_id'].nunique()}\n")
                    
                    # Time range and distribution
                    if district_day_data['order_push_time'].notna().any():
                        start_time = district_day_data['order_push_time'].min()
                        end_time = district_day_data['order_push_time'].max()
                        f.write(f"Time Range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        
                        # Analyze hourly distribution
                        hourly_counts = district_day_data['order_push_time'].dt.hour.value_counts().sort_index()
                        f.write(f"\nHourly Order Distribution:\n")
                        for hour, count in hourly_counts.items():
                            f.write(f"  {hour:02d}:00 - {count} orders\n")
                        
                        f.write(f"\nPeak Hour: {hourly_counts.idxmax():02d}:00 with {hourly_counts.max()} orders\n")

                    # Analyze delivery windows if available
                    if 'delivery_window_minutes' in district_day_data.columns:
                        valid_windows = district_day_data['delivery_window_minutes'].dropna()
                        if not valid_windows.empty:
                            f.write(f"\nDelivery Window Statistics:\n")
                            f.write(f"  Mean: {valid_windows.mean():.1f} minutes\n")
                            f.write(f"  Median: {valid_windows.median():.1f} minutes\n")
                            f.write(f"  Min: {valid_windows.min():.1f} minutes\n")
                            f.write(f"  Max: {valid_windows.max():.1f} minutes\n")
                            
                    # Analyze prep times if available
                    if 'prep_time_minutes' in district_day_data.columns:
                        valid_prep = district_day_data['prep_time_minutes'].dropna()
                        if not valid_prep.empty:
                            f.write(f"\nPreparation Time Statistics:\n")
                            f.write(f"  Mean: {valid_prep.mean():.1f} minutes\n")
                            f.write(f"  Median: {valid_prep.median():.1f} minutes\n")
                            f.write(f"  Min: {valid_prep.min():.1f} minutes\n")
                            f.write(f"  Max: {valid_prep.max():.1f} minutes\n")
                    
                # Restaurant positions for this day
                restaurants = district_day_data[['poi_id', 'sender_lat', 'sender_lng']].drop_duplicates()
                restaurants.to_csv(os.path.join(day_dir, f"district_{district}_restaurants.csv"), index=False)

                # Vehicle positions for this day
                courier_positions = district_day_data.groupby('courier_id').agg({
                    'grab_lat': 'first',
                    'grab_lng': 'first'
                }).reset_index()
                courier_positions.to_csv(os.path.join(day_dir, f"district_{district}_vehicles.csv"), index=False)


def preprocess_coordinates(df, scale_factor=1000000.0):
    """
    Normalize Meituan coordinates by dividing by scale_factor (default: 1,000,000)
    
    Args:
        df: DataFrame containing coordinate columns
        scale_factor: Factor to divide coordinates by
        
    Returns:
        DataFrame with processed coordinates
    """
    coordinate_columns = [
        'sender_lat', 'sender_lng', 'recipient_lat', 'recipient_lng',
        'grab_lat', 'grab_lng', 'rider_lat', 'rider_lng'
    ]
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    for col in coordinate_columns:
        if col in processed_df.columns:
            # Check if scaling is needed (values > 1000)
            if processed_df[col].abs().max() > 1000:
                print(f"Scaling down {col} from Meituan format")
                processed_df[col] = processed_df[col] / scale_factor
    
    return processed_df


def analyze_meal_prep_times(df, output_dir):
    """
    Analyze and calculate meal preparation times from real data.
    
    This function:
    1. Calculates prep time as (estimate_meal_prepare_time - platform_order_time)
    2. Filters to reasonable values (1-60 minutes)
    3. Creates a visualization of prep time distribution
    4. Adds a 'prep_time_minutes' column to the dataframe
    
    Args:
        df: DataFrame containing order data
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
    
    # Check if necessary columns exist
    if 'platform_order_time' not in processed_df.columns or 'estimate_meal_prepare_time' not in processed_df.columns:
        print("Warning: Missing required columns for meal prep time calculation")
        return processed_df
    
    # Calculate prep time (in minutes)
    valid_times = processed_df[processed_df['platform_order_time'].notna() & 
                               processed_df['estimate_meal_prepare_time'].notna()]
    
    if len(valid_times) == 0:
        print("Warning: No valid data for meal prep time calculation")
        return processed_df
    
    # Calculate preparation time in minutes
    prep_times = (valid_times['estimate_meal_prepare_time'] - 
                valid_times['platform_order_time']).dt.total_seconds() / 60
    
    # Add rounded prep time to the original dataframe
    processed_df['prep_time_minutes'] = np.nan
    processed_df.loc[valid_times.index, 'prep_time_minutes'] = np.round(prep_times)
    
    # Filter out unreasonable values
    reasonable_mask = (processed_df['prep_time_minutes'] >= 1) & (processed_df['prep_time_minutes'] <= 60)
    unreasonable_count = len(processed_df[~reasonable_mask & processed_df['prep_time_minutes'].notna()])
    
    if unreasonable_count > 0:
        print(f"Found {unreasonable_count} orders with unreasonable prep times (< 1 min or > 60 min)")
        processed_df.loc[~reasonable_mask & processed_df['prep_time_minutes'].notna(), 'prep_time_minutes'] = np.nan
    
    # Create visualization of overall distribution
    valid_prep_times = processed_df['prep_time_minutes'].dropna()
    
    plt.figure(figsize=(12, 6))
    plt.hist(valid_prep_times, bins=30, alpha=0.7)
    plt.axvline(x=valid_prep_times.mean(), color='r', linestyle='--', 
               label=f'Mean: {valid_prep_times.mean():.1f} min')
    plt.axvline(x=valid_prep_times.median(), color='g', linestyle='-.', 
               label=f'Median: {valid_prep_times.median():.1f} min')
    
    plt.title('Distribution of Meal Preparation Times')
    plt.xlabel('Preparation Time (minutes)')
    plt.ylabel('Number of Orders')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, "meal_prep_time_distribution.png"), dpi=300)
    plt.close()
    
    # Analyze prep times by restaurant
    top_restaurants = processed_df.groupby('poi_id').size().nlargest(20).index
    
    restaurant_stats = []
    for rest_id in top_restaurants:
        rest_prep_times = processed_df[processed_df['poi_id'] == rest_id]['prep_time_minutes'].dropna()
        if len(rest_prep_times) > 10:  # Only include if we have enough data
            restaurant_stats.append({
                'restaurant_id': rest_id,
                'order_count': len(rest_prep_times),
                'mean_prep_time': rest_prep_times.mean(),
                'median_prep_time': rest_prep_times.median(),
                'std_dev': rest_prep_times.std()
            })
    
    # Create restaurant comparison chart
    if restaurant_stats:
        rest_df = pd.DataFrame(restaurant_stats)
        rest_df = rest_df.sort_values('mean_prep_time')
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(rest_df['restaurant_id'].astype(str), rest_df['mean_prep_time'])
        
        # Add error bars for standard deviation
        plt.errorbar(range(len(rest_df)), rest_df['mean_prep_time'], 
                    yerr=rest_df['std_dev'], fmt='none', ecolor='black', capsize=5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        plt.title('Average Preparation Time by Restaurant')
        plt.xlabel('Restaurant ID')
        plt.ylabel('Average Preparation Time (minutes)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "restaurant_prep_times.png"), dpi=300)
        plt.close()
    
    # Print summary statistics
    print(f"Meal preparation time statistics:")
    print(f"  Valid data points: {len(valid_prep_times)}")
    print(f"  Mean prep time: {valid_prep_times.mean():.1f} minutes")
    print(f"  Median prep time: {valid_prep_times.median():.1f} minutes")
    print(f"  Min prep time: {valid_prep_times.min():.1f} minutes")
    print(f"  Max prep time: {valid_prep_times.max():.1f} minutes")
    
    return processed_df


def analyze_delivery_windows(df, output_dir):
    """
    Analyze and calculate delivery windows from real data.
    
    This function:
    1. Calculates delivery window as (estimate_arrived_time - platform_order_time)
    2. Filters to reasonable values (10-120 minutes)
    3. Creates a visualization of delivery window distribution
    4. Adds a 'delivery_window_minutes' column to the dataframe
    
    Args:
        df: DataFrame containing order data
        output_dir: Directory to save visualizations
        
    Returns:
        DataFrame with added 'delivery_window_minutes' column
    """
    print("Analyzing delivery windows...")
    
    # Create output directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Check if necessary columns exist
    if 'platform_order_time' not in processed_df.columns or 'estimate_arrived_time' not in processed_df.columns:
        print("Warning: Missing required columns for delivery window calculation")
        return processed_df
    
    # Calculate delivery window (in minutes)
    valid_times = processed_df[processed_df['platform_order_time'].notna() & 
                           processed_df['estimate_arrived_time'].notna()]
    
    if len(valid_times) == 0:
        print("Warning: No valid data for delivery window calculation")
        return processed_df
    
    # Calculate delivery window in minutes
    delivery_windows = (valid_times['estimate_arrived_time'] - 
                valid_times['platform_order_time']).dt.total_seconds() / 60
    
    # Add rounded delivery window to the original dataframe
    processed_df['delivery_window_minutes'] = np.nan
    processed_df.loc[valid_times.index, 'delivery_window_minutes'] = np.round(delivery_windows)
    
    # Filter out unreasonable values
    reasonable_mask = (processed_df['delivery_window_minutes'] >= 10) & (processed_df['delivery_window_minutes'] <= 120)
    unreasonable_count = len(processed_df[~reasonable_mask & processed_df['delivery_window_minutes'].notna()])
    
    if unreasonable_count > 0:
        print(f"Found {unreasonable_count} orders with unreasonable delivery windows (< 10 min or > 120 min)")
        processed_df.loc[~reasonable_mask & processed_df['delivery_window_minutes'].notna(), 'delivery_window_minutes'] = np.nan
    
    # Create visualization of overall distribution
    valid_windows = processed_df['delivery_window_minutes'].dropna()
    
    plt.figure(figsize=(12, 6))
    plt.hist(valid_windows, bins=30, alpha=0.7)
    plt.axvline(x=valid_windows.mean(), color='r', linestyle='--', 
               label=f'Mean: {valid_windows.mean():.1f} min')
    plt.axvline(x=valid_windows.median(), color='g', linestyle='-.', 
               label=f'Median: {valid_windows.median():.1f} min')
    
    plt.title('Distribution of Delivery Windows')
    plt.xlabel('Delivery Window (minutes)')
    plt.ylabel('Number of Orders')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, "delivery_window_distribution.png"), dpi=300)
    plt.close()
    
    # Print summary statistics
    print(f"Delivery window statistics:")
    print(f"  Valid data points: {len(valid_windows)}")
    print(f"  Mean delivery window: {valid_windows.mean():.1f} minutes")
    print(f"  Median delivery window: {valid_windows.median():.1f} minutes")
    print(f"  Min delivery window: {valid_windows.min():.1f} minutes")
    print(f"  Max delivery window: {valid_windows.max():.1f} minutes")
    
    return processed_df


def main():
    # Set input and output paths
    data_dir = "data/meituan_data"
    input_file = os.path.join(data_dir, "all_waybill_info_meituan_0322.csv")
    output_dir = os.path.join(data_dir, "processed")
    
    # Main processing steps
    df = load_and_clean_data(input_file)
    df = preprocess_coordinates(df)
    df = analyze_meal_prep_times(df, output_dir)
    df = analyze_delivery_windows(df, output_dir) 

    district_stats = analyze_business_districts(df)
    create_district_datasets(df, district_stats, output_dir)
    extract_restaurant_and_vehicle_data(df, district_stats, output_dir)
    generate_simulation_configs(district_stats, output_dir)
    visualize_districts(df, district_stats, output_dir)
    
    # Add the new function to extract daily orders
    extract_daily_orders(df, district_stats, output_dir)
    
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main()