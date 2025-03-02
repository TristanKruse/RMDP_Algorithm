import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta

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
            # Scale coordinates
            sender_lat_scaled = district_data['sender_lat'] / 10000000
            sender_lng_scaled = district_data['sender_lng'] / 10000000
            
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
            
            # Scale coordinates by dividing by 100,000
            restaurants['sender_lat'] = restaurants['sender_lat'] / 1000000
            restaurants['sender_lng'] = restaurants['sender_lng'] / 1000000
            
            restaurants.to_csv(os.path.join(output_dir, f"district_{district}_restaurants.csv"), index=False)
        
        # Extract vehicle positions
        # For simulation, we can use the first position recorded for each courier
        if 'courier_id' in district_data.columns and 'grab_lat' in district_data.columns:
            courier_positions = district_data.groupby('courier_id').agg({
                'grab_lat': 'first',
                'grab_lng': 'first'
            }).reset_index()
            
            # Scale coordinates by dividing by 100,000
            courier_positions['grab_lat'] = courier_positions['grab_lat'] / 1000000
            courier_positions['grab_lng'] = courier_positions['grab_lng'] / 1000000
            
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
        sender_lat_scaled = restaurants['sender_lat'] / 1000000
        sender_lng_scaled = restaurants['sender_lng'] / 1000000
        
        plt.scatter(sender_lng_scaled, sender_lat_scaled, 
                   alpha=0.7, s=30, label='Restaurants', c='blue')
        
        # Plot delivery locations (sample for clarity)
        delivery_sample = district_data[['recipient_lat', 'recipient_lng']].sample(
            min(1000, district_data.shape[0])
        )
        # Scale coordinates
        recipient_lat_scaled = delivery_sample['recipient_lat'] / 1000000
        recipient_lng_scaled = delivery_sample['recipient_lng'] / 1000000
        
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


def main():
    # Set input and output paths
    data_dir = "data/meituan_data"
    input_file = os.path.join(data_dir, "all_waybill_info_meituan_0322.csv")
    output_dir = os.path.join(data_dir, "processed")
    
    # Main processing steps
    df = load_and_clean_data(input_file)
    district_stats = analyze_business_districts(df)
    create_district_datasets(df, district_stats, output_dir)
    extract_restaurant_and_vehicle_data(df, district_stats, output_dir)
    generate_simulation_configs(district_stats, output_dir)
    visualize_districts(df, district_stats, output_dir)
    
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main()