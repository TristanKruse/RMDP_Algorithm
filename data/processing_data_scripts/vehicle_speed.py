import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt
import os
from datetime import datetime

# Set up output directory for results
output_dir = "speed_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Function to calculate the Haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def load_and_prepare_data(waybill_file, rider_file):
    """Load and prepare the Meituan data files"""
    print(f"Loading data from {waybill_file} and {rider_file}...")
    
    # Load the datasets
    waybill_data = pd.read_csv(waybill_file)
    rider_data = pd.read_csv(rider_file)
    
    # Convert Unix timestamps to datetime
    print("Converting timestamps...")
    for col in ['grab_time', 'fetch_time', 'arrive_time']:
        if col in waybill_data.columns:
            waybill_data[col] = pd.to_datetime(waybill_data[col], unit='s', errors='coerce')
    
    if 'dispatch_time' in rider_data.columns:
        rider_data['dispatch_time'] = pd.to_datetime(rider_data['dispatch_time'], unit='s', errors='coerce')
    
    print(f"Loaded {len(waybill_data)} waybill records and {len(rider_data)} rider records")
    return waybill_data, rider_data

def calculate_speeds_from_waybill(waybill_data):
    """Calculate speeds from waybill data (grab to fetch, fetch to arrive)"""
    print("Calculating speeds from waybill data...")
    speeds = []
    
    # Filter valid data for grab to fetch
    valid_grab_fetch = waybill_data[
        (waybill_data['grab_lat'] != 0) & 
        (waybill_data['grab_lng'] != 0) & 
        (waybill_data['sender_lat'] != 0) & 
        (waybill_data['sender_lng'] != 0) & 
        (waybill_data['grab_time'].notna()) &
        (waybill_data['fetch_time'].notna())
    ]
    
    print(f"Found {len(valid_grab_fetch)} valid records for grab-to-fetch analysis")
    
    # Calculate speed from grab location to restaurant (sender location)
    for _, row in valid_grab_fetch.iterrows():
        # Normalize coordinates
        grab_lat = row['grab_lat'] / 1000000
        grab_lng = row['grab_lng'] / 1000000
        sender_lat = row['sender_lat'] / 1000000
        sender_lng = row['sender_lng'] / 1000000
        
        # Calculate distance in km
        distance = haversine_distance(grab_lat, grab_lng, sender_lat, sender_lng)
        
        # Calculate time difference in hours
        time_diff = (row['fetch_time'] - row['grab_time']).total_seconds() / 3600
        
        # Calculate speed in km/h if time difference is positive and distance is reasonable
        if time_diff > 0 and distance > 0.05:  # Minimum 50m to avoid GPS errors
            speed = distance / time_diff
            speeds.append({
                'courier_id': row['courier_id'],
                'distance_km': distance,
                'time_hours': time_diff,
                'speed_kmh': speed,
                'segment': 'grab_to_fetch',
                'timestamp': row['grab_time']
            })
    
    # Filter valid data for fetch to arrive
    valid_fetch_arrive = waybill_data[
        (waybill_data['sender_lat'] != 0) & 
        (waybill_data['sender_lng'] != 0) &
        (waybill_data['recipient_lat'] != 0) & 
        (waybill_data['recipient_lng'] != 0) & 
        (waybill_data['fetch_time'].notna()) &
        (waybill_data['arrive_time'].notna())
    ]
    
    print(f"Found {len(valid_fetch_arrive)} valid records for fetch-to-arrive analysis")
    
    # Calculate speed from restaurant to customer
    for _, row in valid_fetch_arrive.iterrows():
        sender_lat = row['sender_lat'] / 1000000
        sender_lng = row['sender_lng'] / 1000000
        recipient_lat = row['recipient_lat'] / 1000000
        recipient_lng = row['recipient_lng'] / 1000000
        
        distance = haversine_distance(sender_lat, sender_lng, recipient_lat, recipient_lng)
        time_diff = (row['arrive_time'] - row['fetch_time']).total_seconds() / 3600
        
        if time_diff > 0 and distance > 0.05:
            speed = distance / time_diff
            speeds.append({
                'courier_id': row['courier_id'],
                'distance_km': distance,
                'time_hours': time_diff,
                'speed_kmh': speed,
                'segment': 'fetch_to_arrive',
                'timestamp': row['fetch_time']
            })
    
    speed_df = pd.DataFrame(speeds)
    print(f"Generated {len(speed_df)} speed records from waybill data")
    return speed_df

def calculate_speeds_from_rider_data(rider_data):
    """Calculate speeds from consecutive rider positions"""
    print("Calculating speeds from rider data...")
    speeds = []
    
    # Group by courier_id and sort by dispatch_time
    total_couriers = rider_data['courier_id'].nunique()
    print(f"Processing data for {total_couriers} couriers...")
    
    for i, (courier_id, group) in enumerate(rider_data.groupby('courier_id')):
        if i % 50 == 0 and i > 0:
            print(f"Processed {i}/{total_couriers} couriers...")
            
        group = group.sort_values('dispatch_time')
        
        # Calculate speeds between consecutive points
        for i in range(1, len(group)):
            prev_row = group.iloc[i-1]
            curr_row = group.iloc[i]
            
            prev_lat = prev_row['rider_lat'] / 1000000
            prev_lng = prev_row['rider_lng'] / 1000000
            curr_lat = curr_row['rider_lat'] / 1000000
            curr_lng = curr_row['rider_lng'] / 1000000
            
            distance = haversine_distance(prev_lat, prev_lng, curr_lat, curr_lng)
            time_diff = (curr_row['dispatch_time'] - prev_row['dispatch_time']).total_seconds() / 3600
            
            if time_diff > 0 and distance > 0.05:  # Minimum 50m to avoid GPS errors
                speed = distance / time_diff
                speeds.append({
                    'courier_id': courier_id,
                    'distance_km': distance,
                    'time_hours': time_diff,
                    'speed_kmh': speed,
                    'segment': 'dispatch_to_dispatch',
                    'timestamp': prev_row['dispatch_time']
                })
    
    speed_df = pd.DataFrame(speeds)
    print(f"Generated {len(speed_df)} speed records from rider data")
    return speed_df

def filter_realistic_speeds(speed_df, min_speed=1, max_speed=100):
    """Filter out unrealistic speed values"""
    total = len(speed_df)
    filtered = speed_df[(speed_df['speed_kmh'] >= min_speed) & (speed_df['speed_kmh'] <= max_speed)]
    filtered_out = total - len(filtered)
    
    print(f"Filtered out {filtered_out} unrealistic speed values ({filtered_out/total*100:.1f}%)")
    print(f"Retained {len(filtered)} realistic speed records")
    
    return filtered

def analyze_speeds(speeds_df):
    """Analyze speed data and calculate KPIs"""
    print("\n===== Speed Analysis Results =====")
    
    if len(speeds_df) == 0:
        print("No speed data to analyze!")
        return {}
    
    stats = {
        'mean_speed': speeds_df['speed_kmh'].mean(),
        'median_speed': speeds_df['speed_kmh'].median(),
        'std_dev': speeds_df['speed_kmh'].std(),
        'min_speed': speeds_df['speed_kmh'].min(),
        'max_speed': speeds_df['speed_kmh'].max(),
        'q25': speeds_df['speed_kmh'].quantile(0.25),
        'q75': speeds_df['speed_kmh'].quantile(0.75),
        'count': len(speeds_df)
    }
    
    print("Speed Statistics (km/h):")
    print(f"Number of data points: {stats['count']}")
    print(f"Mean speed: {stats['mean_speed']:.2f}")
    print(f"Median speed: {stats['median_speed']:.2f}")
    print(f"Standard deviation: {stats['std_dev']:.2f}")
    print(f"Min speed: {stats['min_speed']:.2f}")
    print(f"Max speed: {stats['max_speed']:.2f}")
    print(f"25th percentile: {stats['q25']:.2f}")
    print(f"75th percentile: {stats['q75']:.2f}")
    
    # Analyze by segment type
    segment_stats = speeds_df.groupby('segment')['speed_kmh'].agg(['mean', 'median', 'std', 'count']).reset_index()
    print("\nSpeed Statistics by Segment Type:")
    print(segment_stats)
    
    # Save segment stats to CSV
    segment_stats.to_csv(f"{output_dir}/segment_speed_stats.csv", index=False)
    
    # Analyze by courier
    courier_stats = speeds_df.groupby('courier_id')['speed_kmh'].agg(['mean', 'median', 'std', 'count']).reset_index()
    min_data_points = 5  # Minimum data points for reliable courier stats
    reliable_courier_stats = courier_stats[courier_stats['count'] >= min_data_points]
    
    print(f"\nCourier Statistics (with at least {min_data_points} data points):")
    print(f"Total couriers analyzed: {len(reliable_courier_stats)}")
    
    if len(reliable_courier_stats) > 0:
        print(f"\nTop 5 fastest couriers:")
        print(reliable_courier_stats.sort_values('mean', ascending=False).head(5)[['courier_id', 'mean', 'median', 'count']])
        
        print(f"\nTop 5 slowest couriers:")
        print(reliable_courier_stats.sort_values('mean').head(5)[['courier_id', 'mean', 'median', 'count']])
        
        # Save courier stats to CSV
        reliable_courier_stats.to_csv(f"{output_dir}/courier_speed_stats.csv", index=False)
    
    # If timestamp is available, analyze by time of day
    if 'timestamp' in speeds_df.columns:
        speeds_df['hour'] = speeds_df['timestamp'].dt.hour
        hourly_stats = speeds_df.groupby('hour')['speed_kmh'].agg(['mean', 'median', 'count']).reset_index()
        
        print("\nSpeed Statistics by Hour of Day:")
        print(hourly_stats)
        
        # Save hourly stats to CSV
        hourly_stats.to_csv(f"{output_dir}/hourly_speed_stats.csv", index=False)
    
    return stats

def advanced_courier_analysis(speeds_df):
    """Perform advanced analysis of courier speed patterns"""
    print("\n===== Advanced Courier Analysis =====")
    
    if len(speeds_df) == 0:
        print("No speed data to analyze!")
        return None, None
    
    # 1. Speed consistency (standard deviation of speeds per courier)
    courier_consistency = speeds_df.groupby('courier_id')['speed_kmh'].agg(['mean', 'std']).reset_index()
    courier_consistency['coefficient_of_variation'] = courier_consistency['std'] / courier_consistency['mean']
    
    # Filter couriers with enough data points
    courier_counts = speeds_df.groupby('courier_id').size().reset_index(name='count')
    courier_consistency = courier_consistency.merge(courier_counts, on='courier_id')
    courier_consistency = courier_consistency[courier_consistency['count'] >= 5]
    
    print("\nCourier Speed Consistency (lower coefficient of variation = more consistent):")
    if len(courier_consistency) > 0:
        top_consistent = courier_consistency.sort_values('coefficient_of_variation').head(5)
        print(top_consistent[['courier_id', 'coefficient_of_variation', 'mean', 'std', 'count']])
        
        # Save consistency data to CSV
        courier_consistency.to_csv(f"{output_dir}/courier_consistency.csv", index=False)
    else:
        print("No couriers with sufficient data points for consistency analysis.")
    
    # 2. Compare speeds during different trip segments
    segment_comparison = None
    if len(speeds_df['segment'].unique()) > 1:
        try:
            segment_comparison = speeds_df.groupby(['courier_id', 'segment'])['speed_kmh'].mean().unstack().reset_index()
            
            # Calculate ratio of fetch_to_arrive vs grab_to_fetch speeds where both exist
            if all(col in segment_comparison.columns for col in ['fetch_to_arrive', 'grab_to_fetch']):
                segment_comparison['delivery_pickup_ratio'] = segment_comparison['fetch_to_arrive'] / segment_comparison['grab_to_fetch']
                
                # Filter out NaN and inf values
                segment_comparison = segment_comparison[
                    segment_comparison['delivery_pickup_ratio'].notna() & 
                    np.isfinite(segment_comparison['delivery_pickup_ratio'])
                ]
                
                if len(segment_comparison) > 0:
                    print("\nCouriers who drive faster during delivery than pickup (top 5):")
                    print(segment_comparison.sort_values('delivery_pickup_ratio', ascending=False).head(5))
                    
                    print("\nCouriers who drive faster during pickup than delivery (top 5):")
                    print(segment_comparison.sort_values('delivery_pickup_ratio').head(5))
                    
                    # Save segment comparison to CSV
                    segment_comparison.to_csv(f"{output_dir}/segment_comparison.csv", index=False)
        except Exception as e:
            print(f"Error in segment comparison analysis: {e}")
            segment_comparison = None
    
    # 3. Calculate time efficiency (% of time spent driving vs waiting)
    if 'grab_to_fetch' in set(speeds_df['segment']):
        try:
            # Extract grab_to_fetch segments
            grab_fetch_df = speeds_df[speeds_df['segment'] == 'grab_to_fetch']
            
            # Group by courier
            courier_efficiency = grab_fetch_df.groupby('courier_id').agg({
                'time_hours': 'sum',  # Total time spent on grab_to_fetch
                'distance_km': 'sum'  # Total distance traveled
            }).reset_index()
            
            # Calculate total driving time assuming average speed of 20 km/h
            avg_speed = 20  # km/h
            courier_efficiency['estimated_driving_hours'] = courier_efficiency['distance_km'] / avg_speed
            
            # Calculate waiting time (difference between total time and driving time)
            courier_efficiency['estimated_waiting_hours'] = courier_efficiency['time_hours'] - courier_efficiency['estimated_driving_hours']
            courier_efficiency['estimated_waiting_hours'] = courier_efficiency['estimated_waiting_hours'].clip(lower=0)
            
            # Calculate efficiency (% of time spent driving)
            courier_efficiency['driving_efficiency'] = (courier_efficiency['estimated_driving_hours'] / 
                                                      courier_efficiency['time_hours'] * 100).clip(upper=100)
            
            print("\nTime Efficiency of Couriers (higher % = more time driving vs waiting):")
            print(courier_efficiency.sort_values('driving_efficiency', ascending=False).head(5))
            
            # Save efficiency data to CSV
            courier_efficiency.to_csv(f"{output_dir}/courier_efficiency.csv", index=False)
        except Exception as e:
            print(f"Error in courier efficiency analysis: {e}")
    
    return courier_consistency, segment_comparison

def plot_speed_visualizations(speeds_df, output_dir):
    """Create visualizations for speed data analysis"""
    print("\nCreating speed visualizations...")
    
    if len(speeds_df) == 0:
        print("No speed data to visualize!")
        return
    
    try:
        # Set up a figure with multiple subplots
        plt.figure(figsize=(16, 12))
        
        # 1. Overall Speed Distribution
        plt.subplot(2, 2, 1)
        sns.histplot(speeds_df['speed_kmh'], bins=30, kde=True)
        plt.title('Distribution of Vehicle Speeds')
        plt.xlabel('Speed (km/h)')
        plt.ylabel('Frequency')
        
        # 2. Speed Distribution by Segment Type
        plt.subplot(2, 2, 2)
        if len(speeds_df['segment'].unique()) > 1:
            sns.boxplot(x='segment', y='speed_kmh', data=speeds_df)
            plt.title('Speed Distribution by Trip Segment')
            plt.xlabel('Segment Type')
            plt.ylabel('Speed (km/h)')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, "Only one segment type available", ha='center', va='center')
            plt.title('Speed Distribution by Trip Segment')
        
        # 3. Scatter plot of distance vs. time
        plt.subplot(2, 2, 3)
        plt.scatter(speeds_df['distance_km'], speeds_df['time_hours'], alpha=0.3)
        plt.title('Distance vs. Time for Each Trip Segment')
        plt.xlabel('Distance (km)')
        plt.ylabel('Time (hours)')
        
        # 4. Speed Distribution by Courier (top 10 couriers with most data points)
        plt.subplot(2, 2, 4)
        top_couriers = speeds_df['courier_id'].value_counts().head(10).index
        courier_speeds = speeds_df[speeds_df['courier_id'].isin(top_couriers)]
        
        if len(courier_speeds) > 0:
            sns.boxplot(x='courier_id', y='speed_kmh', data=courier_speeds)
            plt.title('Speed Distribution for Top 10 Couriers')
            plt.xlabel('Courier ID')
            plt.ylabel('Speed (km/h)')
            plt.xticks(rotation=90)
        else:
            plt.text(0.5, 0.5, "Not enough courier data", ha='center', va='center')
            plt.title('Speed Distribution by Courier')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/speed_distribution.png", dpi=300)
        
        # 5. Speed by time of day (additional analysis)
        if 'timestamp' in speeds_df.columns:
            plt.figure(figsize=(12, 6))
            
            # Extract hour from timestamp
            speeds_df['hour'] = speeds_df['timestamp'].dt.hour
            
            # Calculate average speed by hour
            hourly_speeds = speeds_df.groupby('hour')['speed_kmh'].mean().reset_index()
            
            if len(hourly_speeds) > 0:
                # Plot hourly speeds
                plt.plot(hourly_speeds['hour'], hourly_speeds['speed_kmh'], marker='o', linewidth=2)
                plt.title('Average Speed by Hour of Day')
                plt.xlabel('Hour of Day')
                plt.ylabel('Average Speed (km/h)')
                plt.xticks(range(0, 24))
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(f"{output_dir}/hourly_speed.png", dpi=300)
        
        print(f"Visualizations saved to {output_dir}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def visualize_advanced_metrics(consistency_df, segment_comparison_df, output_dir):
    """Create visualizations for advanced metrics"""
    print("\nCreating advanced metrics visualizations...")
    
    try:
        plt.figure(figsize=(12, 10))
        
        # 1. Speed Consistency Visualization
        plt.subplot(2, 1, 1)
        if consistency_df is not None and len(consistency_df) > 0:
            consistency_sorted = consistency_df.sort_values('coefficient_of_variation').head(15)
            plt.bar(consistency_sorted['courier_id'].astype(str), consistency_sorted['coefficient_of_variation'])
            plt.title('Courier Speed Consistency (Lower = More Consistent)')
            plt.xlabel('Courier ID')
            plt.ylabel('Coefficient of Variation')
            plt.xticks(rotation=90)
        else:
            plt.text(0.5, 0.5, "No consistency data available", ha='center', va='center')
            plt.title('Courier Speed Consistency')
        
        # 2. Pickup vs. Delivery Speed Ratio (if data available)
        plt.subplot(2, 1, 2)
        if (segment_comparison_df is not None and 
            len(segment_comparison_df) > 0 and 
            'delivery_pickup_ratio' in segment_comparison_df.columns):
            
            ratios_sorted = segment_comparison_df.sort_values('delivery_pickup_ratio', ascending=False).head(15)
            
            # Plot a horizontal line at y=1 (equal speeds)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
            
            plt.bar(ratios_sorted['courier_id'].astype(str), ratios_sorted['delivery_pickup_ratio'])
            plt.title('Delivery vs. Pickup Speed Ratio (>1 = Faster during delivery)')
            plt.xlabel('Courier ID')
            plt.ylabel('Delivery/Pickup Speed Ratio')
            plt.xticks(rotation=90)
        else:
            plt.text(0.5, 0.5, "No segment comparison data available", ha='center', va='center')
            plt.title('Delivery vs. Pickup Speed Ratio')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/advanced_metrics.png", dpi=300)
        
        print(f"Advanced visualizations saved to {output_dir}")
    except Exception as e:
        print(f"Error creating advanced visualizations: {e}")

def analyze_speeds_with_weighting(speeds_df):
    """Analyze speed data with different weighting methods"""
    
    # Simple arithmetic mean (current method)
    simple_mean = speeds_df['speed_kmh'].mean()
    
    # Distance-weighted average
    total_distance = speeds_df['distance_km'].sum()
    total_time = speeds_df['time_hours'].sum()
    weighted_avg_speed = total_distance / total_time
    
    # Filter by minimum distance
    min_distance = 0.5  # km
    filtered_df = speeds_df[speeds_df['distance_km'] >= min_distance]
    filtered_mean = filtered_df['speed_kmh'].mean()
    
    print("\nComparison of Speed Calculation Methods:")
    print(f"Simple arithmetic mean: {simple_mean:.2f} km/h")
    print(f"Distance-weighted average: {weighted_avg_speed:.2f} km/h")
    print(f"Mean of trips >= {min_distance} km: {filtered_mean:.2f} km/h")
    
    # Analyze by segment type with weighting
    print("\nDistance-Weighted Speed by Segment Type:")
    segment_weighted_speeds = {}
    
    for segment in speeds_df['segment'].unique():
        segment_df = speeds_df[speeds_df['segment'] == segment]
        segment_dist = segment_df['distance_km'].sum()
        segment_time = segment_df['time_hours'].sum()
        segment_weighted_speed = segment_dist / segment_time
        segment_weighted_speeds[segment] = segment_weighted_speed
        print(f"{segment}: {segment_weighted_speed:.2f} km/h")
    
    return weighted_avg_speed, segment_weighted_speeds



# def main(waybill_file, rider_file):
#     """Main function to perform the complete analysis"""
#     # Create timestamp for this analysis
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     analysis_dir = f"{output_dir}/analysis_{timestamp}"
#     os.makedirs(analysis_dir, exist_ok=True)
    
#     print(f"Starting Meituan vehicle speed analysis. Results will be saved to {analysis_dir}")
    
#     # Load and prepare data
#     waybill_data, rider_data = load_and_prepare_data(waybill_file, rider_file)
    
#     # Calculate speeds
#     waybill_speeds = calculate_speeds_from_waybill(waybill_data)
#     rider_speeds = calculate_speeds_from_rider_data(rider_data)
    
#     # Combine results
#     all_speeds = pd.concat([waybill_speeds, rider_speeds], ignore_index=True)
#     all_speeds.to_csv(f"{analysis_dir}/all_calculated_speeds.csv", index=False)
    
#     # Filter unrealistic values
#     realistic_speeds = filter_realistic_speeds(all_speeds)
#     realistic_speeds.to_csv(f"{analysis_dir}/realistic_speeds.csv", index=False)
    
#     # Basic speed analysis
#     speed_stats = analyze_speeds(realistic_speeds)
    
#     # Advanced courier analysis
#     consistency_df, segment_comparison_df = advanced_courier_analysis(realistic_speeds)
    
#     # Create visualizations
#     plot_speed_visualizations(realistic_speeds, analysis_dir)
#     visualize_advanced_metrics(consistency_df, segment_comparison_df, analysis_dir)
    
#     # Save overall statistics to a summary file
#     with open(f"{analysis_dir}/analysis_summary.txt", "w") as f:
#         f.write("Meituan Vehicle Speed Analysis Summary\n")
#         f.write("=====================================\n\n")
#         f.write(f"Analysis Date: {timestamp}\n")
#         f.write(f"Total waybill records analyzed: {len(waybill_data)}\n")
#         f.write(f"Total rider records analyzed: {len(rider_data)}\n")
#         f.write(f"Total speed records calculated: {len(all_speeds)}\n")
#         f.write(f"Realistic speed records used: {len(realistic_speeds)}\n\n")
        
#         f.write("Speed Statistics (km/h):\n")
#         f.write(f"Mean speed: {speed_stats.get('mean_speed', 'N/A'):.2f}\n")
#         f.write(f"Median speed: {speed_stats.get('median_speed', 'N/A'):.2f}\n") 
#         f.write(f"Standard deviation: {speed_stats.get('std_dev', 'N/A'):.2f}\n")
#         f.write(f"25th percentile: {speed_stats.get('q25', 'N/A'):.2f}\n")
#         f.write(f"75th percentile: {speed_stats.get('q75', 'N/A'):.2f}\n")
    
#     print(f"\nAnalysis complete! All results saved to {analysis_dir}")
#     return realistic_speeds, analysis_dir



def main(waybill_file, rider_file):
    """Main function to perform the complete analysis"""
    # Create timestamp for this analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = f"{output_dir}/analysis_{timestamp}"
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"Starting Meituan vehicle speed analysis. Results will be saved to {analysis_dir}")
    
    # Load and prepare data
    waybill_data, rider_data = load_and_prepare_data(waybill_file, rider_file)
    
    # Calculate speeds
    waybill_speeds = calculate_speeds_from_waybill(waybill_data)
    rider_speeds = calculate_speeds_from_rider_data(rider_data)
    
    # Combine results
    all_speeds = pd.concat([waybill_speeds, rider_speeds], ignore_index=True)
    all_speeds.to_csv(f"{analysis_dir}/all_calculated_speeds.csv", index=False)
    
    # Filter unrealistic values
    realistic_speeds = filter_realistic_speeds(all_speeds)
    realistic_speeds.to_csv(f"{analysis_dir}/realistic_speeds.csv", index=False)
    
    # Basic speed analysis
    speed_stats = analyze_speeds(realistic_speeds)
    
    # Advanced courier analysis
    consistency_df, segment_comparison_df = advanced_courier_analysis(realistic_speeds)
    
    # New: Calculate weighted average speeds
    weighted_avg_speed, segment_weighted_speeds = analyze_speeds_with_weighting(realistic_speeds)
    
    # Create visualizations
    plot_speed_visualizations(realistic_speeds, analysis_dir)
    visualize_advanced_metrics(consistency_df, segment_comparison_df, analysis_dir)
    
    # Save overall statistics to a summary file
    with open(f"{analysis_dir}/analysis_summary.txt", "w") as f:
        f.write("Meituan Vehicle Speed Analysis Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Analysis Date: {timestamp}\n")
        f.write(f"Total waybill records analyzed: {len(waybill_data)}\n")
        f.write(f"Total rider records analyzed: {len(rider_data)}\n")
        f.write(f"Total speed records calculated: {len(all_speeds)}\n")
        f.write(f"Realistic speed records used: {len(realistic_speeds)}\n\n")
        
        f.write("Speed Statistics (km/h):\n")
        f.write(f"Mean speed: {speed_stats.get('mean_speed', 'N/A'):.2f}\n")
        f.write(f"Median speed: {speed_stats.get('median_speed', 'N/A'):.2f}\n") 
        f.write(f"Standard deviation: {speed_stats.get('std_dev', 'N/A'):.2f}\n")
        f.write(f"25th percentile: {speed_stats.get('q25', 'N/A'):.2f}\n")
        f.write(f"75th percentile: {speed_stats.get('q75', 'N/A'):.2f}\n\n")
        
        # Add new weighted speed statistics
        f.write("Distance-Weighted Speed Statistics (km/h):\n")
        f.write(f"Distance-weighted average speed: {weighted_avg_speed:.2f}\n\n")
        
        f.write("Distance-Weighted Speed by Segment Type (km/h):\n")
        for segment, speed in segment_weighted_speeds.items():
            f.write(f"{segment}: {speed:.2f}\n")
        
        # Add the filtered speed results (for trips > 0.5 km)
        min_distance = 0.5  # km
        filtered_df = realistic_speeds[realistic_speeds['distance_km'] >= min_distance]
        filtered_mean = filtered_df['speed_kmh'].mean()
        f.write(f"\nAverage speed for trips >= {min_distance} km: {filtered_mean:.2f}\n")
    
    print(f"\nAnalysis complete! All results saved to {analysis_dir}")
    return realistic_speeds, analysis_dir










if __name__ == "__main__":
    # Define the file paths
    waybill_file = r"C:\Users\trika\Desktop\Masterarbeit\Thesis_Modell\thesis\data\meituan_data\all_waybill_info_meituan_0322.csv"
    rider_file = r"C:\Users\trika\Desktop\Masterarbeit\Thesis_Modell\thesis\data\meituan_data\dispatch_rider_meituan.csv"
    
    # Run the analysis with the defined file paths
    main(waybill_file, rider_file)