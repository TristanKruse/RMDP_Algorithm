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
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
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
    for col in ['grab_time', 'fetch_time', 'arrive_time', 'estimate_meal_prepare_time']:
        if col in waybill_data.columns:
            waybill_data[col] = pd.to_datetime(waybill_data[col], unit='s', errors='coerce')
    
    if 'dispatch_time' in rider_data.columns:
        rider_data['dispatch_time'] = pd.to_datetime(rider_data['dispatch_time'], unit='s', errors='coerce')
    
    print(f"Loaded {len(waybill_data)} waybill records and {len(rider_data)} rider records")
    return waybill_data, rider_data

def estimate_service_times(waybill_data, initial_speed=15):
    """Estimate service times for pickup and drop-off using close-proximity orders."""
    print("Estimating service times...")

    # 1. Drop-Off Service Time (Restaurant-to-Customer)
    valid_fetch_arrive = waybill_data[
        (waybill_data['sender_lat'] != 0) & 
        (waybill_data['sender_lng'] != 0) &
        (waybill_data['recipient_lat'] != 0) & 
        (waybill_data['recipient_lng'] != 0) & 
        (waybill_data['fetch_time'].notna()) &
        (waybill_data['arrive_time'].notna())
    ]

    close_proximity = []
    for _, row in valid_fetch_arrive.iterrows():
        sender_lat = row['sender_lat'] / 1000000
        sender_lng = row['sender_lng'] / 1000000
        recipient_lat = row['recipient_lat'] / 1000000
        recipient_lng = row['recipient_lng'] / 1000000
        
        distance = haversine_distance(sender_lat, sender_lng, recipient_lat, recipient_lng)
        if distance < 0.1:  # Less than 100 meters
            time_diff = (row['arrive_time'] - row['fetch_time']).total_seconds() / 60  # in minutes
            if 0 < time_diff < 10:  # Reasonable service time (0–10 minutes)
                close_proximity.append(time_diff)

    dropoff_service_time = np.mean(close_proximity) if close_proximity else 2.0
    dropoff_service_std = np.std(close_proximity) if close_proximity else 0.0
    print(f"Estimated drop-off service time: {dropoff_service_time:.2f} minutes (std: {dropoff_service_std:.2f}, based on {len(close_proximity)} close-proximity orders)")

    # 2. Pickup Service Time (Grab-to-Restaurant)
    valid_grab_fetch = waybill_data[
        (waybill_data['grab_lat'] != 0) & 
        (waybill_data['grab_lng'] != 0) & 
        (waybill_data['sender_lat'] != 0) & 
        (waybill_data['sender_lng'] != 0) & 
        (waybill_data['grab_time'].notna()) &
        (waybill_data['fetch_time'].notna()) &
        (waybill_data['estimate_meal_prepare_time'].notna())
    ]

    pickup_service_times = []
    for _, row in valid_grab_fetch.iterrows():
        grab_lat = row['grab_lat'] / 1000000
        grab_lng = row['grab_lng'] / 1000000
        sender_lat = row['sender_lat'] / 1000000
        sender_lng = row['sender_lng'] / 1000000
        
        distance = haversine_distance(grab_lat, grab_lng, sender_lat, sender_lng)
        travel_time_hours = distance / initial_speed  # Using initial speed assumption
        travel_time_seconds = travel_time_hours * 3600
        
        arrival_time = row['grab_time'] + pd.Timedelta(seconds=travel_time_seconds)
        
        if arrival_time > row['estimate_meal_prepare_time']:
            handover_time = (row['fetch_time'] - arrival_time).total_seconds() / 60  # in minutes
            if 0 < handover_time < 10:  # Reasonable service time (0–10 minutes)
                pickup_service_times.append(handover_time)

    pickup_service_time = np.mean(pickup_service_times) if pickup_service_times else 5.0
    pickup_service_std = np.std(pickup_service_times) if pickup_service_times else 0.0
    print(f"Estimated pickup service time: {pickup_service_time:.2f} minutes (std: {pickup_service_std:.2f}, based on {len(pickup_service_times)} orders)")

    return pickup_service_time, dropoff_service_time, pickup_service_std, dropoff_service_std, pickup_service_times, close_proximity

def calculate_speeds_from_waybill(waybill_data, pickup_service_time=5.0, dropoff_service_time=2.0):
    """Calculate speeds from waybill data, adjusting for service times."""
    print("Calculating speeds from waybill data...")

    pickup_service_time_hours = pickup_service_time / 60
    dropoff_service_time_hours = dropoff_service_time / 60

    speeds = []
    
    # Grab-to-Restaurant Segment
    valid_grab_fetch = waybill_data[
        (waybill_data['grab_lat'] != 0) & 
        (waybill_data['grab_lng'] != 0) & 
        (waybill_data['sender_lat'] != 0) & 
        (waybill_data['sender_lng'] != 0) & 
        (waybill_data['grab_time'].notna()) &
        (waybill_data['fetch_time'].notna())
    ]
    
    print(f"Found {len(valid_grab_fetch)} valid records for grab-to-fetch analysis")
    
    for _, row in valid_grab_fetch.iterrows():
        grab_lat = row['grab_lat'] / 1000000
        grab_lng = row['grab_lng'] / 1000000
        sender_lat = row['sender_lat'] / 1000000
        sender_lng = row['sender_lng'] / 1000000
        
        distance = haversine_distance(grab_lat, grab_lng, sender_lat, sender_lng)
        total_time = (row['fetch_time'] - row['grab_time']).total_seconds() / 3600
        travel_time = total_time - pickup_service_time_hours
        
        if travel_time > 0 and distance > 0.05:
            speed = distance / travel_time
            speeds.append({
                'courier_id': row['courier_id'],
                'distance_km': distance,
                'time_hours': travel_time,
                'speed_kmh': speed,
                'segment': 'grab_to_fetch',
                'timestamp': row['grab_time']
            })
    
    # Restaurant-to-Customer Segment
    valid_fetch_arrive = waybill_data[
        (waybill_data['sender_lat'] != 0) & 
        (waybill_data['sender_lng'] != 0) &
        (waybill_data['recipient_lat'] != 0) & 
        (waybill_data['recipient_lng'] != 0) & 
        (waybill_data['fetch_time'].notna()) &
        (waybill_data['arrive_time'].notna())
    ]
    
    print(f"Found {len(valid_fetch_arrive)} valid records for fetch-to-arrive analysis")
    
    for _, row in valid_fetch_arrive.iterrows():
        sender_lat = row['sender_lat'] / 1000000
        sender_lng = row['sender_lng'] / 1000000
        recipient_lat = row['recipient_lat'] / 1000000
        recipient_lng = row['recipient_lng'] / 1000000
        
        distance = haversine_distance(sender_lat, sender_lng, recipient_lat, recipient_lng)
        total_time = (row['arrive_time'] - row['fetch_time']).total_seconds() / 3600
        travel_time = total_time - dropoff_service_time_hours
        
        if travel_time > 0 and distance > 0.05:
            speed = distance / travel_time
            speeds.append({
                'courier_id': row['courier_id'],
                'distance_km': distance,
                'time_hours': travel_time,
                'speed_kmh': speed,
                'segment': 'fetch_to_arrive',
                'timestamp': row['fetch_time']
            })
    
    speed_df = pd.DataFrame(speeds)
    print(f"Generated {len(speed_df)} speed records from waybill data")
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
    
    # Overall statistics
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
    
    # Distance-weighted average
    total_distance = speeds_df['distance_km'].sum()
    total_time = speeds_df['time_hours'].sum()
    weighted_avg_speed = total_distance / total_time if total_time > 0 else 0.0
    print(f"Distance-weighted average speed: {weighted_avg_speed:.2f} km/h")
    
    # Analyze by segment type
    segment_stats = speeds_df.groupby('segment')['speed_kmh'].agg(['mean', 'median', 'std', 'count']).reset_index()
    print("\nSpeed Statistics by Segment Type:")
    print(segment_stats)
    
    # Save segment stats to CSV
    segment_stats.to_csv(f"{output_dir}/segment_speed_stats.csv", index=False)
    
    return stats, weighted_avg_speed


# def plot_visualizations(speeds_df, pickup_service_times, dropoff_service_times, output_dir):
#     """Create visualizations for speed and service time analysis"""
#     print("\nCreating visualizations...")
    
#     if len(speeds_df) == 0:
#         print("No speed data to visualize!")
#         return
    
#     try:
#         # 1. Speed Distribution by Segment Type (Figure~\ref{fig:courier_speeds})
#         plt.figure(figsize=(8, 6))
#         sns.boxplot(x='segment', y='speed_kmh', data=speeds_df)
#         plt.title('Speed Distribution by Trip Segment')
#         plt.xlabel('Segment Type')
#         plt.ylabel('Speed (km/h)')
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.savefig(f"{output_dir}/courier_speeds.png", dpi=300)
#         plt.close()
        
#         # 2. Service Time Distribution (Figure~\ref{fig:service_times})
#         plt.figure(figsize=(8, 6))
#         if pickup_service_times and dropoff_service_times:
#             # Combine pickup and drop-off service times into one plot with different colors
#             sns.histplot(pickup_service_times, bins=30, kde=True, color='blue', label='Pickup', alpha=0.5)
#             sns.histplot(dropoff_service_times, bins=30, kde=True, color='orange', label='Drop-Off', alpha=0.5)
#             plt.title('Service Time Distribution')
#             plt.xlabel('Service Time (minutes)')
#             plt.ylabel('Frequency')
#             plt.legend()
#         else:
#             plt.text(0.5, 0.5, "No service time data available", ha='center', va='center')
#             plt.title('Service Time Distribution')
#         plt.tight_layout()
#         plt.savefig(f"{output_dir}/service_times.png", dpi=300)
#         plt.close()
        
#         print(f"Visualizations saved to {output_dir}")
#     except Exception as e:
#         print(f"Error creating visualizations: {e}")

def plot_visualizations(speeds_df, pickup_service_times, dropoff_service_times, output_dir):
    """Create visualizations for speed and service time analysis"""
    print("\nCreating visualizations...")
    
    if len(speeds_df) == 0:
        print("No speed data to visualize!")
        return
    
    try:
        # 1. Speed Distribution by Segment Type (Figure~\ref{fig:courier_speeds})
        plt.figure(figsize=(8, 6))
        # Separate the speed data for each segment
        grab_to_fetch_speeds = speeds_df[speeds_df['segment'] == 'grab_to_fetch']['speed_kmh']
        fetch_to_arrive_speeds = speeds_df[speeds_df['segment'] == 'fetch_to_arrive']['speed_kmh']
        # Plot overlaid histograms with different colors, without KDE
        if len(grab_to_fetch_speeds) > 0 and len(fetch_to_arrive_speeds) > 0:
            sns.histplot(grab_to_fetch_speeds, bins=30, color='blue', label='Grab-to-Restaurant', alpha=0.5, kde=False)
            sns.histplot(fetch_to_arrive_speeds, bins=30, color='orange', label='Restaurant-to-Customer', alpha=0.5, kde=False)
            # Add dashed lines for the mean values
            plt.axvline(grab_to_fetch_speeds.mean(), color='blue', linestyle='--', label=f'Grab-to-Restaurant Mean ({grab_to_fetch_speeds.mean():.1f} km/h)')
            plt.axvline(fetch_to_arrive_speeds.mean(), color='orange', linestyle='--', label=f'Restaurant-to-Customer Mean ({fetch_to_arrive_speeds.mean():.1f} km/h)')
            plt.title('Speed Distribution by Trip Segment')
            plt.xlabel('Speed (km/h)')
            plt.ylabel('Frequency')
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No speed data available", ha='center', va='center')
            plt.title('Speed Distribution by Trip Segment')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/courier_speeds.png", dpi=300)
        plt.close()
        
        # 2. Service Time Distribution (Figure~\ref{fig:service_times})
        plt.figure(figsize=(8, 6))
        if pickup_service_times and dropoff_service_times:
            # Combine pickup and drop-off service times into one plot with different colors, without KDE
            sns.histplot(pickup_service_times, bins=30, color='blue', label='Pickup', alpha=0.5, kde=False)
            sns.histplot(dropoff_service_times, bins=30, color='orange', label='Drop-Off', alpha=0.5, kde=False)
            # Add dashed lines for the mean values
            plt.axvline(np.mean(pickup_service_times), color='blue', linestyle='--', label=f'Pickup Mean ({np.mean(pickup_service_times):.1f} minutes)')
            plt.axvline(np.mean(dropoff_service_times), color='orange', linestyle='--', label=f'Drop-Off Mean ({np.mean(dropoff_service_times):.1f} minutes)')
            plt.title('Service Time Distribution')
            plt.xlabel('Service Time (minutes)')
            plt.ylabel('Frequency')
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No service time data available", ha='center', va='center')
            plt.title('Service Time Distribution')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/service_times.png", dpi=300)
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")


def main(waybill_file, rider_file):
    """Main function to perform the analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = f"{output_dir}/analysis_{timestamp}"
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"Starting Meituan vehicle speed analysis. Results will be saved to {analysis_dir}")
    
    # Load and prepare data
    waybill_data, rider_data = load_and_prepare_data(waybill_file, rider_file)
    
    # Estimate service times
    pickup_service_time, dropoff_service_time, pickup_service_std, dropoff_service_std, pickup_service_times, dropoff_service_times = estimate_service_times(waybill_data, initial_speed=15)
    
    # Calculate speeds with adjusted service times
    waybill_speeds = calculate_speeds_from_waybill(waybill_data, pickup_service_time, dropoff_service_time)
    
    # Filter unrealistic values
    realistic_speeds = filter_realistic_speeds(waybill_speeds)
    realistic_speeds.to_csv(f"{analysis_dir}/realistic_speeds.csv", index=False)
    
    # Basic speed analysis
    speed_stats, weighted_avg_speed = analyze_speeds(realistic_speeds)
    
    # Create visualizations
    plot_visualizations(realistic_speeds, pickup_service_times, dropoff_service_times, analysis_dir)
    
    # Save overall statistics to a summary file
    with open(f"{analysis_dir}/analysis_summary.txt", "w") as f:
        f.write("Meituan Vehicle Speed Analysis Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Analysis Date: {timestamp}\n")
        f.write(f"Total waybill records analyzed: {len(waybill_data)}\n")
        f.write(f"Total rider records analyzed: {len(rider_data)}\n")
        f.write(f"Total speed records calculated: {len(waybill_speeds)}\n")
        f.write(f"Realistic speed records used: {len(realistic_speeds)}\n\n")
        
        f.write("Service Time Estimates (minutes):\n")
        f.write(f"Pickup service time: {pickup_service_time:.2f} (std: {pickup_service_std:.2f})\n")
        f.write(f"Drop-off service time: {dropoff_service_time:.2f} (std: {dropoff_service_std:.2f})\n\n")
        
        f.write("Speed Statistics (km/h):\n")
        f.write(f"Mean speed: {speed_stats.get('mean_speed', 'N/A'):.2f}\n")
        f.write(f"Median speed: {speed_stats.get('median_speed', 'N/A'):.2f}\n") 
        f.write(f"Standard deviation: {speed_stats.get('std_dev', 'N/A'):.2f}\n")
        f.write(f"25th percentile: {speed_stats.get('q25', 'N/A'):.2f}\n")
        f.write(f"75th percentile: {speed_stats.get('q75', 'N/A'):.2f}\n")
        f.write(f"Distance-weighted average speed: {weighted_avg_speed:.2f}\n")
    
    print(f"\nAnalysis complete! All results saved to {analysis_dir}")
    return realistic_speeds, analysis_dir

if __name__ == "__main__":
    # Define the file paths
    waybill_file = r"C:\Users\trika\Desktop\Masterarbeit\Thesis_Modell\thesis\data\meituan_data\all_waybill_info_meituan_0322.csv"
    rider_file = r"C:\Users\trika\Desktop\Masterarbeit\Thesis_Modell\thesis\data\meituan_data\dispatch_rider_meituan.csv"
    
    # Run the analysis with the defined file paths
    main(waybill_file, rider_file)