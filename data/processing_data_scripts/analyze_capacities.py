# analyze_vehicle_capacities.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def analyze_vehicle_capacities(wave_file_path, output_dir):
    """
    Analyze and infer vehicle capacities from courier wave data
    
    Args:
        wave_file_path: Path to the courier_wave_info file
        output_dir: Directory to save results
        
    Returns:
        Recommended vehicle capacity for simulation
    """
    logger.info(f"Analyzing vehicle capacities from {wave_file_path}...")
    
    # Check if file exists
    if not os.path.exists(wave_file_path):
        logger.warning(f"File not found: {wave_file_path}")
        return None
    
    # Load wave data
    try:
        df = pd.read_csv(wave_file_path)
    except Exception as e:
        logger.error(f"Error loading wave data: {str(e)}")
        return None
    
    # Check if we have the necessary data
    if 'order_ids' not in df.columns or 'courier_id' not in df.columns:
        logger.warning("Missing required columns for vehicle capacity analysis")
        return None
    
    logger.info(f"Loaded {len(df)} courier waves")
    
    # Extract the number of orders in each wave
    df['order_count'] = df['order_ids'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).strip() else 0
    )
    
    # Filter out empty waves (0 orders)
    df = df[df['order_count'] > 0]
    
    if df.empty:
        logger.warning("No valid wave data found")
        return None
    
    # Calculate statistics per courier
    courier_stats = df.groupby('courier_id').agg({
        'order_count': ['max', 'mean', 'median', 'count']
    }).reset_index()
    
    courier_stats.columns = ['courier_id', 'max_orders', 'avg_orders', 'median_orders', 'wave_count']
    
    # Filter couriers with too few waves (at least 3)
    courier_stats = courier_stats[courier_stats['wave_count'] >= 3]
    
    if courier_stats.empty:
        logger.warning("No couriers with sufficient data found")
        return None
    
    # Calculate overall statistics
    max_capacity = courier_stats['max_orders'].max()
    avg_max_capacity = courier_stats['max_orders'].mean()
    median_max_capacity = courier_stats['max_orders'].median()
    
    # Create visualizations directory if it doesn't exist
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize the distribution of maximum orders per courier
    plt.figure(figsize=(12, 6))
    plt.hist(courier_stats['max_orders'], bins=range(1, max_capacity + 2), alpha=0.7)
    plt.axvline(x=avg_max_capacity, color='r', linestyle='--', 
               label=f'Mean: {avg_max_capacity:.1f}')
    plt.axvline(x=median_max_capacity, color='g', linestyle='-.', 
               label=f'Median: {median_max_capacity:.1f}')
    
    plt.title('Maximum Orders Per Courier (Inferred Vehicle Capacity)')
    plt.xlabel('Maximum Orders')
    plt.ylabel('Number of Couriers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, "vehicle_capacity_distribution.png"), dpi=300)
    plt.close()
    
    # Create a more detailed view with boxplot
    plt.figure(figsize=(12, 6))
    plt.boxplot(courier_stats['max_orders'], vert=False)
    plt.title('Distribution of Maximum Orders Per Courier')
    plt.xlabel('Maximum Orders')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, "vehicle_capacity_boxplot.png"), dpi=300)
    plt.close()
    
    # Print summary
    logger.info(f"Vehicle capacity analysis:")
    logger.info(f"  Number of couriers analyzed: {len(courier_stats)}")
    logger.info(f"  Maximum orders per courier: {max_capacity}")
    logger.info(f"  Average maximum orders: {avg_max_capacity:.1f}")
    logger.info(f"  Median maximum orders: {median_max_capacity:.1f}")
    
    # Calculate percentiles for better insight
    percentiles = [50, 75, 90, 95, 99]
    percentile_values = np.percentile(courier_stats['max_orders'], percentiles)
    
    logger.info("  Percentiles of maximum orders:")
    for p, v in zip(percentiles, percentile_values):
        logger.info(f"    {p}th percentile: {v:.1f}")
    
    # Save detailed stats
    courier_stats.to_csv(os.path.join(output_dir, "courier_capacity_stats.csv"), index=False)
    
    # Add capacity to simulation parameters - use 90th percentile as recommended
    recommended_capacity = int(np.ceil(np.percentile(courier_stats['max_orders'], 90)))
    logger.info(f"  Recommended vehicle capacity for simulation: {recommended_capacity}")
    
    # Additional analysis: most common maximum
    value_counts = courier_stats['max_orders'].value_counts()
    most_common_max = value_counts.idxmax()
    logger.info(f"  Most common maximum orders: {most_common_max} (occurs for {value_counts[most_common_max]} couriers)")
    
    # Save capacity recommendation to a config file
    with open(os.path.join(output_dir, "recommended_capacity.txt"), 'w') as f:
        f.write(f"Recommended vehicle capacity: {recommended_capacity}\n")
        f.write(f"Based on analysis of {len(courier_stats)} couriers\n")
        f.write(f"90th percentile of maximum orders per courier\n")
        f.write(f"Maximum observed: {max_capacity}\n")
        f.write(f"Average maximum: {avg_max_capacity:.1f}\n")
        f.write(f"Median maximum: {median_max_capacity:.1f}\n")
    
    return recommended_capacity

def main():
    # Set paths
    data_dir = "data/meituan_data"
    wave_file = os.path.join(data_dir, "courier_wave_info_meituan.csv")
    output_dir = os.path.join(data_dir, "processed")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the analysis
    recommended_capacity = analyze_vehicle_capacities(wave_file, output_dir)
    
    # Print final result
    if recommended_capacity:
        logger.info(f"Analysis complete. Recommended vehicle capacity: {recommended_capacity}")
    else:
        logger.warning("Analysis failed or no valid data was found.")

if __name__ == "__main__":
    main()

