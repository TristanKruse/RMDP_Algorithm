import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_courier_capacity(dispatch_file, output_dir):
    """
    Analyze the capacity of couriers using the dispatch_rider_meituan.csv dataset.
    
    This function:
    1. Loads the dispatch data and counts the number of on-hand orders per courier at each dispatch time.
    2. Calculates summary statistics (mean, median, max, percentiles) for the number of on-hand orders.
    3. Creates a histogram to visualize the distribution of on-hand orders.
    4. Investigates high on-hand order counts for potential anomalies.
    
    Args:
        dispatch_file: Path to the dispatch_rider_meituan.csv file
        output_dir: Directory to save visualizations
    
    Returns:
        None
    """
    print(f"Loading data from {dispatch_file}...")
    
    # Create output directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Load the dataset
    try:
        df = pd.read_csv(dispatch_file)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Step 1: Count the number of on-hand orders
    # The 'courier_waybills' column contains a set of order IDs (e.g., "(57635, 57636, 57637)")
    # We need to count the number of orders in each set
    df['num_orders'] = df['courier_waybills'].apply(lambda x: len(eval(x)) if pd.notna(x) else 0)

    # Step 2: Investigate high on-hand order counts
    high_order_threshold = 5  # Threshold for considering counts as high
    high_order_records = df[df['num_orders'] > high_order_threshold]
    print(f"\nRecords with high on-hand order counts (> {high_order_threshold} orders): {len(high_order_records)}")
    if len(high_order_records) > 0:
        print("Details of high on-hand order records:")
        print(high_order_records[['courier_id', 'dispatch_time', 'num_orders', 'courier_waybills']].head(10))

    # Check for duplicates in courier_waybills sets
    def check_duplicates(order_set):
        if pd.isna(order_set):
            return False
        orders = eval(order_set)
        return len(orders) != len(set(orders))

    df['has_duplicates'] = df['courier_waybills'].apply(check_duplicates)
    duplicate_records = df[df['has_duplicates']]
    print(f"\nRecords with duplicate orders in courier_waybills: {len(duplicate_records)}")
    if len(duplicate_records) > 0:
        print("Details of records with duplicates:")
        print(duplicate_records[['courier_id', 'dispatch_time', 'num_orders', 'courier_waybills']].head(10))

    # Step 3: Calculate summary statistics
    num_orders = df['num_orders']
    stats = {
        'mean': num_orders.mean(),
        'median': num_orders.median(),
        'std': num_orders.std(),
        'min': num_orders.min(),
        'max': num_orders.max(),
        'p90': np.percentile(num_orders, 90),
        'p95': np.percentile(num_orders, 95),
        'count': len(num_orders)
    }

    print("\nSummary Statistics for Number of On-Hand Orders per Courier at Dispatch Time:")
    print(f"Total records: {stats['count']}")
    print(f"Mean number of orders: {stats['mean']:.2f}")
    print(f"Median number of orders: {stats['median']:.2f}")
    print(f"Standard deviation: {stats['std']:.2f}")
    print(f"Minimum number of orders: {stats['min']}")
    print(f"Maximum number of orders: {stats['max']}")
    print(f"90th percentile: {stats['p90']}")
    print(f"95th percentile: {stats['p95']}")

    # Step 4: Visualize the distribution of on-hand orders
    plt.figure(figsize=(10, 6))
    sns.histplot(num_orders, bins=range(int(num_orders.min()), int(num_orders.max()) + 2), color='blue', alpha=0.5, edgecolor='black')
    plt.xlabel('Number of On-Hand Orders')
    plt.ylabel('Frequency')
    plt.title('Distribution of On-Hand Orders per Courier at Dispatch Time')
    plt.grid(True, alpha=0.3)

    # Add vertical lines for mean, median, and 90th percentile
    plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.1f}')
    plt.axvline(stats['median'], color='green', linestyle='-.', label=f'Median: {stats["median"]:.1f}')
    plt.axvline(stats['p90'], color='black', linestyle='-', label=f'90th Percentile: {stats["p90"]:.1f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "on_hand_orders_distribution.pdf"), dpi=300, format='pdf')
    plt.close()

    print(f"Visualization saved to {viz_dir}/on_hand_orders_distribution.pdf")

def main():
    """
    Main function to run the courier capacity analysis.
    """
    # Define input and output paths
    dispatch_file = "data/meituan_data/dispatch_rider_meituan.csv"
    output_dir = "data/meituan_data/abb"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the analysis
    analyze_courier_capacity(dispatch_file, output_dir)
    print("Courier capacity analysis complete!")

if __name__ == "__main__":
    main()