import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_restaurant_order_volume(df, output_dir):
    """
    Visualize order volume for the top 20 restaurants by order count.
    
    Args:
        df: DataFrame with order data (all_waybill_info_meituan.csv)
        output_dir: Directory to save visualizations
    """
    print("Analyzing restaurant order volume...")

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Count orders per restaurant (poi_id)
    restaurant_counts = df['poi_id'].value_counts().head(20)

    # Calculate top 10% contribution
    top_10_percent = int(len(df['poi_id'].unique()) * 0.1)
    top_10_counts = df['poi_id'].value_counts().head(top_10_percent)
    top_10_percentage = (top_10_counts.sum() / len(df)) * 100

    plt.figure(figsize=(12, 6))
    sns.barplot(x=restaurant_counts.index.astype(str), y=restaurant_counts.values, color='skyblue')
    plt.xlabel('Restaurant ID')
    plt.ylabel('Number of Orders')
    plt.title('Order Volume for Top 20 Restaurants')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "restaurant_order_volume.pdf"), dpi=300, format='pdf')
    plt.close()

    print(f"Top 10% ({top_10_percent}) restaurants account for {top_10_percentage:.1f}% of orders")

def plot_orders_per_courier(df, output_dir):
    """
    Visualize the distribution of orders per courier.
    
    Args:
        df: DataFrame with order data (all_waybill_info_meituan.csv)
        output_dir: Directory to save visualizations
    """
    print("Analyzing orders per courier...")

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Count orders per courier
    orders_per_courier = df.groupby('courier_id')['waybill_id'].count()

    plt.figure(figsize=(10, 6))
    sns.histplot(orders_per_courier, bins=30, kde=True, color='skyblue', stat='count')
    plt.xlabel('Orders per Courier')
    plt.ylabel('Number of Couriers')
    plt.title('Distribution of Orders per Courier')
    plt.grid(True, alpha=0.3)
    plt.axvline(orders_per_courier.mean(), color='red', linestyle='--', 
                label=f'Mean: {orders_per_courier.mean():.1f}')
    plt.axvline(orders_per_courier.median(), color='green', linestyle='-.', 
                label=f'Median: {orders_per_courier.median():.1f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "orders_per_courier.pdf"), dpi=300, format='pdf')
    plt.close()

    print(f"Orders per courier: Mean={orders_per_courier.mean():.1f}, Median={orders_per_courier.median():.1f}")

def analyze_order_allocation_and_restaurant_activity(df, output_dir):
    """
    Combined analysis of restaurant order volume and orders per courier.
    
    Args:
        df: DataFrame with order data
        output_dir: Directory to save visualizations
    """
    plot_restaurant_order_volume(df, output_dir)
    plot_orders_per_courier(df, output_dir)

# Example usage
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("data/meituan_data/all_waybill_info_meituan_0322.csv")  # Replace with your actual file path
    
    # Set output directory
    output_dir = "data/meituan_data/abb"
    
    # Run combined analysis
    analyze_order_allocation_and_restaurant_activity(df, output_dir)