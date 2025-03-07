import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

def visualize_restaurant_order_distribution(input_dir, output_dir):
    """
    Create visualizations showing the distribution of orders across restaurants
    for each district and day in the Meituan dataset.
    
    Args:
        input_dir: Directory containing the daily order data
        output_dir: Directory where visualizations will be saved
    """
    print("Generating restaurant order distribution visualizations...")
    
    # Create output directory for restaurant visualizations
    viz_dir = os.path.join(output_dir, "restaurant_distribution")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Navigate through the daily_orders directory structure
    daily_orders_dir = os.path.join(input_dir, "daily_orders")
    
    if not os.path.exists(daily_orders_dir):
        print(f"Error: {daily_orders_dir} does not exist")
        return
    
    # Process each day folder
    for day in os.listdir(daily_orders_dir):
        day_dir = os.path.join(daily_orders_dir, day)
        
        # Skip if not a directory
        if not os.path.isdir(day_dir):
            continue
            
        print(f"Processing day: {day}")
        
        # Find all order CSV files for this day
        order_files = [f for f in os.listdir(day_dir) if f.endswith('_orders.csv')]
        
        for order_file in order_files:
            # Extract district ID from filename
            district_id = order_file.split('_')[1]
            file_path = os.path.join(day_dir, order_file)
            
            try:
                # Load order data
                df = pd.read_csv(file_path)
                
                # Skip if empty
                if df.empty:
                    print(f"  Skipping empty file: {file_path}")
                    continue
                
                # Define restaurants by their coordinates and ID
                if 'sender_lat' in df.columns and 'sender_lng' in df.columns and 'poi_id' in df.columns:
                    # Create a unique identifier for each restaurant
                    df['restaurant_id'] = df['poi_id']
                    
                    # Count orders per restaurant
                    restaurant_counts = df['restaurant_id'].value_counts()
                    
                    # Create visualization if we have data
                    if len(restaurant_counts) > 0:
                        print(f"  Creating visualization for district {district_id} with {len(restaurant_counts)} restaurants")
                        
                        # Create bar chart for top 30 restaurants (or fewer if less than 30 restaurants)
                        plt.figure(figsize=(12, 8))
                        
                        # Get top 30 restaurants by order count
                        top_restaurants = restaurant_counts.nlargest(30)
                        
                        # Create bar chart
                        bars = plt.bar(range(len(top_restaurants)), top_restaurants.values)
                        
                        # Add labels and title
                        plt.title(f'Top Restaurants by Order Count - District {district_id}, Day {day}')
                        plt.xlabel('Restaurant (ID)')
                        plt.ylabel('Number of Orders')
                        
                        # Add restaurant IDs as x-tick labels
                        plt.xticks(range(len(top_restaurants)), top_restaurants.index, rotation=90)
                        
                        # Add count labels on top of bars
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                    f'{height:.0f}',
                                    ha='center', va='bottom', rotation=0)
                        
                        plt.tight_layout()
                        
                        # Save visualization
                        viz_file = os.path.join(viz_dir, f"district_{district_id}_day_{day}_restaurant_distribution.png")
                        plt.savefig(viz_file, dpi=300)
                        plt.close()
                        
                        # Create a secondary visualization showing distribution statistics
                        plt.figure(figsize=(10, 6))
                        
                        # Calculate distribution statistics
                        order_stats = {
                            '1 order': sum(restaurant_counts == 1),
                            '2-5 orders': sum((restaurant_counts >= 2) & (restaurant_counts <= 5)),
                            '6-10 orders': sum((restaurant_counts >= 6) & (restaurant_counts <= 10)),
                            '11-20 orders': sum((restaurant_counts >= 11) & (restaurant_counts <= 20)),
                            '21-50 orders': sum((restaurant_counts >= 21) & (restaurant_counts <= 50)),
                            '51+ orders': sum(restaurant_counts > 50)
                        }
                        
                        # Create bar chart for distribution
                        categories = list(order_stats.keys())
                        values = list(order_stats.values())
                        
                        bars = plt.bar(categories, values)
                        
                        # Add count labels on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            if height > 0:
                                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                        f'{height:.0f}',
                                        ha='center', va='bottom')
                        
                        plt.title(f'Restaurant Order Volume Distribution - District {district_id}, Day {day}')
                        plt.xlabel('Order Volume Category')
                        plt.ylabel('Number of Restaurants')
                        plt.tight_layout()
                        
                        # Save second visualization
                        viz_file = os.path.join(viz_dir, f"district_{district_id}_day_{day}_restaurant_volume_categories.png")
                        plt.savefig(viz_file, dpi=300)
                        plt.close()
                        
                    else:
                        print(f"  No restaurant data found for district {district_id}")
                
                else:
                    print(f"  Missing required columns in {order_file}")
            
            except Exception as e:
                print(f"  Error processing {file_path}: {str(e)}")

def main():
    # Set input and output paths
    data_dir = "data/meituan_data"
    input_dir = os.path.join(data_dir, "processed")
    output_dir = os.path.join(data_dir, "processed", "visualizations")
    
    # Create restaurant order distribution visualizations
    visualize_restaurant_order_distribution(input_dir, output_dir)
    
    print("Restaurant order distribution visualization complete!")

if __name__ == "__main__":
    main()