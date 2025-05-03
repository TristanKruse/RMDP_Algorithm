import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2
from matplotlib.ticker import FuncFormatter


def load_and_preprocess_data(file_path):
    """
    Load the Meituan dataset and preprocess coordinates.
    
    Args:
        file_path (str): Path to the Meituan dataset CSV file.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with scaled coordinates.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")

    # Scale coordinates (as per preprocess_meituan_data.py)
    coordinate_columns = [
        'sender_lat', 'sender_lng', 'recipient_lat', 'recipient_lng',
        'grab_lat', 'grab_lng', 'rider_lat', 'rider_lng'
    ]
    scale_factor = 1000000.0  # Meituan coordinates are scaled by 1,000,000
    for col in coordinate_columns:
        if col in df.columns:
            if df[col].abs().max() > 1000:  # Check if scaling is needed
                print(f"Scaling down {col} from Meituan format")
                df[col] = df[col] / scale_factor

    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points in kilometers.
    
    Args:
        lat1, lon1 (float): Latitude and longitude of the first point (in degrees).
        lat2, lon2 (float): Latitude and longitude of the second point (in degrees).
    
    Returns:
        float: Distance in kilometers.
    """
    # Earth's radius in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance


def plot_restaurant_map(df, output_dir):
    """
    Create a scatter plot of restaurant locations, color-coded by business district (da_id).
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with restaurant coordinates.
        output_dir (str): Directory to save the plot.
    
    Returns:
        None
    """
    print("Generating restaurant map...")
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique restaurant locations
    restaurants = df[['poi_id', 'sender_lat', 'sender_lng', 'da_id']].drop_duplicates(subset=['poi_id'])
    print(f"Number of unique restaurants: {len(restaurants)}")

    # Convert da_id to integer to ensure consistent typing
    restaurants['da_id'] = restaurants['da_id'].astype(int)
    print(f"Unique da_id values: {sorted(restaurants['da_id'].unique())}")
    print(f"Number of unique business districts: {restaurants['da_id'].nunique()}")

    # Create a custom palette with enough colors for 23 districts
    # Start with tab20 colors (20 colors) and add 3 more distinct colors
    tab20_colors = sns.color_palette("tab20", 20)
    extra_colors = sns.color_palette("Set2", 3)  # Add 3 more colors from Set2
    custom_palette = tab20_colors + extra_colors  # Combine to get 23 colors

    # After sns.scatterplot, add manual legend handling
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=restaurants,
        x='sender_lng',
        y='sender_lat',
        hue='da_id',
        palette=custom_palette,
        size=10,
        alpha=0.7,
        legend=False  # Disable automatic legend
    )

    # Manually create the legend
    unique_da_ids = sorted(restaurants['da_id'].unique())
    handles = [plt.scatter([], [], color=custom_palette[i], label=da_id, s=50, alpha=0.7) 
            for i, da_id in enumerate(unique_da_ids)]
    plt.legend(handles=handles, title="Business District (da_id)", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Rest of the plot customization remains the same
    plt.title("Spatial Distribution of Restaurants by Business District")
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "restaurant_map.pdf")
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved restaurant map to {output_path}")
    plt.close()


def plot_distance_histogram(df, output_dir):
    """
    Create a histogram of distances for grab-to-restaurant and restaurant-to-customer segments.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with coordinates.
        output_dir (str): Directory to save the plot.
    
    Returns:
        None
    """
    print("Generating distance histogram...")
    os.makedirs(output_dir, exist_ok=True)

    # Filter out invalid records (e.g., zero coordinates for rejected waybills)
    valid_df = df[
        (df['grab_lat'] != 0) & (df['grab_lng'] != 0) &
        (df['sender_lat'].notna()) & (df['sender_lng'].notna()) &
        (df['recipient_lat'].notna()) & (df['recipient_lng'].notna())
    ]
    print(f"Number of valid records for distance calculation: {len(valid_df)}")

    # Calculate distances
    # Grab-to-Restaurant
    grab_to_restaurant = valid_df.apply(
        lambda row: haversine_distance(
            row['grab_lat'], row['grab_lng'],
            row['sender_lat'], row['sender_lng']
        ),
        axis=1
    )
    # Restaurant-to-Customer
    restaurant_to_customer = valid_df.apply(
        lambda row: haversine_distance(
            row['sender_lat'], row['sender_lng'],
            row['recipient_lat'], row['recipient_lng']
        ),
        axis=1
    )

    # Filter out zero distances and unreasonable distances (> 50 km)
    reasonable_mask_gr = (grab_to_restaurant > 0) & (grab_to_restaurant <= 50)
    reasonable_mask_rc = (restaurant_to_customer > 0) & (restaurant_to_customer <= 50)
    grab_to_restaurant = grab_to_restaurant[reasonable_mask_gr]
    restaurant_to_customer = restaurant_to_customer[reasonable_mask_rc]
    print(f"Grab-to-Restaurant distances after filtering: {len(grab_to_restaurant)}")
    print(f"Restaurant-to-Customer distances after filtering: {len(restaurant_to_customer)}")

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.hist(
        grab_to_restaurant,
        bins=30,  # Reduced number of bins for clarity
        alpha=0.5,
        label='Grab-to-Restaurant',
        color='blue',
        edgecolor='black'
    )
    plt.hist(
        restaurant_to_customer,
        bins=30,
        alpha=0.5,
        label='Restaurant-to-Customer',
        color='green',
        edgecolor='black'
    )

    # Add vertical lines for the mean distances
    plt.axvline(
        grab_to_restaurant.mean(),
        color='blue',
        linestyle='--',
        label=f'Grab-to-Restaurant Mean: {grab_to_restaurant.mean():.2f} km'
    )
    plt.axvline(
        restaurant_to_customer.mean(),
        color='green',
        linestyle='--',
        label=f'Restaurant-to-Customer Mean: {restaurant_to_customer.mean():.2f} km'
    )

    # Customize the plot
    plt.xlabel("Distance (km)")
    plt.ylabel("Number of Deliveries")
    plt.title("Distribution of Distances for Delivery Segments")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot as a PDF in the 'abb' folder
    output_path = os.path.join(output_dir, "distance_histogram.pdf")
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved distance histogram to {output_path}")
    plt.close()

    # Print summary statistics for inclusion in the LaTeX document
    print("\nDistance Summary Statistics:")
    print("Grab-to-Restaurant:")
    print(f"  Mean: {grab_to_restaurant.mean():.2f} km")
    print(f"  Min: {grab_to_restaurant.min():.2f} km")
    print(f"  Max: {grab_to_restaurant.max():.2f} km")
    print(f"  Coefficient of Variation: {grab_to_restaurant.std() / grab_to_restaurant.mean():.2f}")
    print("Restaurant-to-Customer:")
    print(f"  Mean: {restaurant_to_customer.mean():.2f} km")
    print(f"  Min: {restaurant_to_customer.min():.2f} km")
    print(f"  Max: {restaurant_to_customer.max():.2f} km")
    print(f"  Coefficient of Variation: {restaurant_to_customer.std() / restaurant_to_customer.mean():.2f}")


# def plot_route_heatmap(df, output_dir):
#     """
#     Create a heatmap of delivery route density for a single business district using all orders.
    
#     Args:
#         df (pd.DataFrame): Preprocessed DataFrame with coordinates.
#         output_dir (str): Directory to save the plot.
    
#     Returns:
#         None
#     """
#     print("Generating route heatmap...")
#     os.makedirs(output_dir, exist_ok=True)

#     # Filter out invalid records (missing coordinates)
#     valid_df = df[
#         (df['sender_lat'].notna()) & (df['sender_lng'].notna()) &
#         (df['recipient_lat'].notna()) & (df['recipient_lng'].notna())
#     ]
#     print(f"Number of valid records for route heatmap: {len(valid_df)}")

#     # Analyze the number of orders per business district to select one
#     orders_per_district = valid_df.groupby('da_id').size().sort_values()
#     print("\nOrders per business district:")
#     print(orders_per_district)

#     # Select a district with a moderate number of orders (e.g., around the median)
#     median_orders = orders_per_district.median()
#     selected_district = orders_per_district[
#         (orders_per_district >= median_orders * 0.8) & (orders_per_district <= median_orders * 1.2)
#     ].index[0]
#     print(f"Selected business district (da_id): {selected_district} with {orders_per_district[selected_district]} orders")

#     # Filter data for the selected district
#     district_df = valid_df[valid_df['da_id'] == selected_district]
#     print(f"Using all {len(district_df)} orders for density estimation")

#     # Set up the plot
#     plt.figure(figsize=(10, 8))

#     # Create a list of points along each route for density estimation
#     x_coords = []
#     y_coords = []
#     for _, row in district_df.iterrows():
#         # Generate points along the straight line between restaurant and customer
#         num_points = 10  # Keep 10 points per route for efficiency
#         lats = np.linspace(row['sender_lat'], row['recipient_lat'], num_points)
#         lngs = np.linspace(row['sender_lng'], row['recipient_lng'], num_points)
#         x_coords.extend(lngs)
#         y_coords.extend(lats)

#     # Plot the density heatmap of route paths
#     sns.kdeplot(
#         x=x_coords,
#         y=y_coords,
#         cmap='Reds',
#         fill=True,
#         alpha=0.6,
#         levels=20,
#         gridsize=50  # Keep grid size at 50 for efficiency
#     )

#     # Overlay restaurant and customer locations as scatter points for context
#     plt.scatter(
#         district_df['sender_lng'],
#         district_df['sender_lat'],
#         color='blue',
#         label='Restaurants',
#         alpha=0.3,
#         s=10
#     )
#     plt.scatter(
#         district_df['recipient_lng'],
#         district_df['recipient_lat'],
#         color='red',
#         label='Customers',
#         alpha=0.3,
#         s=10
#     )

#     # Customize the plot
#     plt.xlabel("Longitude (degrees)")
#     plt.ylabel("Latitude (degrees)")
#     plt.title(f"Heat Map for Business District {selected_district}")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     # Save the plot as a PDF in the 'abb' folder
#     output_path = os.path.join(output_dir, "route_heatmap.pdf")
#     plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
#     print(f"Saved route heatmap to {output_path}")
#     plt.close()
#         # In plot_route_heatmap, after filtering district_df
#     print(f"Longitude range: {district_df['sender_lng'].min():.6f} to {district_df['sender_lng'].max():.6f}")
#     print(f"Latitude range: {district_df['sender_lat'].min():.6f} to {district_df['sender_lat'].max():.6f}")






def plot_route_heatmap(df, output_dir):
    """
    Create a heatmap of delivery route density for a single business district using all orders.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with coordinates.
        output_dir (str): Directory to save the plot.
    
    Returns:
        None
    """
    print("Generating route heatmap...")
    os.makedirs(output_dir, exist_ok=True)

    # Filter out invalid records (missing coordinates)
    valid_df = df[
        (df['sender_lat'].notna()) & (df['sender_lng'].notna()) &
        (df['recipient_lat'].notna()) & (df['recipient_lng'].notna())
    ]
    print(f"Number of valid records for route heatmap: {len(valid_df)}")

    # Analyze the number of orders per business district to select one
    orders_per_district = valid_df.groupby('da_id').size().sort_values()
    print("\nOrders per business district:")
    print(orders_per_district)

    # Select a district with a moderate number of orders (e.g., around the median)
    median_orders = orders_per_district.median()
    selected_district = orders_per_district[
        (orders_per_district >= median_orders * 0.8) & (orders_per_district <= median_orders * 1.2)
    ].index[0]
    print(f"Selected business district (da_id): {selected_district} with {orders_per_district[selected_district]} orders")

    # Filter data for the selected district
    district_df = valid_df[valid_df['da_id'] == selected_district]
    print(f"Using all {len(district_df)} orders for density estimation")

    # Debug: Print the range of longitude and latitude values
    print(f"Longitude range: {district_df['sender_lng'].min():.6f} to {district_df['sender_lng'].max():.6f}")
    print(f"Latitude range: {district_df['sender_lat'].min():.6f} to {district_df['sender_lat'].max():.6f}")

    # Set up the plot
    plt.figure(figsize=(10, 8))

    # Create a list of points along each route for density estimation
    x_coords = []
    y_coords = []
    for _, row in district_df.iterrows():
        num_points = 10
        lats = np.linspace(row['sender_lat'], row['recipient_lat'], num_points)
        lngs = np.linspace(row['sender_lng'], row['recipient_lng'], num_points)
        x_coords.extend(lngs)
        y_coords.extend(lats)

    # Plot the density heatmap of route paths
    sns.kdeplot(
        x=x_coords,
        y=y_coords,
        cmap='Reds',
        fill=True,
        alpha=0.6,
        levels=20,
        gridsize=50
    )

    # Overlay restaurant and customer locations as scatter points
    plt.scatter(
        district_df['sender_lng'],
        district_df['sender_lat'],
        color='blue',
        label='Restaurants',
        alpha=0.3,
        s=10
    )
    plt.scatter(
        district_df['recipient_lng'],
        district_df['recipient_lat'],
        color='red',
        label='Customers',
        alpha=0.3,
        s=10
    )

    # Customize the plot
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plt.title(f"Heat Map for Business District {selected_district}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Format the x-axis to show full longitude values without offset
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))

    # Save the plot as a PDF in the 'abb' folder
    output_path = os.path.join(output_dir, "route_heatmap.pdf")
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved route heatmap to {output_path}")
    plt.close()





def main():
    # Set input and output paths
    data_dir = "data/meituan_data"
    input_file = os.path.join(data_dir, "all_waybill_info_meituan_0322.csv")  # Adjust to your file name
    output_dir = os.path.join(data_dir, "abb")  # Save in the 'abb' folder

    # Load and preprocess the data
    df = load_and_preprocess_data(input_file)

    # Generate the plots
    plot_restaurant_map(df, output_dir)
    plot_distance_histogram(df, output_dir)
    plot_route_heatmap(df, output_dir)

    print("Plot generation complete!")

if __name__ == "__main__":
    main()