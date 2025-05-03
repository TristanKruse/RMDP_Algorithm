import pandas as pd
import numpy as np
from math import radians, cos, sin, sqrt, atan2
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Haversine formula to calculate distance between two lat/lng points (in km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

# Load the full Meituan dataset
data_dir = "data/meituan_data"
order_file = f"{data_dir}/all_waybill_info_meituan_0322.csv"
print(f"Loading data from {order_file}...")
df = pd.read_csv(order_file)

# Filter orders with valid coordinates and da_id
valid_df = df[df['sender_lat'].notna() & df['sender_lng'].notna() &
              df['recipient_lat'].notna() & df['recipient_lng'].notna() &
              df['da_id'].notna()].copy()
print(f"Number of orders with valid coordinates and da_id: {len(valid_df)}")

# Debug: Print raw coordinate ranges
print("\nRaw Coordinate Ranges (before scaling):")
print(f"Min sender_lat: {valid_df['sender_lat'].min()}, Max sender_lat: {valid_df['sender_lat'].max()}")
print(f"Min sender_lng: {valid_df['sender_lng'].min()}, Max sender_lng: {valid_df['sender_lng'].max()}")

# Convert da_id to integer
valid_df['da_id'] = valid_df['da_id'].astype(int)

# Step 1: Compute bounding box for each district (simulation's method, without buffer)
district_bounds = []
scale_factor = 1000000.0  # Simulation's scaling factor
for da_id in valid_df['da_id'].unique():
    district_orders = valid_df[valid_df['da_id'] == da_id]
    
    # Combine sender and recipient coordinates to get the full extent
    all_lats = pd.concat([district_orders['sender_lat'], district_orders['recipient_lat']])
    all_lons = pd.concat([district_orders['sender_lng'], district_orders['recipient_lng']])
    
    min_lat, max_lat = all_lats.min(), all_lats.max()
    min_lng, max_lng = all_lons.min(), all_lons.max()
    
    # Scale coordinates (as in simulation)
    scaled_min_lat = min_lat / scale_factor
    scaled_max_lat = max_lat / scale_factor
    scaled_min_lng = min_lng / scale_factor
    scaled_max_lng = max_lng / scale_factor
    
    # Calculate dimensions in km (simulation's method, without buffer)
    mean_lat = (scaled_min_lat + scaled_max_lat) / 2
    lat_km = (scaled_max_lat - scaled_min_lat) * 111  # Height in km
    lng_km = (scaled_max_lng - scaled_min_lng) * 111 * cos(radians(mean_lat))  # Width in km
    area_km2 = lat_km * lng_km
    
    district_bounds.append({
        'District': f"DA {da_id}",
        'Width (km)': lng_km,
        'Height (km)': lat_km,
        'Area (km²)': area_km2
    })

# Create DataFrame for district bounds and remove duplicates
bounds_df = pd.DataFrame(district_bounds)

# Extract DA number for sorting
bounds_df['DA_Number'] = bounds_df['District'].str.extract(r'(\d+)').astype(int)
bounds_df = bounds_df.sort_values('DA_Number').drop(columns='DA_Number')

print("\nDistrict Bounding Boxes and Sizes (Sorted, No Buffer):")
print(bounds_df[['District', 'Width (km)', 'Height (km)', 'Area (km²)']])

# Step 2: Save to LaTeX table
# Format the table with District, Width, and Height
latex_df = bounds_df[['District', 'Width (km)', 'Height (km)']].copy()
latex_df['Width (km)'] = latex_df['Width (km)'].round(2)
latex_df['Height (km)'] = latex_df['Height (km)'].round(2)

# Generate LaTeX table with two columns (split into two sets for readability)
latex_table = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{lrr|lrr}\n\\toprule\n"
latex_table += "\\textbf{District} & \\textbf{Width (km)} & \\textbf{Height (km)} & " + \
               "\\textbf{District} & \\textbf{Width (km)} & \\textbf{Height (km)} \\\\\n\\midrule\n"

# Split into two sets (first half and second half)
midpoint = (len(latex_df) + 1) // 2
first_half = latex_df.iloc[:midpoint]
second_half = latex_df.iloc[midpoint:]

for i in range(midpoint):
    if i < len(first_half):
        row1 = first_half.iloc[i]
        row1_str = f"{row1['District']} & {row1['Width (km)']:.2f} & {row1['Height (km)']:.2f}"
    else:
        row1_str = "& &"
    
    if i < len(second_half):
        row2 = second_half.iloc[i]
        row2_str = f"{row2['District']} & {row2['Width (km)']:.2f} & {row2['Height (km)']:.2f}"
    else:
        row2_str = "& &"
    
    latex_table += f"{row1_str} & {row2_str} \\\\\n"

latex_table += "\\bottomrule\n\\end{tabular}\n"
latex_table += "\\caption{Widths and Heights of Service Areas for Each District in the Meituan Dataset (Without Buffer)}\n"
latex_table += "\\label{tab:service_area_dimensions_no_buffer}\n\\end{table}"

# Determine the project root (script is in thesis/data/processing_data_scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute path of the script
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels to thesis/

# Save the LaTeX table to the specified directory relative to the project root
output_dir = os.path.join(project_root, "data", "meituan_data", "abb")
os.makedirs(output_dir, exist_ok=True)
latex_path = os.path.join(output_dir, "service_area_dimensions_no_buffer.tex")
with open(latex_path, 'w') as f:
    f.write(latex_table)
print(f"\nLaTeX table saved to: {latex_path}")
print(f"Absolute path for verification: {os.path.abspath(latex_path)}")

# Step 3: Enhanced Downtown Concentration Analysis
# Extract unique restaurants based on poi_id
restaurants_df = valid_df[['poi_id', 'sender_lat', 'sender_lng', 'da_id']].drop_duplicates(subset=['poi_id'])
restaurants_df['da_id'] = restaurants_df['da_id'].astype(int)
print(f"\nNumber of unique restaurants: {len(restaurants_df)}")

# Scale coordinates
restaurants_df['sender_lat'] = restaurants_df['sender_lat'] / scale_factor
restaurants_df['sender_lng'] = restaurants_df['sender_lng'] / scale_factor

# Compute district centers (mean sender_lat and sender_lng for each da_id)
district_centers = restaurants_df.groupby('da_id')[['sender_lat', 'sender_lng']].mean().reset_index()
district_centers.rename(columns={'sender_lat': 'center_lat', 'sender_lng': 'center_lng'}, inplace=True)

# Merge district centers back into restaurants_df
restaurants_df = restaurants_df.merge(district_centers, on='da_id', how='left')

# Calculate distance of each restaurant from its district center
restaurants_df['distance_to_center_km'] = restaurants_df.apply(
    lambda row: haversine(row['sender_lat'], row['sender_lng'], 
                          row['center_lat'], row['center_lng']),
    axis=1
)

# Step 3.1: Analyze the distribution of distances to district centers
print("\nDistance to District Center Statistics:")
print(f"Mean distance: {restaurants_df['distance_to_center_km'].mean():.2f} km")
print(f"Median distance: {restaurants_df['distance_to_center_km'].median():.2f} km")
print(f"25th percentile: {restaurants_df['distance_to_center_km'].quantile(0.25):.2f} km")
print(f"50th percentile: {restaurants_df['distance_to_center_km'].quantile(0.50):.2f} km")
print(f"75th percentile: {restaurants_df['distance_to_center_km'].quantile(0.75):.2f} km")
print(f"Max distance: {restaurants_df['distance_to_center_km'].max():.2f} km")

# Step 3.2: Plot histogram of distances to district centers
plt.figure(figsize=(10, 6))
sns.histplot(restaurants_df['distance_to_center_km'], bins=50, kde=True, color='blue', stat='density')
plt.axvline(restaurants_df['distance_to_center_km'].median(), color='red', linestyle='--', 
            label=f'Median: {restaurants_df["distance_to_center_km"].median():.2f} km')
plt.xlabel("Distance to District Center (km)")
plt.ylabel("Density")
plt.title("Distribution of Restaurant Distances to District Centers")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the histogram as a PDF
histogram_path = os.path.join(output_dir, "restaurant_distance_histogram.pdf")
plt.savefig(histogram_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"\nSaved restaurant distance histogram to: {histogram_path}")
plt.close()

# Step 3.3: Determine downtown radius (distance within which ~50% of restaurants lie)
downtown_radius_km = restaurants_df['distance_to_center_km'].quantile(0.50)  # Median distance
print(f"\nSelected downtown radius (median distance): {downtown_radius_km:.2f} km")

# Step 3.4: Calculate the proportion of restaurants within the downtown radius for each district
concentration_by_district = restaurants_df.groupby('da_id').apply(
    lambda x: (x['distance_to_center_km'] <= downtown_radius_km).mean()
).reset_index(name='downtown_concentration')

# Step 3.5: Compute summary statistics for downtown concentration
average_downtown_concentration = concentration_by_district['downtown_concentration'].mean()
median_downtown_concentration = concentration_by_district['downtown_concentration'].median()
q1_downtown_concentration = concentration_by_district['downtown_concentration'].quantile(0.25)
q3_downtown_concentration = concentration_by_district['downtown_concentration'].quantile(0.75)

print("\nDowntown Concentration Statistics:")
print(f"Average downtown concentration: {average_downtown_concentration:.2f}")
print(f"Median downtown concentration: {median_downtown_concentration:.2f}")
print(f"25th percentile: {q1_downtown_concentration:.2f}")
print(f"75th percentile: {q3_downtown_concentration:.2f}")

print("\nDowntown Concentration by District:")
print(concentration_by_district)

# Step 3.6: Save downtown concentration by district to a LaTeX table in the annex
concentration_latex_df = concentration_by_district.copy()
concentration_latex_df['da_id'] = concentration_latex_df['da_id'].apply(lambda x: f"DA {x}")
concentration_latex_df['downtown_concentration'] = (concentration_latex_df['downtown_concentration'] * 100).round(2)  # Convert to percentage
concentration_latex_df.rename(columns={'da_id': 'District', 'downtown_concentration': 'Concentration (\\%)'}, inplace=True)

# Generate LaTeX table
concentration_latex_table = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{lr|lr}\n\\toprule\n"
concentration_latex_table += "\\textbf{District} & \\textbf{Concentration (\\%)} & " + \
                             "\\textbf{District} & \\textbf{Concentration (\\%)} \\\\\n\\midrule\n"

# Split into two sets
midpoint = (len(concentration_latex_df) + 1) // 2
first_half = concentration_latex_df.iloc[:midpoint]
second_half = concentration_latex_df.iloc[midpoint:]

for i in range(midpoint):
    if i < len(first_half):
        row1 = first_half.iloc[i]
        row1_str = f"{row1['District']} & {row1['Concentration (\\%)']:.2f}"
    else:
        row1_str = "&"
    
    if i < len(second_half):
        row2 = second_half.iloc[i]
        row2_str = f"{row2['District']} & {row2['Concentration (\\%)']:.2f}"
    else:
        row2_str = "&"
    
    concentration_latex_table += f"{row1_str} & {row2_str} \\\\\n"

concentration_latex_table += "\\bottomrule\n\\end{tabular}\n"
concentration_latex_table += f"\\caption{{Downtown Concentration by District (Restaurants within {downtown_radius_km:.2f} km of District Center)}}\n"
concentration_latex_table += "\\label{tab:downtown_concentration_by_district}\n\\end{table}"

# Save the LaTeX table
concentration_latex_path = os.path.join(output_dir, "downtown_concentration_by_district.tex")
with open(concentration_latex_path, 'w') as f:
    f.write(concentration_latex_table)
print(f"Saved downtown concentration table to: {concentration_latex_path}")

# Step 4: Compute average and max delivery distances (Haversine-like, as in simulation)
valid_df['sender_lat'] = valid_df['sender_lat'] / scale_factor
valid_df['sender_lng'] = valid_df['sender_lng'] / scale_factor
valid_df['recipient_lat'] = valid_df['recipient_lat'] / scale_factor
valid_df['recipient_lng'] = valid_df['recipient_lng'] / scale_factor
valid_df['delivery_distance_km'] = valid_df.apply(
    lambda row: haversine(row['sender_lat'], row['sender_lng'], 
                          row['recipient_lat'], row['recipient_lng']),
    axis=1
)

# Sample 10,000 orders if the dataset is large to speed up computation
if len(valid_df) > 10000:
    sample_df = valid_df.sample(10000, random_state=42)
else:
    sample_df = valid_df

avg_delivery_distance = sample_df['delivery_distance_km'].mean()
max_delivery_distance = sample_df['delivery_distance_km'].max()

print(f"\nDelivery Distance Statistics (based on {len(sample_df)} orders):")
print(f"Average delivery distance: {avg_delivery_distance:.2f} km")
print(f"Maximum delivery distance: {max_delivery_distance:.2f} km")