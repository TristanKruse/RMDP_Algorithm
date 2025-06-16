import pandas as pd
import numpy as np
from math import radians, cos, sin, sqrt, atan2
import os

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
    # First half row
    if i < len(first_half):
        row1 = first_half.iloc[i]
        row1_str = f"{row1['District']} & {row1['Width (km)']:.2f} & {row1['Height (km)']:.2f}"
    else:
        row1_str = "& &"
    
    # Second half row
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

# Step 3: Compute average and max delivery distances (Haversine-like, as in simulation)
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