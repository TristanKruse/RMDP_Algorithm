import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
csv_file = r"C:\Users\trika\Desktop\Masterarbeit\Thesis_Modell\thesis\data\simulation_results\vehicle_buffer_tuning.csv"
output_dir = r"C:\Users\trika\Desktop\Masterarbeit\Thesis_Modell\thesis\data\simulation_results"
output_file = r"C:\Users\trika\Desktop\Masterarbeit\Thesis_Modell\thesis\data\simulation_results\buffer_total_delay_bar_chart.pdf"

# Create output directory if it doesn't exist
import os
os.makedirs(output_dir, exist_ok=True)

# Load the CSV data
print(f"Loading data from {csv_file}...")
try:
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Ensure required columns are present
required_columns = ['buffer_size', 'total_delay', 'bundling_rate']
if not all(col in df.columns for col in required_columns):
    print(f"Error: CSV file must contain columns: {required_columns}")
    exit()

# Sort by buffer_size to ensure correct order in the plot
df = df.sort_values('buffer_size')

# Define colors: highlight optimal buffer size in orange, others in sky blue
optimal_buffer = df.loc[df['total_delay'].idxmin(), 'buffer_size']  # Buffer with minimum total delay
colors = ['orange' if buffer == optimal_buffer else 'skyblue' for buffer in df['buffer_size']]

# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(df['buffer_size'], df['total_delay'], color=colors, edgecolor='black')

# Start y-axis at zero for better visual comparison
y_max = max(df['total_delay']) * 1.1  # 10% padding above max value
ax.set_ylim(0, y_max)

# Customize the plot
ax.set_xlabel('Buffer Size')
ax.set_ylabel('Average Total Delay (minutes)')
ax.set_xlim(-1, 78)  # Limit x-axis to meaningful buffer range (0-78)
ax.set_xticks(range(0, 79, 10))  # Show every 10th buffer size for cleaner display
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as a PDF
plt.savefig(output_file, dpi=300, format='pdf')
plt.close()
print(f"Saved bar chart to {output_file}")

# Create bundling rate development plot
print("Creating bundling rate development plot...")
bundling_output_file = os.path.join(output_dir, "buffer_bundling_rate_development.pdf")

# Create a wider, shorter figure for the bundling rate plot
fig, ax = plt.subplots(figsize=(14, 5))

# Create line plot showing bundling rate development
ax.plot(df['buffer_size'], df['bundling_rate'], color='steelblue', linewidth=2, marker='o', markersize=4)

# Customize the plot
ax.set_xlabel('Buffer Size')
ax.set_ylabel('Bundling Rate (%)')
ax.set_xlim(0, 78)  # Limit x-axis to meaningful buffer range (0-78)
ax.set_xticks(range(0, 79, 10))  # Show every 10th buffer size for cleaner display
ax.grid(True, linestyle='--', alpha=0.7)

# Start y-axis at 0 for better comparison
y_max = max(df['bundling_rate']) * 1.05
ax.set_ylim(0, y_max)

plt.tight_layout()

# Save the bundling rate plot as a PDF
plt.savefig(bundling_output_file, dpi=300, format='pdf')
plt.close()
print(f"Saved bundling rate development plot to {bundling_output_file}")