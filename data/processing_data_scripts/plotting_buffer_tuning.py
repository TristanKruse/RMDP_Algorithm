import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
csv_file = r"C:\Users\trika\Desktop\Masterarbeit\Thesis_Modell\thesis\data\bundling_tuning\100 episoden\vehicle_buffer_tuning.csv"
output_dir = r"C:\Users\trika\Desktop\Masterarbeit\Thesis_Modell\thesis\data\bundling_tuning\100 episoden"
output_file = r"C:\Users\trika\Desktop\Masterarbeit\Thesis_Modell\thesis\data\bundling_tuning\100 episoden\buffer_total_delay_bar_chart.pdf"

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
required_columns = ['buffer_size', 'total_delay']
if not all(col in df.columns for col in required_columns):
    print(f"Error: CSV file must contain columns: {required_columns}")
    exit()

# Sort by buffer_size to ensure correct order in the plot
df = df.sort_values('buffer_size')

# Define colors: highlight buffer size 17 in orange, others in sky blue
colors = ['orange' if buffer == 17 else 'skyblue' for buffer in df['buffer_size']]

# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(df['buffer_size'], df['total_delay'], color=colors, edgecolor='black')

# Zoom in on the y-axis starting at 500
y_start = 500
y_max = max(df['total_delay']) * 1.1  # Add 10% padding above the max value
ax.set_ylim(y_start, y_max)

# Customize the plot
ax.set_xlabel('Buffer Size')
ax.set_ylabel('Average Total Delay (minutes)')
ax.set_title('Average Total Delay vs. ACA Vehicle Buffer Size (100 Episodes per Buffer)')
ax.set_xticks(df['buffer_size'])
ax.set_xticklabels(df['buffer_size'], rotation=45)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as a PDF
plt.savefig(output_file, dpi=300, format='pdf')
plt.close()
print(f"Saved bar chart to {output_file}")