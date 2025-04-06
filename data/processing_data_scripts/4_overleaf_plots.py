import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_courier_performance(waybill_df, output_dir):
    print("Analyzing courier performance...")
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Check required columns
    required_cols = ['arrive_time', 'estimate_arrived_time']
    missing_cols = [col for col in required_cols if col not in waybill_df.columns]
    if missing_cols:
        print(f"Error: Missing columns in waybill_df: {missing_cols}")
        return

    # Convert timestamps to datetime
    for col in ['arrive_time', 'estimate_arrived_time']:
        waybill_df[col] = pd.to_numeric(waybill_df[col], errors='coerce')
        waybill_df[col] = pd.to_datetime(waybill_df[col], unit='s', errors='coerce')

    # Filter out invalid timestamps and create a copy to avoid SettingWithCopyWarning
    valid_df = waybill_df[
        (waybill_df['arrive_time'].notna()) &
        (waybill_df['arrive_time'] != pd.Timestamp('1970-01-01')) &
        (waybill_df['estimate_arrived_time'].notna()) &
        (waybill_df['estimate_arrived_time'] != pd.Timestamp('1970-01-01'))
    ].copy()
    print(f"Valid orders for performance analysis: {len(valid_df)}")

    # Calculate delay in minutes using .loc
    valid_df.loc[:, 'delay_minutes'] = (valid_df['arrive_time'] - valid_df['estimate_arrived_time']).dt.total_seconds() / 60.0

    # On-time delivery rate
    on_time_orders = valid_df[valid_df['delay_minutes'] <= 0]
    on_time_rate = len(on_time_orders) / len(valid_df) * 100
    print(f"On-time delivery rate: {on_time_rate:.2f}%")

    # Analyze delays for late orders
    late_orders = valid_df[valid_df['delay_minutes'] > 0]
    if late_orders.empty:
        print("No late orders to analyze.")
        return

    mean_delay = late_orders['delay_minutes'].mean()
    std_delay = late_orders['delay_minutes'].std()
    print(f"Average delay for late orders: {mean_delay:.2f} minutes")
    print(f"Standard deviation of delays: {std_delay:.2f} minutes")

    # Filter late orders to 0-60 minutes for plotting
    late_orders_filtered = late_orders[(late_orders['delay_minutes'] >= 0) & (late_orders['delay_minutes'] <= 60)]

    # Plot: Histogram of delays for late orders with x-axis from 0 to 60 minutes
    plt.figure(figsize=(8, 5))
    sns.histplot(
        late_orders_filtered['delay_minutes'], 
        binwidth=1,  # Set bin width to 1 minute
        color='salmon', 
        stat='count'
    )
    plt.xlim(0, 60)  # Set x-axis limits from 0 to 60 minutes
    plt.xlabel('Delay (minutes)')
    plt.ylabel('Number of Late Orders')
    plt.title('Distribution of Delays for Late Orders (0 to 60 Minutes)')
    plt.grid(True, alpha=0.3)
    plt.axvline(
        mean_delay, 
        color='red', 
        linestyle='--', 
        label=f'Mean Delay: {mean_delay:.2f} minutes'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "delay_distribution.pdf"), dpi=300, format='pdf')
    plt.close()

if __name__ == "__main__":
    wave_path = "data/meituan_data/courier_wave_info_meituan.csv"
    waybill_path = "data/meituan_data/all_waybill_info_meituan_0322.csv"
    output_dir = "data/meituan_data/abb"

    wave_df = pd.read_csv(wave_path)
    waybill_df = pd.read_csv(waybill_path)
    print(f"Loaded wave data: {wave_path} ({len(wave_df)} rows)")
    print(f"Loaded waybill data: {waybill_path} ({len(waybill_df)} rows)")
    
    plot_courier_performance(waybill_df, output_dir)