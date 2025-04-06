# I think we don't really need a time window, instead we would need to make it so we count the number of 
# active orders during a timespan, for a specific courier, for a specific wave, so e.g., 
# we have courier '1', and he starts an order with the grab time and ends an order with the arrive time, 
# we just want to look for overlaps in that timespan for that courier, for that wave.
# Do you understand my approach? Does this make sense to you?
# Also courier_wave_info_meituan.csv has about 206 k of lines, so not sure we are doing this correctly, 
# all of the order ids that are not within the wave dataset we could immediately filter out.
# Furthermore, I think the additional filters you proposed are great and the ones we already have we should 
# probably keep.

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_vehicle_capacity(wave_df, waybill_df, output_dir):
    print("Analyzing bundling potential and vehicle capacity...")
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Saving plot to: {os.path.abspath(viz_dir)}")

    # Check columns
    required_cols = ['courier_id', 'order_id', 'grab_time', 'arrive_time']
    missing_cols = [col for col in required_cols if col not in waybill_df.columns]
    if missing_cols:
        print(f"Error: Missing columns in waybill_df: {missing_cols}")
        return

    # Parse order_ids from waves
    wave_df['order_list'] = wave_df['order_ids'].apply(
        lambda x: [i.strip('[] $') for i in str(x).split(',')] if pd.notna(x) else []
    )
    all_wave_order_ids = set().union(*wave_df['order_list'])

    # Convert order_id to string
    waybill_df['order_id'] = waybill_df['order_id'].astype(str)

    # Filter waybill_df to wave orders
    waybill_df = waybill_df[waybill_df['order_id'].isin(all_wave_order_ids)].copy()
    print(f"Filtered waybill rows: {len(waybill_df)}")

    # Convert timestamps to datetime, handling invalid values
    for col in ['grab_time', 'arrive_time']:
        waybill_df[col] = pd.to_numeric(waybill_df[col], errors='coerce')
        waybill_df[col] = pd.to_datetime(waybill_df[col], unit='s', errors='coerce')

    # Filter out invalid waybills
    waybill_df = waybill_df[
        (waybill_df['grab_time'].notna()) & 
        (waybill_df['grab_time'] != pd.Timestamp('1970-01-01')) &
        (waybill_df['arrive_time'].notna()) & 
        (waybill_df['arrive_time'] != pd.Timestamp('1970-01-01')) &
        (waybill_df['grab_time'] < waybill_df['arrive_time'])
    ]
    print(f"Filtered waybill rows after timestamp validation: {len(waybill_df)}")

    # Explode wave_df to map orders to waves, preserving dt
    exploded_waves = wave_df.explode('order_list').rename(columns={'order_list': 'order_id'})
    exploded_waves['order_id'] = exploded_waves['order_id'].astype(str)

    # Merge with waybill_df, preserving dt
    merged_df = exploded_waves.merge(
        waybill_df, 
        on=['courier_id', 'order_id'], 
        how='inner',
        suffixes=('', '_waybill')
    )
    print(f"Merged rows: {len(merged_df)}")
    if merged_df.empty:
        print("Error: Merge resulted in no data.")
        return

    # Debugging: Sample wave
    sample_wave = merged_df[['courier_id', 'dt', 'wave_id']].drop_duplicates().iloc[0]
    sample_orders = merged_df[
        (merged_df['courier_id'] == sample_wave['courier_id']) & 
        (merged_df['dt'] == sample_wave['dt']) & 
        (merged_df['wave_id'] == sample_wave['wave_id'])
    ]
    print(f"Sample wave (courier {sample_wave['courier_id']}, dt {sample_wave['dt']}, wave {sample_wave['wave_id']}):")
    print(sample_orders[['order_id', 'grab_time', 'arrive_time']])

    # Calculate maximum concurrent orders per wave
    max_concurrent_per_wave = []
    for (courier, dt, wave), group in merged_df.groupby(['courier_id', 'dt', 'wave_id']):
        if group.empty:
            continue
        # For each order in the wave, count overlapping orders
        max_concurrent = 0
        for _, row in group.iterrows():
            overlap = len(group[
                (group['grab_time'] <= row['arrive_time']) &
                (group['arrive_time'] >= row['grab_time'])
            ])
            max_concurrent = max(max_concurrent, overlap)
        max_concurrent_per_wave.append({
            'courier_id': courier,
            'dt': dt,
            'wave_id': wave,
            'max_concurrent_orders': max_concurrent
        })

    # Convert to DataFrame
    concurrent_df = pd.DataFrame(max_concurrent_per_wave)
    if concurrent_df.empty:
        print("Error: No valid data to plot")
        return

    # Calculate statistics for bundling potential
    bundling_potential = concurrent_df['max_concurrent_orders']
    mean_bp = bundling_potential.mean()
    median_bp = bundling_potential.median()
    percentile_90 = bundling_potential.quantile(0.9)
    max_bp = bundling_potential.max()

    # Plot 1: Distribution of Maximum Concurrent Orders per Wave (Bundling Potential)
    plt.figure(figsize=(10, 6))
    sns.histplot(
        bundling_potential, 
        bins=range(1, int(bundling_potential.max()) + 2), 
        color='skyblue', 
        stat='count'
    )
    plt.xlabel('Maximum Concurrent Orders per Wave (Bundling Potential)')
    plt.ylabel('Number of Waves')
    plt.title('Distribution of Bundling Potential')
    plt.grid(True, alpha=0.3)
    plt.axvline(
        mean_bp, 
        color='red', 
        linestyle='--', 
        label=f'Mean: {mean_bp:.1f}'
    )
    plt.axvline(
        median_bp, 
        color='green', 
        linestyle='-.', 
        label=f'Median: {median_bp:.1f}'
    )
    plt.axvline(
        percentile_90, 
        color='blue', 
        linestyle=':', 
        label=f'90th Percentile (Vehicle Capacity): {percentile_90:.1f}'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "bundling_potential.pdf"), dpi=300, format='pdf')
    plt.close()

    # Plot 2: Distribution of 90th to 100th Percentile
    upper_tail = bundling_potential[bundling_potential >= percentile_90]
    plt.figure(figsize=(10, 6))
    sns.histplot(
        upper_tail, 
        bins=range(int(percentile_90), int(bundling_potential.max()) + 2), 
        color='lightcoral', 
        stat='count'
    )
    plt.xlabel('Maximum Concurrent Orders per Wave (90th to 100th Percentile)')
    plt.ylabel('Number of Waves')
    plt.title('Upper Tail of Bundling Potential (90th to 100th Percentile)')
    plt.grid(True, alpha=0.3)
    plt.axvline(
        max_bp, 
        color='purple', 
        linestyle='--', 
        label=f'Maximum: {max_bp:.1f}'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "bundling_potential_upper_tail.pdf"), dpi=300, format='pdf')
    plt.close()

    # Print statistics
    print("Bundling Potential (Maximum Concurrent Orders per Wave):")
    print(f"Mean: {mean_bp:.1f}")
    print(f"Median: {median_bp:.1f}")
    print(f"90th percentile (Vehicle Capacity): {percentile_90:.1f}")
    print(f"Maximum Bundling Potential: {max_bp:.1f}")

if __name__ == "__main__":
    wave_path = "data/meituan_data/courier_wave_info_meituan.csv"
    waybill_path = "data/meituan_data/all_waybill_info_meituan_0322.csv"
    output_dir = "data/meituan_data/abb"

    wave_df = pd.read_csv(wave_path)
    waybill_df = pd.read_csv(waybill_path)
    print(f"Loaded wave data: {wave_path} ({len(wave_df)} rows)")
    print(f"Loaded waybill data: {waybill_path} ({len(waybill_df)} rows)")
    plot_vehicle_capacity(wave_df, waybill_df, output_dir)