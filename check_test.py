import pandas as pd
import glob

# Find the most recent filtered results file
files = glob.glob("data/simulation_results/fastest_aca_filtered_results_*.csv")
if files:
    latest_file = max(files)
    print(f"Using file: {latest_file}")

    df = pd.read_csv(latest_file)
    performance = df.groupby("method")["on_time_delivery_rate"].mean().sort_values(ascending=False)
    print("\nAll methods performance:")
    for method, rate in performance.items():
        print(f"{method:20s}: {rate:6.1f}%")

    print(f"\nTotal records: {len(df)}")
    print(f"Records per method:")
    print(df["method"].value_counts())
else:
    print("No filtered results files found!")
