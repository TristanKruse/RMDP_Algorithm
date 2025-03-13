import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_scatter_plot(
    csv_path="C:/Users/trika/Desktop/Masterarbeit/Thesis_Modell/thesis/data/meituan_data/all_waybill_info_meituan_0322.csv",
    initial_rows=30,
    save_path="C:/Users/trika/Desktop/Masterarbeit/Thesis_Modell/thesis/data_analysis/delivery_routes.png",
):
    # Read initial data
    df = pd.read_csv(csv_path, nrows=initial_rows)

    # Divide coordinates by 1000
    coordinate_columns = ["sender_lng", "sender_lat", "recipient_lng", "recipient_lat", "grab_lng", "grab_lat"]
    for col in coordinate_columns:
        df[col] = df[col] / 1000000

    # Filter zeros and NaN values
    df = df[
        (df["sender_lng"] != 0)
        & df["sender_lng"].notna()
        & (df["sender_lat"] != 0)
        & df["sender_lat"].notna()
        & (df["recipient_lng"] != 0)
        & df["recipient_lng"].notna()
        & (df["recipient_lat"] != 0)
        & df["recipient_lat"].notna()
        & (df["grab_lng"] != 0)
        & df["grab_lng"].notna()
        & (df["grab_lat"] != 0)
        & df["grab_lat"].notna()
    ]

    # Reset index
    df = df.reset_index(drop=True)

    # Select specific deliveries (skipping the problematic one)
    # Restaurant for index 2 is below another point
    selected_indices = [0, 1, 2, 3, 4]

    df = df.iloc[selected_indices].reset_index(drop=True)

    # Create plot

    plt.figure(figsize=(12, 8))

    # Generate colors for the deliveries
    colors = plt.cm.rainbow(np.linspace(0, 1, len(df)))

    # Plot each delivery
    for idx, row in df.iterrows():
        color = colors[idx]

        # Plot points without labels (no legend entries)
        plt.scatter(row["sender_lng"], row["sender_lat"], marker="o", color=color, s=100)
        plt.scatter(row["recipient_lng"], row["recipient_lat"], marker="s", color=color, s=100)
        plt.scatter(row["grab_lng"], row["grab_lat"], marker="^", color=color, s=100)

        # Connect points with lines
        x = [row["grab_lng"], row["sender_lng"], row["recipient_lng"]]
        y = [row["grab_lat"], row["sender_lat"], row["recipient_lat"]]
        plt.plot(x, y, color=color, linestyle="--", alpha=0.5)

    # Add general legend entries with black markers
    plt.scatter([], [], marker="o", color="black", s=100, label="Restaurant")
    plt.scatter([], [], marker="s", color="black", s=100, label="Customer")
    plt.scatter([], [], marker="^", color="black", s=100, label="Courier")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Meituan Food Delivery Routes\nShowing Courier → Restaurant → Customer paths")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure with high DPI for better quality
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Show the plot
    plt.show()


# Run the function
create_scatter_plot()
