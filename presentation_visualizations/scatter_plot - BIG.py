import pandas as pd
import matplotlib.pyplot as plt


def create_scatter_plot(
    csv_path="C:/Users/trika/Desktop/Masterarbeit/Thesis_Modell/thesis/data/meituan_data/all_waybill_info_meituan_0322.csv",
    limit=1000,
):
    # Read the CSV file with a limit
    df = pd.read_csv(csv_path, nrows=limit)

    # Divide coordinates by 1000
    coordinate_columns = ["sender_lng", "sender_lat", "recipient_lng", "recipient_lat", "grab_lng", "grab_lat"]
    for col in coordinate_columns:
        df[col] = df[col] / 1000000

    # Filter out rows where any of the coordinates are 0 or over 174.8
    df = df[
        # Filter zeros
        (df["sender_lng"] != 0)
        & (df["sender_lat"] != 0)
        & (df["recipient_lng"] != 0)
        & (df["recipient_lat"] != 0)
        & (df["grab_lng"] != 0)
        & (df["grab_lat"] != 0)
        &
        # Filter values over 174.8
        (df["sender_lng"] <= 174.8)
        & (df["sender_lat"] <= 174.8)
        & (df["recipient_lng"] <= 174.8)
        & (df["recipient_lat"] <= 174.8)
        & (df["grab_lng"] <= 174.8)
        & (df["grab_lat"] <= 174.8)
    ]

    # Print the number of remaining points after filtering
    print(f"Number of points after filtering zeros and values over 174.8: {len(df)}")

    # Create a figure with a specific size
    plt.figure(figsize=(12, 8))

    # Create scatter plots for each location type with different markers
    plt.scatter(df["sender_lng"], df["sender_lat"], marker="o", label="Sender Location", alpha=0.6, color="blue")
    plt.scatter(
        df["recipient_lng"], df["recipient_lat"], marker="s", label="Recipient Location", alpha=0.6, color="red"
    )
    plt.scatter(df["grab_lng"], df["grab_lat"], marker="^", label="Grab Location", alpha=0.6, color="green")

    # Customize the plot
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographic Distribution of Delivery Locations (Coordinates scaled by 1/1000)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.show()


# Run the function
create_scatter_plot()
