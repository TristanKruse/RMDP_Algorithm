import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_bundling_example(
    with_paths=True, save_path="C:/Users/trika/Desktop/Masterarbeit/Thesis_Modell/thesis/data_analysis/bundling_example"
):
    # Create example coordinates for bundled delivery
    example_points = {
        "vehicle": (0.03, 45.87),
        "restaurant1": (0.04, 45.89),
        "restaurant2": (0.05, 45.90),
        "customer1": (0.06, 45.88),
        "customer2": (0.07, 45.89),
    }

    plt.figure(figsize=(12, 8))

    # Plot points in black
    plt.scatter(example_points["vehicle"][0], example_points["vehicle"][1], marker="^", color="black", s=100)
    plt.scatter(example_points["restaurant1"][0], example_points["restaurant1"][1], marker="o", color="black", s=100)
    plt.scatter(example_points["restaurant2"][0], example_points["restaurant2"][1], marker="o", color="black", s=100)
    plt.scatter(example_points["customer1"][0], example_points["customer1"][1], marker="s", color="black", s=100)
    plt.scatter(example_points["customer2"][0], example_points["customer2"][1], marker="s", color="black", s=100)

    # Connect points in order (only if with_paths is True)
    if with_paths:
        x = [
            example_points["vehicle"][0],
            example_points["restaurant1"][0],
            example_points["restaurant2"][0],
            example_points["customer1"][0],
            example_points["customer2"][0],
        ]
        y = [
            example_points["vehicle"][1],
            example_points["restaurant1"][1],
            example_points["restaurant2"][1],
            example_points["customer1"][1],
            example_points["customer2"][1],
        ]
        plt.plot(x, y, color="black", linestyle="--", alpha=0.5)

    # Add general legend entries with black markers
    plt.scatter([], [], marker="o", color="black", s=100, label="Restaurant")
    plt.scatter([], [], marker="s", color="black", s=100, label="Customer")
    plt.scatter([], [], marker="^", color="black", s=100, label="Courier")

    # Set specific axis limits
    plt.xlim(0.02, 0.09)
    plt.ylim(45.86, 45.92)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Example of Bundled Food Delivery Route")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure with appropriate suffix
    suffix = "_with_paths" if with_paths else "_without_paths"
    plt.savefig(save_path + suffix + ".png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()


# Create both versions
create_bundling_example(with_paths=True)
create_bundling_example(with_paths=False)

