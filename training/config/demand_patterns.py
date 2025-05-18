hourly_pattern = {
    "type": "hourly",
    "hourly_rates": [
        6,
        3,
        1,
        0.5,
        0.5,
        0.5,  # 0-5 AM
        2,
        5,
        10,
        15,
        18,
        90,  # 6-11 AM
        40,
        22,
        19,
        19,
        30,
        40,  # 12-5 PM
        62,
        62,
        40,
        25,
        18,
        10,  # 6-11 PM
    ],
}

# Lunch dinner pattern for 12 hours, starting at 10 a.m
lunch_dinner_pattern = {
    "type": "hourly",
    "hourly_rates": {
        0: 1.59,  # Maps to 10:00
        1: 4.38,  # Maps to 11:00
        2: 2.38,  # Maps to 12:00
        3: 1.15,  # Maps to 13:00
        4: 0.84,  # Maps to 14:00
        5: 0.76,  # Maps to 15:00
        6: 1.00,  # Maps to 16:00
        7: 2.25,  # Maps to 17:00
        8: 2.86,  # Maps to 18:00
        9: 1.94,  # Maps to 19:00
        10: 1.23,  # Maps to 20:00
        11: 0.85,  # Maps to 21:00
        12: 0.55,  # Maps to 22:00
        13: 0.37,  # Maps to 23:00
        14: 0.21,  # Maps to 00:00
        15: 0.13,  # Maps to 01:00
        16: 0.08,  # Maps to 02:00
        17: 0.05,  # Maps to 03:00
        18: 0.04,  # Maps to 04:00
        19: 0.04,  # Maps to 05:00
        20: 0.12,  # Maps to 06:00
        21: 0.24,  # Maps to 07:00
        22: 0.41,  # Maps to 08:00
        23: 0.53,  # Maps to 09:00
    },
}


def bimodal_demand(time_percent):
    """Generate a bimodal distribution with peaks at lunch and dinner."""
    import numpy as np

    # Create two normal distributions centered at lunch and dinner times
    lunch_peak = np.exp(-((time_percent - 0.4) ** 2) / 0.005)  # Peak around 40% of the day
    dinner_peak = np.exp(-((time_percent - 0.7) ** 2) / 0.005)  # Peak around 70% of the day
    # Combine the distributions and scale
    return 0.2 + 2.8 * (lunch_peak + dinner_peak)  # Base 0.2, peaks at 3.0


function_pattern = {"type": "function", "function": bimodal_demand}
