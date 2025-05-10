# Restaurant Meal Delivery Problem (RMDP)

A Python implementation of a dynamic vehicle routing system for restaurant meal deliveries, inspired by the research paper by Ulmer et al. (2021). This project simulates and optimizes meal delivery operations with real-time order processing and route planning.

## Overview

The Restaurant Meal Delivery Problem (RMDP) addresses the challenge of efficiently assigning and delivering meal orders to customers using a fleet of vehicles. This implementation:

- Dynamically assigns incoming orders to delivery vehicles.
- Updates routes in real-time based on food preparation times and traffic conditions.
- Optimizes delivery operations to balance customer satisfaction, restaurant efficiency, and vehicle utilization.
- Visualizes vehicle movements and delivery statuses in real-time.

## Features

- **Dynamic Order Processing**: Assigns orders to vehicles in real-time while optimizing routes.
- **Routing Strategies**:
  - **Reinforcement Learning Anticipatory Customer Assignment (RL-ACA)**: Predicts future orders for proactive routing, postpones based on a reinforcement learning algorithm.
  - **Anticipatory Customer Assignment (ACA)**: Predicts future orders for proactive routing.
  - **Fastest Vehicle Assignment**: Assigns orders to the nearest available vehicle.
  - **Order Bundling Optimization**: Groups orders to minimize delivery times.
- **Performance Metrics**: Tracks key metrics like delivery delays, total delivery times, and vehicle utilization rates.
- **Real-time Visualization**: Displays vehicle routes, order statuses, and delivery progress (pause visualization with the **Spacebar**).

## Project Structure

```
restaurant-delivery/
├── environment/                # Core environment logic
│   ├── route_processing/       # Route calculation and optimization
│   ├── location_manager.py     # Manages locations and distances
│   ├── order_manager.py        # Handles order creation and tracking
│   ├── vehicle_manager.py      # Tracks vehicle states and assignments
│   └── visualization.py        # Renders real-time delivery visualization
├── models/                     # Routing strategy implementations
│   ├── aca_policy/            # Anticipatory Customer Assignment logic
│   ├── fastest_bundling/      # Order bundling optimization
│   └── fastest_vehicle/       # Nearest vehicle assignment strategy
├── config.yaml                # Configuration settings (e.g., simulation parameters)
├── datatypes.py               # Core data structures for orders, vehicles, etc.
└── train.py                    # Entry point for running the simulation
```

## State vs. Decision State

- **State**: Represents the full system state, including all information about orders, vehicles, and locations, used by the environment for simulation.
- **Decision State**: A subset of the state containing only the information necessary for making routing decisions, used by the routing algorithms.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/TristanKruse/RMDP_Algorithm.git
   cd RMDP_Algorithm
   ```

2. **Set up a Python environment** (Python 3.8+ recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *Note*: If `requirements.txt` is not provided, install required packages manually:

   ```bash
   pip install numpy pyyaml matplotlib
   ```

## Usage

1. **Configure the simulation**: Edit `config.yaml` to adjust parameters like the number of vehicles, order frequency, or simulation duration.

2. **Run the simulation**:

   ```bash
   python train.py
   ```

3. **Interact with the visualization**:

   - The visualization displays vehicle movements and delivery statuses.
   - Press the **Spacebar** to pause/resume the visualization.

4. **Example output**:

   - The simulation logs performance metrics (e.g., average delivery time, vehicle utilization).
   - Visualizations show real-time vehicle routes and order statuses.


## License

Copyright © 2024. All rights reserved.

## Reference

Ulmer, M. W., Thomas, B. W., Campbell, A. M., & Woyak, N. (2021). The restaurant meal delivery problem: Dynamic pickup and delivery with deadlines and random ready times. *Transportation Science*, 55(1), 75-100. DOI: 10.1287/trsc.2020.1000

## Contact

For questions or suggestions, please open an issue on the GitHub repository or contact the maintainer at krusetristan1@gmail.com
