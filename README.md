# Restaurant Meal Delivery Problem (RMDP)

A Python implementation of a dynamic vehicle routing system for restaurant meal deliveries, based on the research paper by Ulmer et al. (2021).
open-webui serve

## Overview

This project implements a restaurant meal delivery system that:
- Manages dynamic order assignments to delivery vehicles
- Handles real-time food preparation updates
- Optimizes delivery routes considering multiple stakeholders
- Provides visualization of delivery operations

## Features

- **Dynamic Order Processing**: Real-time order assignment and route optimization
- **Multiple Routing Strategies**: 
  - Anticipatory Customer Assignment (ACA)
  - Fastest Vehicle Assignment
  - Order Bundling Optimization
- **Performance Metrics**: Tracks delays, delivery times, and vehicle utilization
- **Real-time Visualization**: Shows vehicle movements and delivery status

## Project Structure

```
restaurant-delivery/
├── environment/
│   ├── route_processing/     # Route calculation and processing
│   ├── location_manager.py   # Location and distance management
│   ├── order_manager.py      # Order handling and tracking
│   ├── vehicle_manager.py    # Vehicle state management
│   └── visualization.py      # Real-time visualization
├── models/
│   ├── aca_policy/          # Anticipatory Customer Assignment
│   ├── fastest_bundling/    # Bundling optimization
│   └── fastest_vehicle/     # Simple nearest vehicle assignment
├── config.yaml              # Configuration settings
├── datatypes.py            # Core data structures
└── main.py                # Entry point
```

## License

Copyright © 2024. All rights reserved.

## Paper Reference

Ulmer, M. W., Thomas, B. W., Campbell, A. M., & Woyak, N. (2021). The restaurant meal delivery problem: Dynamic pickup and delivery with deadlines and random ready times. Transportation Science, 55(1), 75-100.
