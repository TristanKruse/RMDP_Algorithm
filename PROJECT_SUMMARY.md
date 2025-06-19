# Restaurant Meal Delivery Problem (RMDP) - Project Summary

## Overview

This project implements a comprehensive solution to the Restaurant Meal Delivery Problem (RMDP), a complex vehicle routing and scheduling optimization challenge inspired by real-world food delivery platforms like Meituan. The system combines advanced algorithmic approaches with reinforcement learning to optimize meal delivery operations under dynamic, real-time conditions.

## Project Motivation

The Restaurant Meal Delivery Problem addresses the critical challenge of efficiently managing food delivery operations for online platforms. Key challenges include:

- **Dynamic Order Processing**: Orders arrive continuously with uncertain timing
- **Multiple Stakeholder Objectives**: Balancing customer satisfaction, courier efficiency, restaurant reputation, and platform profitability
- **Real-time Decision Making**: Adapting to unpredictable food preparation times and traffic conditions
- **Resource Optimization**: Maximizing vehicle utilization while maintaining service quality

## Core Features

### 1. Dynamic Order Processing System
- **Real-time Order Assignment**: Assigns incoming orders to couriers as they arrive
- **Route Optimization**: Continuously optimizes delivery routes based on current conditions
- **Uncertainty Handling**: Manages unpredictable food preparation times and travel conditions

### 2. Advanced Routing Algorithms

#### Reinforcement Learning-Enhanced ACA (RL-ACA)
- **Intelligent Postponement**: Uses deep reinforcement learning to decide when to delay order assignments
- **Future Order Prediction**: Anticipates future orders for proactive routing decisions
- **Phased Training**: Progressive curriculum learning from simple to complex environments
- **Safety Mechanisms**: Fallback to heuristic methods when RL confidence is low

#### Traditional Algorithms
- **Anticipatory Customer Assignment (ACA)**: Heuristic-based postponement with configurable time buffers
- **Fastest Vehicle Assignment**: Assigns orders to nearest available vehicle
- **Order Bundling Optimization**: Groups orders to minimize total delivery times

### 3. Performance Monitoring
- **Comprehensive KPIs**: Tracks 6+ key performance indicators
- **Real-time Metrics**: On-time delivery rates, total delays, vehicle utilization
- **Comparative Analysis**: Benchmarks against real-world Meituan performance data

### 4. Visualization and Analysis
- **Real-time Visualization**: Interactive display of vehicle routes and order statuses
- **Performance Dashboards**: Comprehensive charts and metrics visualization
- **Statistical Analysis**: Significance testing and effect size calculations

## Technical Architecture

### Core Components

```
restaurant-delivery/
├── environment/                 # Simulation Environment
│   ├── route_processing/       # Route calculation and optimization
│   ├── meituan_data/          # Real-world data integration
│   ├── location_manager.py    # Manages locations and distances
│   ├── order_manager.py       # Handles order lifecycle
│   ├── vehicle_manager.py     # Tracks vehicle states
│   └── visualization.py       # Real-time visualization
├── models/                     # Algorithm Implementations
│   ├── aca_policy/            # ACA and RL-ACA algorithms
│   │   ├── cheapest_insertion/ # C++ optimization module
│   │   ├── rl_postponement.py # RL decision making
│   │   └── postponement.py    # Heuristic postponement
│   ├── fastest_bundling/      # Order bundling optimization
│   └── fastest_vehicle/       # Nearest vehicle assignment
├── training/                   # RL Training Infrastructure
│   ├── config/                # Training configurations
│   ├── core/                  # Training core logic
│   └── utils/                 # Training utilities
├── benchmarking/              # Performance Analysis
├── data/                      # Datasets and Results
└── visualization/             # Analysis and Plots
```

### Key Technologies

- **Python 3.8+**: Core implementation language
- **PyTorch**: Deep reinforcement learning framework
- **C++**: High-performance route optimization (cheapest insertion)
- **NumPy/Pandas**: Data processing and analysis
- **Matplotlib/Seaborn**: Visualization and plotting
- **YAML**: Configuration management

## Algorithm Details

### Reinforcement Learning Architecture

#### State Representation
- Vehicle positions and current routes
- Active orders and their deadlines
- Restaurant preparation times and locations
- Historical performance metrics

#### Action Space
- **Binary Decision**: Postpone order assignment (Yes/No)
- **Safety Constraints**: Fallback to heuristic when confidence is low

#### Reward Structure
- **On-time Delivery Bonus**: Rewards successful deliveries
- **Bundling Efficiency**: Incentivizes route optimization
- **Postponement Penalties**: Balances delay risks

#### Training Strategy
1. **Phase 1**: Simple environment (1 vehicle, 2 restaurants)
2. **Phase 2**: Intermediate complexity (10 vehicles, 20 restaurants)
3. **Phase 3**: Full-scale environment (160 vehicles, 320 restaurants)

### Heuristic Algorithms

#### ACA (Anticipatory Customer Assignment)
- **Time Buffer Strategy**: Uses configurable delay buffers (e.g., 17 minutes)
- **Future Order Prediction**: Anticipates incoming orders based on historical patterns
- **Insertion Cost Analysis**: Evaluates cost of adding orders to existing routes

#### Fastest Vehicle
- **Greedy Assignment**: Assigns each order to nearest available vehicle
- **Minimal Computation**: Fast decision making for high-throughput scenarios

## Data Integration

### Meituan Dataset
- **Real-world Performance Data**: 22 districts × 8 days of operational data
- **Ground Truth Benchmarking**: Validates simulation accuracy
- **Key Metrics**: 86.6% on-time delivery rate baseline

### Data Processing Pipeline
- **Order Generation**: Realistic order patterns based on Meituan data
- **Geographic Distribution**: Accurate restaurant and customer locations
- **Demand Patterns**: Time-varying order intensity throughout the day

## Performance Results

### Benchmarking Findings (Filtered Data)

| Method | On-Time Rate | Gap vs Meituan |
|--------|-------------|----------------|
| Meituan Baseline | 86.6% | - |
| ACA (Buffer=17) | 43.0% | -43.6pp |
| Fastest Vehicle | 43.1% | -43.5pp |
| RL-ACA | 62.5%* | -24.1pp |

*After implementing safety fallback mechanisms

### Key Insights
- **Significant Performance Gap**: All simulated algorithms underperform real-world operations
- **RL Potential**: RL-ACA shows promise but requires careful tuning
- **Data Quality**: Filtering problematic datasets crucial for meaningful analysis
- **Safety Mechanisms**: Essential for RL deployment in production

## Research Contributions

### 1. Comprehensive Benchmarking Framework
- First systematic comparison of RMDP algorithms against real-world data
- Rigorous statistical analysis with significance testing
- Reproducible methodology for future research

### 2. Advanced RL Architecture
- Novel state representation for delivery optimization
- Phased training curriculum for complex environments
- Safety-constrained RL with heuristic fallbacks

### 3. Data Quality Analysis
- Identification of simulation accuracy issues
- Development of filtering methodologies
- Establishment of realistic performance baselines

## Installation and Usage

### Prerequisites
- Python 3.8+
- Required packages: `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `pyyaml`

### Quick Start
```bash
# Clone repository
git clone https://github.com/TristanKruse/RMDP_Algorithm.git
cd RMDP_Algorithm

# Install dependencies
pip install -r requirements.txt

# Run basic simulation
python main.py

# Train RL model
python train_rl.py

# Compare algorithms
python train_rl.py --compare-only

# Run benchmarking
python benchmarking/main_benchmarking.py
```

### Configuration
- **Environment Settings**: Modify `config.yaml` for simulation parameters
- **RL Hyperparameters**: Adjust training parameters in `train_rl.py`
- **Algorithm Selection**: Choose between FV, ACA, and RL-ACA in configuration

## Development Workflow

### 1. Algorithm Development
- Implement new algorithms in `models/` directory
- Follow existing interface patterns for compatibility
- Add comprehensive unit tests

### 2. Training Pipeline
- Use phased training approach for RL algorithms
- Monitor convergence and performance metrics
- Implement safety mechanisms for production deployment

### 3. Benchmarking Process
- Run algorithms on standardized datasets
- Collect comprehensive performance metrics
- Generate statistical significance reports

### 4. Analysis and Visualization
- Create performance comparison charts
- Generate detailed performance reports
- Conduct root cause analysis for performance gaps

## Future Work

### Short-term Improvements
1. **Enhanced Safety Mechanisms**: Improve RL fallback strategies
2. **Simulation Accuracy**: Address performance gap with real-world operations
3. **Algorithm Tuning**: Optimize hyperparameters for better performance

### Medium-term Research
1. **Multi-objective Optimization**: Balance multiple conflicting objectives
2. **Advanced RL Architectures**: Explore more sophisticated neural networks
3. **Real-time Adaptation**: Dynamic algorithm selection based on conditions

### Long-term Vision
1. **Production Deployment**: Integrate with real delivery platforms
2. **Cross-platform Generalization**: Adapt to different geographic regions
3. **Autonomous Fleet Management**: Extend to autonomous vehicle coordination

## Research Publications

This work is based on and extends research by:
- Ulmer, M. W., Thomas, B. W., Campbell, A. M., & Woyak, N. (2021). "The restaurant meal delivery problem: Dynamic pickup and delivery with deadlines and random ready times." *Transportation Science*, 55(1), 75-100.

## License and Contact

**Copyright**: © 2024. All rights reserved.
**Author**: Tristan Kruse
**Contact**: krusetristan1@gmail.com
**Repository**: [GitHub - RMDP Algorithm](https://github.com/TristanKruse/RMDP_Algorithm)

## Acknowledgments

- Meituan for providing real-world operational data
- Research community for algorithmic foundations
- Open-source contributors for essential libraries and tools

---

*This project represents a comprehensive approach to solving one of the most challenging problems in modern logistics: optimizing food delivery operations in real-time under uncertainty. The combination of advanced algorithms, real-world data integration, and rigorous performance analysis provides a solid foundation for both academic research and practical applications.*