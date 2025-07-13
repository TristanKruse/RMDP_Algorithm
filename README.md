# Restaurant Meal Delivery Problem (RMDP) with Reinforcement Learning

A comprehensive Python implementation of a reinforcement learning approach to optimize food delivery logistics, addressing the Restaurant Meal Delivery Problem through an enhanced Anticipatory Customer Assignment (ACA) framework. This project introduces **RL-ACA** - a novel algorithm that uses dynamic postponement strategies learned through reinforcement learning.

## Overview

This thesis project tackles the Restaurant Meal Delivery Problem using real-world Meituan data (647,395 orders across 22 districts), developing an RL-enhanced algorithm that adapts postponement decisions to optimize delivery operations. The implementation addresses the $894 billion meal delivery industry's need for efficient solutions to dynamic challenges like stochastic demand and time-sensitive deliveries.

**Key Contributions:**
- **RL-ACA Algorithm**: Novel reinforcement learning approach for dynamic postponement in delivery assignment
- **Real-world Validation**: Comprehensive benchmarking on Meituan dataset across 176 scenarios 
- **Multi-stakeholder Optimization**: Balances efficiency gains for drivers/platforms with service quality for customers/restaurants
- **Adaptive Decision Making**: Learns optimal assignment windows through feature engineering and temporal patterns


## Results Summary

**Algorithm Performance (120 validated scenarios):**

| Method | On-Time Rate | Avg Distance/Order | Total Delay | Idle Rate |
|--------|-------------|-------------------|-------------|-----------|
| **RL-ACA** | 85.22% ±21.10% | **8.11 km ±1.28** | 826.71 min | **28.8% ±0.127** |
| ACA-17 | 86.03% ±21.03% | 8.58 km ±1.25 | 770.95 min | 30.31% ±0.129 |
| Fastest ACA | 86.04% ±21.07% | 8.59 km ±1.26 | 766.53 min | 30.31% ±0.130 |

**Key Findings:**
- **✅ H1 Confirmed**: RL-ACA reduces average distance by 5.5-5.6% (p<0.001, large effect)
- **⚠️ H2 Confirmed**: Lower on-time rates (-0.81pp) but improved efficiency in high-stress scenarios (+4.41pp)
- **📊 Postponement Strategy**: 61.05% dynamic postponement rate enables strategic bundling
- **🎯 Context-Dependent**: Excels in complex urban environments, trade-offs in simpler scenarios

## Features

- **RL-ACA Algorithm**: Dynamic postponement using Deep Q-Network with state features (time, congestion, bundling potential)
- **Comprehensive Simulation**: 12-hour operational periods with real Meituan order patterns and timing
- **Multi-Method Comparison**: Benchmarks against ACA-17, Fastest ACA with statistical significance testing
- **Real-world Integration**: Uses actual restaurant locations, delivery deadlines, and preparation times
- **Performance Analytics**: Detailed KPI tracking across district sizes, temporal patterns, and stress levels
- **Visualization Tools**: Route optimization display and performance monitoring dashboards

## Project Structure

```
thesis/
├── environment/                # Core simulation environment
│   ├── route_processing/       # Route calculation and optimization
│   ├── meituan_data/          # Real-world data integration utilities
│   ├── location_manager.py    # Geographic and distance management
│   ├── order_manager.py       # Order lifecycle and validation
│   ├── vehicle_manager.py     # Fleet management and tracking
│   └── visualization.py       # Real-time delivery visualization
├── models/                     # Algorithm implementations
│   ├── aca_policy/            # Enhanced ACA with postponement logic
│   ├── fastest_bundling/      # Order bundling optimization
│   └── fastest_vehicle/       # Baseline nearest-vehicle assignment
├── training/                   # RL training infrastructure
│   ├── config/                # Training configurations and hyperparameters
│   ├── core/                  # Episode management and statistics
│   └── utils/                 # Training utilities and metrics
├── benchmarking/              # Comprehensive performance analysis
│   ├── detailed_performance_analysis/  # District and demand analysis
│   ├── postponement_analysis/ # Postponement strategy evaluation
│   └── algorithm_benchmarking.py       # Multi-method comparison
├── data/                      # Datasets and results
│   ├── meituan_benchmark/     # Real-world Meituan data (647K orders)
│   ├── simulation_results/    # Algorithm performance outputs
│   └── processing_data_scripts/        # Data analysis and visualization
├── config.yaml               # Main simulation configuration
├── train_rl.py               # RL training entry point
└── datatypes.py              # Core data structures and types
```

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


## Usage

### Training RL-ACA Model

1. **Configure training parameters**: Edit `config.yaml` and training configs in `training/config/`

2. **Train the RL model**:
   ```bash
   python train_rl.py
   ```

3. **Monitor training progress**: Check loss convergence and postponement rate stabilization

### Running Benchmarks

1. **Single algorithm evaluation**:
   ```bash
   python benchmarking/algorithm_benchmarking.py
   ```

2. **Comprehensive analysis**:
   ```bash
   python benchmarking/detailed_performance_analysis/run_all_analyses.py
   ```

3. **Postponement strategy analysis**:
   ```bash
   python benchmarking/postponement_analysis/investigate_postponement_strategy.py
   ```

### Simulation Scenarios

- **Real-world validation**: 176 Meituan scenarios (22 districts × 8 days)
- **Filtered dataset**: 120 validated scenarios after quality control
- **Demand classification**: Low/Medium/High based on total delay terciles
- **Temporal analysis**: Weekend vs weekday performance patterns

### Key Configuration Options

```yaml
# config.yaml
simulation:
  duration_hours: 12          # 10:00-22:00 operational window
  timestep_seconds: 30        # Simulation granularity
  
environment:
  vehicle_ratio: 0.54         # Couriers per restaurant
  travel_speed_kmh: 8         # Urban delivery speed
  
rl_training:
  learning_rate: 0.0005
  discount_factor: 0.95
  batch_size: 32
  target_update_frequency: 25
```


## Research Impact & Applications

**Academic Contributions:**
- Novel RL approach to dynamic postponement in delivery logistics
- Comprehensive benchmarking framework for RMDP algorithms
- Statistical validation with real-world data across multiple contexts
- Feature engineering insights for delivery optimization

**Industry Applications:**
- **Delivery Platforms**: Enhanced assignment algorithms for complex urban environments
- **Fleet Management**: Dynamic postponement strategies for better resource utilization  
- **Urban Logistics**: Scalable solutions for high-demand delivery scenarios
- **Algorithm Integration**: RL postponement modules for existing dispatch systems

**Future Research Directions:**
- Spatial density features for enhanced decision-making
- Stochastic travel times and courier rejection modeling
- Multi-objective optimization across stakeholder priorities
- Real-world pilot testing and validation

## Technical Specifications

**Machine Learning:**
- Deep Q-Network (DQN) with experience replay
- State space: 7 features (time, congestion, bundling potential, etc.)
- Action space: Binary postponement decisions
- Reward function: Bundling optimization with penalty terms

**Data Processing:**
- 647,395 real orders from Meituan dataset
- Geographic clustering and demand pattern analysis
- Statistical significance testing (paired t-tests, effect sizes)
- Demand classification using total delay terciles

**Performance Metrics:**
- On-time delivery rate, average/maximum delay
- Distance efficiency, idle time, total system delay
- Postponement rates and bundling effectiveness
- Multi-stakeholder KPI framework

## License

This project is licensed under the MIT License - see the [License](License) file for details.

## References

1. Ulmer, M. W., Thomas, B. W., Campbell, A. M., & Woyak, N. (2021). The restaurant meal delivery problem: Dynamic pickup and delivery with deadlines and random ready times. *Transportation Science*, 55(1), 75-100.

2. Meituan Challenge Dataset. Restaurant meal delivery optimization competition data.

## Contact

**Author**: Tristan Kruse  
**Email**: krusetristan1@gmail.com  
**Institution**: Master's Thesis Project  
**Repository**: [RMDP_Algorithm](https://github.com/TristanKruse/RMDP_Algorithm)

For questions about the research, implementation details, or collaboration opportunities, please open an issue on GitHub or contact directly.
