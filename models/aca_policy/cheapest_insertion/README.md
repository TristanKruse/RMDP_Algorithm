# RMDP Insertion

C++ implementation of cheapest insertion algorithm for the Restaurant Meal Delivery Problem.

## Installation

1. Ensure you have a C++ compiler installed:
   - Windows: Visual Studio Build Tools
   - Linux: gcc/g++
   - macOS: clang (install Xcode command line tools)

2. Install Python dependencies:
```bash
pip install pybind11 setuptools
```

3. Build the extension:
```bash
python setup.py build_ext --inplace
```

## Usage

```python
from rmdp_insertion import CheastpestInsertion, Location, Stop

# Create optimizer
optimizer = CheastpestInsertion(
    service_time=2.0,
    vehicle_speed=40.0,
    street_network_factor=1.4
)

# Use the optimizer...
```