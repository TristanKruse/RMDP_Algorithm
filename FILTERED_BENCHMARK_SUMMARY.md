# Filtered Benchmark Dataset Summary

## Overview
This document provides the exact district/day combinations that are included in the filtered benchmark results from `fastest_aca_filtered_results.csv`. These combinations ensure consistency across all benchmarked methods by excluding problematic districts.

## Dataset Statistics
- **Total Combinations**: 120
- **Number of Districts**: 15
- **Number of Days**: 8 (1 week period)

## Included Districts
```
[1, 3, 6, 7, 8, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21]
```

## Included Days
```
[20221017, 20221018, 20221019, 20221020, 20221021, 20221022, 20221023, 20221024]
```
*Note: Days are in YYYYMMDD format representing October 17-24, 2022*

## Complete List of District/Day Combinations

### District 1 (8 combinations)
- (1, 20221017), (1, 20221018), (1, 20221019), (1, 20221020), (1, 20221021), (1, 20221022), (1, 20221023), (1, 20221024)

### District 3 (8 combinations)
- (3, 20221017), (3, 20221018), (3, 20221019), (3, 20221020), (3, 20221021), (3, 20221022), (3, 20221023), (3, 20221024)

### District 6 (8 combinations)
- (6, 20221017), (6, 20221018), (6, 20221019), (6, 20221020), (6, 20221021), (6, 20221022), (6, 20221023), (6, 20221024)

### District 7 (8 combinations)
- (7, 20221017), (7, 20221018), (7, 20221019), (7, 20221020), (7, 20221021), (7, 20221022), (7, 20221023), (7, 20221024)

### District 8 (8 combinations)
- (8, 20221017), (8, 20221018), (8, 20221019), (8, 20221020), (8, 20221021), (8, 20221022), (8, 20221023), (8, 20221024)

### District 10 (8 combinations)
- (10, 20221017), (10, 20221018), (10, 20221019), (10, 20221020), (10, 20221021), (10, 20221022), (10, 20221023), (10, 20221024)

### District 11 (8 combinations)
- (11, 20221017), (11, 20221018), (11, 20221019), (11, 20221020), (11, 20221021), (11, 20221022), (11, 20221023), (11, 20221024)

### District 14 (8 combinations)
- (14, 20221017), (14, 20221018), (14, 20221019), (14, 20221020), (14, 20221021), (14, 20221022), (14, 20221023), (14, 20221024)

### District 15 (8 combinations)
- (15, 20221017), (15, 20221018), (15, 20221019), (15, 20221020), (15, 20221021), (15, 20221022), (15, 20221023), (15, 20221024)

### District 16 (8 combinations)
- (16, 20221017), (16, 20221018), (16, 20221019), (16, 20221020), (16, 20221021), (16, 20221022), (16, 20221023), (16, 20221024)

### District 17 (8 combinations)
- (17, 20221017), (17, 20221018), (17, 20221019), (17, 20221020), (17, 20221021), (17, 20221022), (17, 20221023), (17, 20221024)

### District 18 (8 combinations)
- (18, 20221017), (18, 20221018), (18, 20221019), (18, 20221020), (18, 20221021), (18, 20221022), (18, 20221023), (18, 20221024)

### District 19 (8 combinations)
- (19, 20221017), (19, 20221018), (19, 20221019), (19, 20221020), (19, 20221021), (19, 20221022), (19, 20221023), (19, 20221024)

### District 20 (8 combinations)
- (20, 20221017), (20, 20221018), (20, 20221019), (20, 20221020), (20, 20221021), (20, 20221022), (20, 20221023), (20, 20221024)

### District 21 (8 combinations)
- (21, 20221017), (21, 20221018), (21, 20221019), (21, 20221020), (21, 20221021), (21, 20221022), (21, 20221023), (21, 20221024)

## Usage

### Using the Python Utility
The file `filtered_benchmark_datasets.py` provides several utility functions:

```python
from filtered_benchmark_datasets import (
    get_filtered_district_day_combinations,
    get_filtered_districts,
    get_filtered_days,
    is_valid_combination,
    filter_combinations,
    get_total_dataset_count
)

# Get all valid combinations
combinations = get_filtered_district_day_combinations()

# Check if a combination is valid
if is_valid_combination(district, day):
    # Run benchmark on this combination
    pass

# Filter a list of combinations to only valid ones
valid_only = filter_combinations(my_combinations)
```

### For ACA-Postponement Benchmarking
To ensure consistency with existing benchmark results, use these exact combinations when running ACA-postponement:

```python
from filtered_benchmark_datasets import get_filtered_district_day_combinations

# Run ACA-postponement only on these combinations
for district, day in get_filtered_district_day_combinations():
    # Run benchmark with (district, day)
    run_aca_postponement_benchmark(district, day)
```

## Verification
- Total combinations extracted: **120**
- Verified against original CSV: ✅ **Perfect match**
- All utility functions tested: ✅ **Working correctly**

## Notes
- These combinations represent datasets that have been pre-filtered to remove problematic districts
- All existing benchmark methods (fastest_aca, aca_17, rl_aca) have been evaluated on exactly these combinations
- Using these exact combinations ensures fair and consistent comparison across all methods