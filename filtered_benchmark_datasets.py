"""
Filtered benchmark datasets utility

This module provides the exact district/day combinations that are included in the 
filtered benchmark results (fastest_aca_filtered_results.csv).

These combinations represent datasets that have been pre-filtered to remove 
problematic districts and ensure consistent evaluation across all methods.
"""

def get_filtered_district_day_combinations():
    """
    Returns the exact list of (district, day) combinations from the filtered benchmark results.
    
    This ensures consistency with the existing benchmarked methods (fastest_aca, aca_17, rl_aca)
    that have already been filtered for problematic districts.
    
    Returns:
        list: List of tuples (district, day) representing valid dataset combinations
    """
    return [
        (1, 20221017), (1, 20221018), (1, 20221019), (1, 20221020), (1, 20221021), (1, 20221022), (1, 20221023), (1, 20221024),
        (3, 20221017), (3, 20221018), (3, 20221019), (3, 20221020), (3, 20221021), (3, 20221022), (3, 20221023), (3, 20221024),
        (6, 20221017), (6, 20221018), (6, 20221019), (6, 20221020), (6, 20221021), (6, 20221022), (6, 20221023), (6, 20221024),
        (7, 20221017), (7, 20221018), (7, 20221019), (7, 20221020), (7, 20221021), (7, 20221022), (7, 20221023), (7, 20221024),
        (8, 20221017), (8, 20221018), (8, 20221019), (8, 20221020), (8, 20221021), (8, 20221022), (8, 20221023), (8, 20221024),
        (10, 20221017), (10, 20221018), (10, 20221019), (10, 20221020), (10, 20221021), (10, 20221022), (10, 20221023), (10, 20221024),
        (11, 20221017), (11, 20221018), (11, 20221019), (11, 20221020), (11, 20221021), (11, 20221022), (11, 20221023), (11, 20221024),
        (14, 20221017), (14, 20221018), (14, 20221019), (14, 20221020), (14, 20221021), (14, 20221022), (14, 20221023), (14, 20221024),
        (15, 20221017), (15, 20221018), (15, 20221019), (15, 20221020), (15, 20221021), (15, 20221022), (15, 20221023), (15, 20221024),
        (16, 20221017), (16, 20221018), (16, 20221019), (16, 20221020), (16, 20221021), (16, 20221022), (16, 20221023), (16, 20221024),
        (17, 20221017), (17, 20221018), (17, 20221019), (17, 20221020), (17, 20221021), (17, 20221022), (17, 20221023), (17, 20221024),
        (18, 20221017), (18, 20221018), (18, 20221019), (18, 20221020), (18, 20221021), (18, 20221022), (18, 20221023), (18, 20221024),
        (19, 20221017), (19, 20221018), (19, 20221019), (19, 20221020), (19, 20221021), (19, 20221022), (19, 20221023), (19, 20221024),
        (20, 20221017), (20, 20221018), (20, 20221019), (20, 20221020), (20, 20221021), (20, 20221022), (20, 20221023), (20, 20221024),
        (21, 20221017), (21, 20221018), (21, 20221019), (21, 20221020), (21, 20221021), (21, 20221022), (21, 20221023), (21, 20221024)
    ]

def get_filtered_districts():
    """
    Returns the list of districts included in the filtered benchmark.
    
    Returns:
        list: List of district IDs that passed the filtering criteria
    """
    return [1, 3, 6, 7, 8, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21]

def get_filtered_days():
    """
    Returns the list of days included in the filtered benchmark.
    
    Returns:
        list: List of day values (YYYYMMDD format) included in the benchmark
    """
    return [20221017, 20221018, 20221019, 20221020, 20221021, 20221022, 20221023, 20221024]

def get_total_dataset_count():
    """
    Returns the total number of district/day combinations in the filtered benchmark.
    
    Returns:
        int: Total number of datasets (120)
    """
    return len(get_filtered_district_day_combinations())

def is_valid_combination(district, day):
    """
    Check if a specific district/day combination is valid (included in filtered benchmark).
    
    Args:
        district (int): District ID
        day (int): Day in YYYYMMDD format
        
    Returns:
        bool: True if the combination is valid, False otherwise
    """
    return (district, day) in get_filtered_district_day_combinations()

def filter_combinations(district_day_pairs):
    """
    Filter a list of district/day pairs to only include valid combinations.
    
    Args:
        district_day_pairs (list): List of (district, day) tuples to filter
        
    Returns:
        list: Filtered list containing only valid combinations
    """
    valid_combinations = set(get_filtered_district_day_combinations())
    return [pair for pair in district_day_pairs if pair in valid_combinations]

# For convenience, create a constant with all combinations
FILTERED_BENCHMARK_COMBINATIONS = get_filtered_district_day_combinations()

if __name__ == "__main__":
    # Print summary when run as script
    combinations = get_filtered_district_day_combinations()
    print(f"Filtered benchmark dataset summary:")
    print(f"- Total combinations: {len(combinations)}")
    print(f"- Districts: {get_filtered_districts()}")
    print(f"- Days: {get_filtered_days()}")
    print(f"- First 5 combinations: {combinations[:5]}")
    print(f"- Last 5 combinations: {combinations[-5:]}")