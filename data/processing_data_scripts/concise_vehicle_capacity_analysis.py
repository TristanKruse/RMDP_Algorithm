# concise_vehicle_capacity_analysis.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Optional imports
try:
    from tqdm import tqdm
except ImportError:
    # Define a simple replacement if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from sklearn.cluster import DBSCAN
    has_sklearn = True
except ImportError:
    has_sklearn = False

class VehicleCapacityAnalyzer:
    """Analyzes vehicle capacity using multiple methods."""
    
    def __init__(self, data_dir=None, output_dir=None):
        """Initialize with data directory."""
        # Find data directory
        if data_dir is None:
            data_dir = self._find_data_directory()
        self.data_dir = data_dir
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(data_dir, "processed", "capacity_analysis")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations directory
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Find data files
        self.wave_file = self._find_wave_file()
        self.order_file = self._find_order_file()
        
    def _find_data_directory(self):
        """Find the data directory."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
        
        # Try multiple possible locations
        possible_dirs = [
            os.path.join(project_dir, "data", "meituan_data"),
            os.path.join(project_dir, "data"),
            os.path.join(os.getcwd(), "data", "meituan_data"),
            os.path.join(os.getcwd(), "data")
        ]
        
        for directory in possible_dirs:
            if os.path.exists(directory):
                logger.info(f"Found data directory: {directory}")
                return directory
                
        logger.warning("Could not find data directory. Using 'data/meituan_data'")
        return "data/meituan_data"
    
    def _find_wave_file(self):
        """Find the wave data file."""
        wave_candidates = [
            os.path.join(self.data_dir, "courier_wave_info_meituan.csv"),
            os.path.join(self.data_dir, "processed", "courier_wave_info_meituan.csv")
        ]
        
        for file_path in wave_candidates:
            if os.path.exists(file_path):
                logger.info(f"Found wave file: {file_path}")
                return file_path
                
        logger.warning("Could not find wave file. Using default path.")
        return wave_candidates[0]
    
    def _find_order_file(self):
        """Find the order data file."""
        # Look for files with specific patterns
        order_candidates = [
            os.path.join(self.data_dir, "all_waybill_info_meituan_0322.csv"),
            os.path.join(self.data_dir, "all_waybill_info_meituan.csv"),
            os.path.join(self.data_dir, "processed", "all_waybill_info_meituan_0322.csv")
        ]
        
        # Search for any matching files
        if not any(os.path.exists(f) for f in order_candidates):
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.startswith("all_waybill_info_meituan"):
                        path = os.path.join(root, file)
                        order_candidates.insert(0, path)
        
        for file_path in order_candidates:
            if os.path.exists(file_path):
                logger.info(f"Found order file: {file_path}")
                return file_path
                
        logger.warning("Could not find order file. Using default path.")
        return order_candidates[0]
    
    def analyze_wave_based(self):
        """Original method: Count all orders in a wave."""
        logger.info("Running wave-based capacity analysis...")
        
        try:
            # Load wave data
            waves_df = pd.read_csv(self.wave_file)
            logger.info(f"Loaded {len(waves_df)} waves")
            
            # Extract order counts
            waves_df['order_count'] = waves_df['order_ids'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).strip() else 0
            )
            
            # Filter out empty waves and calculate per-courier statistics
            waves_df = waves_df[waves_df['order_count'] > 0]
            courier_stats = waves_df.groupby('courier_id').agg({
                'order_count': ['max', 'mean', 'median', 'count']
            }).reset_index()
            
            courier_stats.columns = [
                'courier_id', 'max_orders', 'avg_orders', 
                'median_orders', 'wave_count'
            ]
            
            # Filter couriers with too few waves
            courier_stats = courier_stats[courier_stats['wave_count'] >= 3]
            
            if courier_stats.empty:
                logger.warning("No couriers with sufficient data")
                return None
                
            # Calculate results
            return {
                'max': courier_stats['max_orders'].max(),
                'mean': courier_stats['max_orders'].mean(),
                'median': courier_stats['max_orders'].median(),
                'p90': np.percentile(courier_stats['max_orders'], 90),
                'description': "Based on total orders in each wave",
                'courier_count': len(courier_stats),
                'distribution': courier_stats['max_orders'].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in wave-based analysis: {str(e)}")
            return None
    
    def analyze_order_timing(self):
        """Analyze actual vehicle capacity using order timing within waves."""
        logger.info("Running order timing capacity analysis...")
        
        try:
            # Load data
            waves_df = pd.read_csv(self.wave_file)
            orders_df = pd.read_csv(self.order_file)
            
            # Check required columns
            required_cols = ['grab_time', 'fetch_time', 'arrive_time', 'waybill_id']
            if not all(col in orders_df.columns for col in required_cols):
                logger.warning("Missing required columns for timing analysis")
                return None
            
            # Filter orders
            orders_df = orders_df[orders_df['is_courier_grabbed'] == 1]
            
            # Process waves
            max_concurrent_orders = []
            processed_waves = 0
            
            for _, wave in tqdm(waves_df.iterrows(), desc="Processing waves"):
                if pd.isna(wave['order_ids']) or str(wave['order_ids']).strip() == '':
                    continue
                
                # Extract order IDs
                order_ids = str(wave['order_ids']).split(',')
                
                # Create timeline of events
                events = []
                for order_id in order_ids:
                    order_id = order_id.strip()
                    matching_orders = orders_df[orders_df['order_id'].astype(str) == order_id]
                    
                    for _, order in matching_orders.iterrows():
                        if order['fetch_time'] > 0 and order['arrive_time'] > 0:
                            events.append((order['fetch_time'], 1))  # Pickup
                            events.append((order['arrive_time'], -1))  # Delivery
                
                if len(events) < 2:
                    continue
                
                # Calculate maximum concurrent orders
                events.sort(key=lambda x: x[0])
                current_load = max_load = 0
                
                for _, change in events:
                    current_load += change
                    max_load = max(max_load, current_load)
                
                if max_load > 0:
                    max_concurrent_orders.append(max_load)
                    processed_waves += 1
            
            logger.info(f"Processed {processed_waves} waves with valid timing data")
            
            if not max_concurrent_orders:
                logger.warning("No valid timing data found")
                return None
                
            return {
                'max': max(max_concurrent_orders),
                'mean': np.mean(max_concurrent_orders),
                'median': np.median(max_concurrent_orders),
                'p90': np.percentile(max_concurrent_orders, 90),
                'description': "Based on concurrent orders within waves using pickup/delivery times",
                'wave_count': processed_waves,
                'distribution': max_concurrent_orders
            }
            
        except Exception as e:
            logger.error(f"Error in order timing analysis: {str(e)}")
            return None
    
    def analyze_pickup_delivery_windows(self):
        """Analyze capacity using pickup-delivery windows per courier."""
        logger.info("Running pickup-delivery window capacity analysis...")
        
        try:
            # Load order data
            orders_df = pd.read_csv(self.order_file)
            
            # Check required columns
            required_cols = ['courier_id', 'fetch_time', 'arrive_time']
            if not all(col in orders_df.columns for col in required_cols):
                logger.warning("Missing required columns for window analysis")
                return None
            
            # Filter valid orders
            valid_orders = orders_df[
                (orders_df['fetch_time'] > 0) & 
                (orders_df['arrive_time'] > 0) & 
                (orders_df['is_courier_grabbed'] == 1)
            ]
            
            if len(valid_orders) < 100:
                logger.warning(f"Too few valid orders ({len(valid_orders)})")
                return None
                
            logger.info(f"Found {len(valid_orders)} valid orders")
            
            # Group by courier
            courier_groups = valid_orders.groupby('courier_id')
            max_concurrent_per_courier = []
            
            for courier_id, courier_orders in tqdm(courier_groups, desc="Processing couriers"):
                if len(courier_orders) < 3:
                    continue
                
                # Create timeline of events
                events = []
                for _, order in courier_orders.iterrows():
                    events.append((order['fetch_time'], 1))  # Pickup
                    events.append((order['arrive_time'], -1))  # Delivery
                
                # Calculate maximum concurrent orders
                events.sort(key=lambda x: x[0])
                current_load = max_load = 0
                
                for _, change in events:
                    current_load += change
                    max_load = max(max_load, current_load)
                
                if max_load > 0:
                    max_concurrent_per_courier.append(max_load)
            
            logger.info(f"Analyzed {len(max_concurrent_per_courier)} couriers")
            
            if not max_concurrent_per_courier:
                logger.warning("No valid courier data")
                return None
                
            return {
                'max': max(max_concurrent_per_courier),
                'mean': np.mean(max_concurrent_per_courier),
                'median': np.median(max_concurrent_per_courier),
                'p90': np.percentile(max_concurrent_per_courier, 90),
                'description': "Based on concurrent orders per courier using pickup/delivery times",
                'courier_count': len(max_concurrent_per_courier),
                'distribution': max_concurrent_per_courier
            }
            
        except Exception as e:
            logger.error(f"Error in pickup-delivery window analysis: {str(e)}")
            return None
    
    def analyze_time_windows(self, window_minutes=15):
        """Analyze capacity using fixed time windows."""
        logger.info(f"Running time-window analysis (window size: {window_minutes} minutes)...")
        
        try:
            # Load order data
            orders_df = pd.read_csv(self.order_file)
            
            # Check required columns
            required_cols = ['courier_id', 'fetch_time', 'arrive_time']
            if not all(col in orders_df.columns for col in required_cols):
                logger.warning("Missing required columns for time-window analysis")
                return None
            
            # Filter valid orders
            valid_orders = orders_df[
                (orders_df['fetch_time'] > 0) & 
                (orders_df['arrive_time'] > 0) & 
                (orders_df['is_courier_grabbed'] == 1)
            ]
            
            if len(valid_orders) < 100:
                logger.warning(f"Too few valid orders ({len(valid_orders)})")
                return None
                
            logger.info(f"Found {len(valid_orders)} valid orders")
            
            # Ensure dt column is present
            if 'dt' not in valid_orders.columns:
                try:
                    valid_orders['dt'] = pd.to_datetime(valid_orders['fetch_time'], unit='s').dt.strftime('%Y%m%d')
                except:
                    valid_orders['dt'] = '20000101'
            
            # Group by courier and date
            grouped = valid_orders.groupby(['courier_id', 'dt'])
            max_per_window = []
            
            for (courier_id, date), group in tqdm(grouped, desc="Processing time windows"):
                if len(group) < 3:
                    continue
                
                # Get active period
                group = group.sort_values('fetch_time')
                min_time = group['fetch_time'].min()
                max_time = group['arrive_time'].max()
                
                if max_time - min_time < window_minutes * 60:
                    continue
                
                # Create time windows
                current_time = min_time
                window_size = window_minutes * 60
                
                while current_time < max_time:
                    window_end = current_time + window_size
                    
                    # Count active orders in window
                    active_orders = 0
                    for _, order in group.iterrows():
                        if (order['fetch_time'] <= window_end and 
                            order['arrive_time'] >= current_time):
                            active_orders += 1
                    
                    if active_orders > 0:
                        max_per_window.append(active_orders)
                        
                    current_time += window_size
            
            logger.info(f"Analyzed {len(max_per_window)} time windows")
            
            if not max_per_window:
                logger.warning("No valid time windows")
                return None
                
            return {
                'max': max(max_per_window),
                'mean': np.mean(max_per_window),
                'median': np.median(max_per_window),
                'p90': np.percentile(max_per_window, 90),
                'description': f"Based on {window_minutes}-minute time windows",
                'window_count': len(max_per_window),
                'distribution': max_per_window
            }
            
        except Exception as e:
            logger.error(f"Error in time-window analysis: {str(e)}")
            return None
    
    def analyze_daily_capacity(self):
        """Analyze maximum concurrent orders per courier per day."""
        logger.info("Running daily courier capacity analysis...")
        
        try:
            # Load order data
            orders_df = pd.read_csv(self.order_file)
            
            # Check required columns
            required_cols = ['courier_id', 'fetch_time', 'arrive_time']
            if not all(col in orders_df.columns for col in required_cols):
                logger.warning("Missing required columns for daily capacity analysis")
                return None
            
            # Filter valid orders
            valid_orders = orders_df[
                (orders_df['fetch_time'] > 0) & 
                (orders_df['arrive_time'] > 0) & 
                (orders_df['is_courier_grabbed'] == 1)
            ]
            
            if len(valid_orders) < 100:
                logger.warning(f"Too few valid orders ({len(valid_orders)})")
                return None
                
            # Ensure dt column is present
            if 'dt' not in valid_orders.columns:
                try:
                    valid_orders['dt'] = pd.to_datetime(valid_orders['fetch_time'], unit='s').dt.strftime('%Y%m%d')
                except:
                    valid_orders['dt'] = '20000101'
            
            # Group by courier and date
            grouped = valid_orders.groupby(['courier_id', 'dt'])
            daily_max_capacities = []
            
            for (courier_id, date), group in tqdm(grouped, desc="Processing daily courier data"):
                if len(group) < 3:
                    continue
                
                # Create timeline of events
                events = []
                for _, order in group.iterrows():
                    events.append((order['fetch_time'], 1))  # Pickup
                    events.append((order['arrive_time'], -1))  # Delivery
                
                # Calculate maximum concurrent orders
                events.sort(key=lambda x: x[0])
                current_load = max_load = 0
                
                for _, change in events:
                    current_load += change
                    max_load = max(max_load, current_load)
                
                if max_load > 0:
                    daily_max_capacities.append(max_load)
            
            logger.info(f"Analyzed {len(daily_max_capacities)} courier-days")
            
            if not daily_max_capacities:
                logger.warning("No valid daily data")
                return None
                
            return {
                'max': max(daily_max_capacities),
                'mean': np.mean(daily_max_capacities),
                'median': np.median(daily_max_capacities),
                'p90': np.percentile(daily_max_capacities, 90),
                'description': "Based on maximum concurrent orders per courier per day",
                'courier_day_count': len(daily_max_capacities),
                'distribution': daily_max_capacities
            }
            
        except Exception as e:
            logger.error(f"Error in daily capacity analysis: {str(e)}")
            return None
    
    def analyze_spatial_clusters(self, max_distance_km=1.0):
        """Analyze capacity using spatial clusters of orders."""
        logger.info("Running spatial clustering capacity analysis...")
        
        # Check if sklearn is available
        if not has_sklearn:
            logger.warning("scikit-learn package not available. Skipping spatial clustering analysis.")
            return None
        
        try:
            # Load order data
            orders_df = pd.read_csv(self.order_file)
            
            # Check required columns
            required_cols = ['courier_id', 'dt', 'sender_lat', 'sender_lng']
            if not all(col in orders_df.columns for col in required_cols):
                logger.warning("Missing required columns for spatial clustering")
                return None
            
            # Filter valid orders
            valid_orders = orders_df[
                (orders_df['sender_lat'] != 0) & 
                (orders_df['sender_lng'] != 0) & 
                (orders_df['is_courier_grabbed'] == 1)
            ]
            
            if len(valid_orders) < 100:
                logger.warning(f"Too few valid orders ({len(valid_orders)})")
                return None
                
            # Group by courier and date
            grouped = valid_orders.groupby(['courier_id', 'dt'])
            cluster_sizes = []
            
            for (courier_id, date), group in tqdm(grouped, desc="Processing spatial clusters"):
                if len(group) < 3:
                    continue
                
                # Get pickup coordinates
                pickup_coords = group[['sender_lat', 'sender_lng']].values
                
                if np.any(np.isnan(pickup_coords)) or np.any(pickup_coords == 0):
                    continue
                
                # Create distance matrix
                n = len(pickup_coords)
                dist_matrix = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(i+1, n):
                        # Simple Euclidean distance scaled to approximate km
                        lat_diff = (pickup_coords[i, 0] - pickup_coords[j, 0]) / 100000
                        lng_diff = (pickup_coords[i, 1] - pickup_coords[j, 1]) / 100000
                        dist = np.sqrt(lat_diff**2 + lng_diff**2)
                        dist_matrix[i, j] = dist
                        dist_matrix[j, i] = dist
                
                # Cluster using DBSCAN
                db = DBSCAN(eps=max_distance_km, min_samples=2, metric='precomputed').fit(dist_matrix)
                
                # Count size of each cluster
                for label in set(db.labels_):
                    if label != -1:  # Skip noise points
                        cluster_size = np.sum(db.labels_ == label)
                        if cluster_size > 1:  # Only include actual clusters
                            cluster_sizes.append(cluster_size)
            
            logger.info(f"Found {len(cluster_sizes)} spatial clusters")
            
            if not cluster_sizes:
                logger.warning("No valid spatial clusters")
                return None
                
            return {
                'max': max(cluster_sizes),
                'mean': np.mean(cluster_sizes),
                'median': np.median(cluster_sizes),
                'p90': np.percentile(cluster_sizes, 90),
                'description': f"Based on spatial clusters of orders within {max_distance_km}km",
                'cluster_count': len(cluster_sizes),
                'distribution': cluster_sizes
            }
            
        except Exception as e:
            logger.error(f"Error in spatial clustering analysis: {str(e)}")
            return None
        
    def run_analysis(self):
        """Run analysis methods and calculate consensus capacity, skipping order timing analysis."""
        logger.info("Starting vehicle capacity analysis...")
        start_time = datetime.now()
        
        # Run all analysis methods EXCEPT timing_method
        results = {
            'wave_method': self.analyze_wave_based(),
            'window_method': self.analyze_pickup_delivery_windows(),
            'time_window_method': self.analyze_time_windows(),
            'daily_courier_method': self.analyze_daily_capacity()
        }
        
        # Add spatial clustering if sklearn is available
        if has_sklearn:
            results['spatial_method'] = self.analyze_spatial_clusters()
        
        # Calculate consensus capacity
        all_p90_values = []
        for method_name, result in results.items():
            if result is not None and 'p90' in result:
                all_p90_values.append(result['p90'])
        
        consensus_capacity = int(np.ceil(np.median(all_p90_values))) if all_p90_values else None
        
        # Create visualizations
        self._create_visualizations(results, consensus_capacity)
        
        # Save results
        self._save_results(results, consensus_capacity)
        
        # Print summary
        self._print_summary(results, consensus_capacity)
        
        # Report execution time
        end_time = datetime.now()
        logger.info(f"Analysis completed in {end_time - start_time}")
        
        return consensus_capacity
    
    def _create_visualizations(self, results, consensus_capacity):
        """Create visualizations comparing methods."""
        try:
            # 1. Method comparison chart
            method_names = []
            p90_values = []
            
            for method_name, result in results.items():
                if result is not None and 'p90' in result:
                    readable_name = ' '.join(method_name.split('_')).title().replace('Method', '')
                    method_names.append(readable_name)
                    p90_values.append(float(result['p90']))
            
            if method_names:
                plt.figure(figsize=(12, 6))
                bars = plt.bar(method_names, p90_values, alpha=0.7)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.1f}', ha='center', va='bottom')
                
                # Add consensus line
                if consensus_capacity is not None:
                    plt.axhline(y=float(consensus_capacity), color='r', linestyle='--', 
                            label=f'Consensus: {consensus_capacity}')
                    plt.legend()
                
                plt.title('Vehicle Capacity Estimates by Different Methods (90th Percentile)')
                plt.xlabel('Method')
                plt.ylabel('Estimated Capacity')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                plt.savefig(os.path.join(self.viz_dir, "capacity_method_comparison.png"), dpi=300)
                plt.close()
            
            # 2. Create distribution plots for each method
            for method_name, result in results.items():
                if result is not None and 'distribution' in result and result['distribution']:
                    distribution = result['distribution']
                    
                    plt.figure(figsize=(12, 6))
                    
                    # Calculate bins
                    max_val = max(distribution)
                    min_val = min(distribution)
                    bin_width = max(1, (max_val - min_val) / 20)
                    bins = np.arange(min_val, max_val + bin_width, bin_width)
                    
                    plt.hist(distribution, bins=bins, alpha=0.7)
                    
                    # Add statistics lines
                    plt.axvline(x=np.mean(distribution), color='r', linestyle='--', 
                            label=f'Mean: {np.mean(distribution):.1f}')
                    plt.axvline(x=np.median(distribution), color='g', linestyle='-.', 
                            label=f'Median: {np.median(distribution):.1f}')
                    plt.axvline(x=np.percentile(distribution, 90), color='b', linestyle=':', 
                            label=f'90th percentile: {np.percentile(distribution, 90):.1f}')
                    
                    readable_name = ' '.join(method_name.split('_')).title().replace('Method', '')
                    plt.title(f'Distribution of Capacity Estimates: {readable_name}')
                    plt.xlabel('Capacity')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(self.viz_dir, f"distribution_{method_name}.png"), dpi=300)
                    plt.close()
        
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def _save_results(self, results, consensus_capacity):
        """Save analysis results to files."""
        try:
            import json
            
            # Helper function to convert numpy types
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return convert_to_serializable(obj.tolist())
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                else:
                    return obj
            
            # Prepare results for JSON
            json_results = {}
            
            for method_name, result in results.items():
                if result is not None:
                    # Remove distribution arrays to keep JSON manageable
                    method_result = {k: v for k, v in result.items() if k != 'distribution'}
                    
                    # Add distribution statistics
                    if 'distribution' in result and result['distribution']:
                        dist = result['distribution']
                        method_result['distribution_stats'] = {
                            'min': float(min(dist)),
                            'max': float(max(dist)),
                            'count': len(dist),
                            'percentiles': {
                                '10': float(np.percentile(dist, 10)),
                                '25': float(np.percentile(dist, 25)),
                                '50': float(np.percentile(dist, 50)),
                                '75': float(np.percentile(dist, 75)),
                                '90': float(np.percentile(dist, 90)),
                                '95': float(np.percentile(dist, 95)),
                                '99': float(np.percentile(dist, 99))
                            }
                        }
                    
                    # Convert to serializable objects
                    json_results[method_name] = convert_to_serializable(method_result)
            
            # Add consensus capacity
            json_results['consensus'] = {
                'capacity': int(consensus_capacity) if consensus_capacity is not None else None,
                'description': "Median of 90th percentile values across all methods"
            }
            
            # Save to JSON file
            json_path = os.path.join(self.output_dir, "vehicle_capacity_analysis.json")
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"Saved complete analysis results to {json_path}")
            
            # Save recommendation to text file
            with open(os.path.join(self.output_dir, "recommended_capacity.txt"), 'w') as f:
                f.write(f"Recommended vehicle capacity: {consensus_capacity}\n")
                f.write(f"Based on consensus across multiple analysis methods\n\n")
                
                # Add individual method recommendations
                f.write("Individual method recommendations (90th percentile):\n")
                for method_name, result in results.items():
                    if result is not None and 'p90' in result:
                        readable_name = ' '.join(method_name.split('_')).title().replace('Method', '')
                        f.write(f"{readable_name}: {result['p90']:.1f}\n")
        
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def _print_summary(self, results, consensus_capacity):
        """Print summary of analysis results."""
        logger.info("\n========== Vehicle Capacity Analysis Results ==========")
        
        for method_name, result in results.items():
            if result is not None:
                logger.info(f"\n--- {method_name} ---")
                logger.info(f"Description: {result.get('description', 'No description')}")
                logger.info(f"Maximum capacity: {result.get('max', 'N/A')}")
                logger.info(f"Mean capacity: {result.get('mean', 'N/A'):.2f}")
                logger.info(f"Median capacity: {result.get('median', 'N/A'):.2f}")
                logger.info(f"90th percentile capacity: {result.get('p90', 'N/A'):.2f}")
                
                count_type = next((k for k in result.keys() if k.endswith('_count')), None)
                if count_type:
                    logger.info(f"Sample size: {result.get(count_type, 'N/A')}")
            else:
                logger.warning(f"\n--- {method_name} failed or returned no results ---")
        
        logger.info(f"\n========== Consensus Capacity Estimate: {consensus_capacity} ==========")


def main():
    """Main function to run the analysis."""
    analyzer = VehicleCapacityAnalyzer()
    consensus_capacity = analyzer.run_analysis()
    
    if consensus_capacity:
        logger.info(f"\nAnalysis complete. Consensus vehicle capacity recommendation: {consensus_capacity}")
        logger.info(f"This represents a more accurate estimate than the original wave-based method.")
        logger.info(f"See {analyzer.output_dir} for detailed results and visualizations.")
    else:
        logger.warning("Analysis failed or no valid consensus was reached.")


if __name__ == "__main__":
    main()