# Meituan Benchmark Data: Field Mapping and Calculations

## Overview
This document explains how the Meituan ground truth performance metrics are calculated from the raw order data, and how they map to the KPIs used in your algorithm benchmarking.

## Available Fields in Meituan Order Data

Based on the dataset structure, each order record contains:

### Timestamps (UNIX format, converted to datetime):
- `platform_order_time` - When the order was created by customer
- `estimate_meal_prepare_time` - Estimated meal preparation completion time
- `order_push_time` - When order was pushed to dispatch system  
- `dispatch_time` - When order was dispatched to a courier
- `grab_time` - When courier accepted/grabbed the order
- `fetch_time` - When courier picked up order from restaurant
- `arrive_time` - When courier delivered order to customer
- `estimate_arrived_time` - Promised/estimated delivery time

### Geographic Data:
- `sender_lat/sender_lng` - Restaurant pickup location
- `recipient_lat/recipient_lng` - Customer delivery location
- `grab_lat/grab_lng` - Courier location when accepting order

### Identifiers:
- `da_id` - District/area ID (1-22)
- `poi_id` - Restaurant identifier
- `courier_id` - Courier identifier

## Calculated Metrics

### 1. **On-time Delivery Rate**
```python
# Calculate delay for each delivered order
delay_minutes = (arrive_time - promised_delivery_time) / 60

# Count on-time vs late orders
on_time_orders = orders[delay_minutes <= 0]
late_orders = orders[delay_minutes > 0]

# Rate calculation (can be negative if more late than on-time)
on_time_rate = (len(on_time_orders) - len(late_orders)) / total_delivered * 100
```

**Promised Delivery Time Logic:**
- Use `estimate_arrived_time` if available (Meituan's promise to customer)
- Fallback: `platform_order_time + 40 minutes` (standard delivery window)

### 2. **Total Delay** 
```python
# Sum of all delays (positive for late orders, negative for early)
total_delay = sum(arrive_time - promised_delivery_time) in minutes
```

### 3. **Average Delay for Late Orders**
```python
# Only for orders with positive delay
late_orders = orders[delay_minutes > 0]
avg_delay_late = late_orders['delay_minutes'].mean()
```

### 4. **Maximum Delay**
```python
# Worst delay experienced by any order
max_delay = late_orders['delay_minutes'].max()
```

### 5. **Total Orders & Orders Delivered**
```python
total_orders = len(all_orders)
orders_delivered = len(orders[arrive_time.notna() & grab_time.notna()])
undelivered_orders = total_orders - orders_delivered
```

### 6. **Average Distance per Order**
```python
# Haversine distance calculation
distance_km = haversine_distance(sender_lat, sender_lng, recipient_lat, recipient_lng)
avg_distance = mean(all_distances)
```

### 7. **Late Orders Count**
```python
late_orders_count = len(orders[delay_minutes > 0])
```

## Metric Matching with Simulation Results

| Simulation KPI | Meituan Benchmark | Calculation Source |
|----------------|-------------------|-------------------|
| `on_time_delivery_rate` | ✅ Calculated | `arrive_time` vs `estimate_arrived_time` |
| `total_delay` | ✅ Calculated | Sum of all delivery delays |
| `avg_delay_late_orders` | ✅ Calculated | Mean delay for late orders only |
| `max_delay` | ✅ Calculated | Maximum delay experienced |
| `avg_distance_per_order` | ✅ Calculated | Haversine distance pickup→delivery |
| `total_orders` | ✅ Calculated | Count of all orders |
| `orders_delivered` | ✅ Calculated | Orders with valid `arrive_time` |
| `late_orders_count` | ✅ Calculated | Count of orders with positive delay |
| `undelivered_orders` | ✅ Calculated | Orders without `arrive_time` |
| `active_period_idle_rate` | ❌ Not available | Requires courier tracking data |

## Data Quality Considerations

### Missing Data Handling:
- Orders with `arrive_time = 0` or `NaT` are considered undelivered
- Orders with missing geographic coordinates are excluded from distance calculations
- Days are extracted from `platform_order_time` (order creation time)

### Coordinate Processing:
- Meituan coordinates are scaled by 1,000,000 (e.g., 39996108 → 39.996108°)
- Distance calculated using Haversine formula for geographic accuracy

### Time Zone:
- All timestamps are processed in UTC for consistency
- Original data appears to be in Chinese time zone

## File Structure Expected

The script expects data organized as:
```
data/meituan_data/daily_orders/
├── 20221017/
│   ├── district_1_orders.csv
│   ├── district_2_orders.csv
│   └── ...
├── 20221018/
│   └── ...
└── 20221024/
    └── ...
```

## Output Format

The benchmark extraction creates a CSV with columns:
```csv
district,day,on_time_delivery_rate,total_delay,avg_delay_late_orders,max_delay,avg_distance_per_order,total_orders,orders_delivered,late_orders_count,undelivered_orders,active_period_idle_rate
1,20221017,67.5,1250.3,15.2,45.8,2.4,598,587,89,11,0
...
```

This matches the structure of your simulation results for easy comparison and analysis.
