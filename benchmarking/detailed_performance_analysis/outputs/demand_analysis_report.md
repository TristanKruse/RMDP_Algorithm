# Demand-Based Performance Analysis Report
==================================================

## Dataset Overview
- Total records: 360
- Algorithms analyzed: Fastest ACA, ACA (Buffer=17), RL-ACA
- Date range: 2022-10-17 to 2022-10-24

## Weekend vs Weekday Performance
```
                 Weekday  Weekend  Weekend_Advantage
method_display                                      
ACA (Buffer=17)    86.06    85.94              -0.12
Fastest ACA        86.06    85.97              -0.08
RL-ACA             85.24    85.14              -0.10
```

## Performance by Demand Level
```
demand_level     High Demand  Low Demand  Medium Demand
method_display                                         
ACA (Buffer=17)        67.31       95.31          91.48
Fastest ACA            67.93       95.36          91.49
RL-ACA                 71.72       95.29          91.64
```

## Performance by District Complexity
```
district_complexity  High Complexity  Low Complexity  Medium Complexity
method_display                                                         
ACA (Buffer=17)                71.70           95.24              91.15
Fastest ACA                    71.74           95.31              91.07
RL-ACA                         70.80           94.42              90.43
```

## Key Findings
### Fastest ACA
- Best day type: Weekday
- Best demand level: Low Demand
- Best complexity level: Low Complexity

### ACA (Buffer=17)
- Best day type: Weekday
- Best demand level: Low Demand
- Best complexity level: Low Complexity

### RL-ACA
- Best day type: Weekday
- Best demand level: Low Demand
- Best complexity level: Low Complexity
