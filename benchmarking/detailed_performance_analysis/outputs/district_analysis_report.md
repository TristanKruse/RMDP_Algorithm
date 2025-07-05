# District Performance Analysis Report
==================================================

## Dataset Overview
- Districts analyzed: 22
- Days of data: 8
- Algorithms compared: Fastest ACA, ACA (Buffer=17), RL-ACA, Meituan Baseline

## District Performance Summary
```
         on_time_delivery_rate        total_delay          avg_distance_per_order
                          mean    std        mean      std                   mean
district                                                                         
1                        27.94  33.38     7287.55  3344.27                   9.99
2                        82.67   1.50     3118.26   796.77                   1.85
3                        89.83   2.70      345.76   305.52                   6.40
4                        83.81   1.18     1643.62   299.61                   1.86
5                        80.87   1.52     8349.86  1166.06                   2.01
6                        92.61   2.35      278.51   265.06                   6.25
7                        86.68   2.49      926.57  1223.50                   6.62
8                        93.23   2.91      303.49   413.69                   6.44
9                        83.51   1.42     6826.30   597.26                   1.68
10                       88.76   6.35      561.26   617.89                   6.04
11                       90.92   2.93      340.81   422.72                   6.33
12                       82.95   1.57     1607.19   386.37                   1.93
13                       80.50   1.95     2360.20   320.68                   1.86
14                       87.43   3.83      408.58   248.88                   6.51
15                       92.80   3.80      198.96   218.47                   6.40
16                       90.95   5.14      199.90   166.41                   6.14
17                       87.12   3.79      326.88   177.07                   6.69
18                       96.68   0.81      100.24   117.53                   6.28
19                       93.33   3.79      226.43   300.73                   6.29
20                       92.36   1.28      152.31    83.30                   6.53
21                       86.32   2.11      309.12    97.68                   6.34
22                       82.15   5.15      132.98    69.18                   1.58
```

## Key Findings
### Best Performing Districts (On-Time Rate)
- District 18: 96.7%
- District 19: 93.3%
- District 8: 93.2%
### Worst Performing Districts (On-Time Rate)
- District 1: 27.9%
- District 13: 80.5%
- District 5: 80.9%

### RL-ACA vs Fastest ACA Performance
- Districts where RL-ACA wins: 0/22
- Districts where Fastest ACA wins: 15/22
- Average RL-ACA advantage: -0.82%