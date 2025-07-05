# Detailed Postponement Analysis Report
==================================================

## Data Overview
- Postponement records: 120
- Districts with postponement data: 15
- Days with postponement data: 8
- Mean postponement rate: 61.05%
- Max postponement rate: 83.16%
- Min postponement rate: 22.22%
- Districts analyzed: 15
- Total scenario records: 120
- Bundling analysis skipped due to data structure mismatch

## Key Findings
### Postponement Patterns
Districts with highest postponement rates:
- District 18: 78.42%
- District 8: 76.82%
- District 7: 75.79%
Districts with lowest postponement rates:
- District 10: 29.30%
- District 14: 35.53%
- District 21: 41.47%
- Highest postponement day: 20221023 (63.01%)
- Lowest postponement day: 20221019 (59.44%)

### Postponement Performance Analysis
**Top Performing Districts (On-Time Rate):**
- District 18: 96.5% on-time, 78.4% postponement
- District 19: 94.8% on-time, 70.2% postponement
- District 8: 94.4% on-time, 76.8% postponement

- Correlation between postponement and performance: 0.062

**Highest Postponement Districts:**
- District 18: 78.4% postponement, 96.5% on-time
- District 8: 76.8% postponement, 94.4% on-time
- District 7: 75.8% postponement, 86.9% on-time

*Note: Bundling analysis temporarily disabled due to data structure mismatch between JSON files and CSV matrix*

## Recommendations
### Optimization Opportunities
- Focus postponement strategy on districts with proven bundling success
- Investigate why some districts have higher postponement rates
- Optimize postponement timing based on daily patterns
- Improve bundling efficiency in low-performing districts