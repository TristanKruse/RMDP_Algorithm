# Postponement Analysis Investigation

This folder contains comprehensive analysis tools to investigate why RL-ACA underperforms despite learning an active postponement strategy (14.2% postponement rate vs 0% for ACA variants).

## Research Question
**Why does RL-ACA's learned postponement strategy (14.2% rate) lead to worse performance (-4.3pp on-time rate) compared to ACA variants that never postpone?**

## Key Findings from Benchmarking
- **RL-ACA**: 76.0% on-time, 463.1 min delay, 14.2% postponement
- **ACA-17**: 80.3% on-time, 396.0 min delay, 0.0% postponement  
- **Fastest ACA**: 80.3% on-time, 391.4 min delay, 0.0% postponement

## Investigation Framework

### 1. Core Hypothesis
The learned postponement strategy is **counterproductive** - RL-ACA postpones the wrong orders at the wrong times, reducing rather than improving system efficiency.

### 2. Analysis Components

#### A. Pattern Analysis (`investigate_postponement_strategy.py`)
- **Postponement Distribution**: When and how often orders are postponed
- **Context Analysis**: How postponement varies by district size, day type, operational conditions
- **Performance Correlation**: Direct relationship between postponement rate and performance metrics

#### B. Counterfactual Simulations
- **Zero Postponement**: Performance if RL-ACA never postponed (use ACA-17 as proxy)
- **Selective Postponement**: Performance with optimized postponement rules
- **Theoretical Optimal**: Upper bound estimation for perfect postponement strategy

#### C. Root Cause Investigation
- **Feature Analysis**: Which state features drive postponement decisions
- **Training Mismatch**: Differences between training and evaluation environments
- **Reward Alignment**: Whether training rewards match evaluation metrics

## Expected Insights

### Performance Improvement Potential
- **Conservative estimate**: +4.3pp (match ACA-17 by eliminating postponement)
- **Optimistic estimate**: +6-8pp (optimal selective postponement strategy)

### Strategic Recommendations
1. **Immediate**: Implement selective postponement (only high-bundling scenarios)
2. **Short-term**: Redesign reward function to penalize unproductive postponement
3. **Long-term**: Hybrid approach combining ACA efficiency with strategic RL postponement

## Files

### Scripts
- `investigate_postponement_strategy.py` - Main investigation script
- `feature_importance_analysis.py` - (Future) Analyze which features drive postponement
- `counterfactual_experiments.py` - (Future) Detailed alternative strategy testing

### Outputs
- `postponement_investigation_YYYYMMDD_HHMMSS.json` - Investigation results
- `postponement_analysis.png` - Visualization plots
- `optimization_recommendations.md` - Actionable improvement suggestions

## Usage

Run the main investigation:
```bash
python benchmarking/postponement_analysis/investigate_postponement_strategy.py
```

## Expected Timeline
- **Phase 1**: Pattern analysis and counterfactual simulation (immediate)
- **Phase 2**: Feature importance and training mismatch analysis (follow-up)
- **Phase 3**: Hybrid approach design and testing (future work)

## Academic Contribution
This investigation transforms disappointing RL results into valuable research insights about:
- Limitations of RL in operations research
- Importance of reward function design
- Value of heuristic baselines
- Training-deployment gap challenges

Even if RL improvements prove difficult, the analysis provides important negative results for the academic community about when and why RL approaches may fail in operational optimization problems.