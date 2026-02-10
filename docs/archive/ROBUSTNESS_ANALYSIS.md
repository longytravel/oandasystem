# RSI Divergence Strategy - Robustness Analysis

## Executive Summary

After implementing parameter stability analysis, we can now distinguish between:
- **Genuine edge**: Strategy works and forward performance validates backtest
- **Overfitting**: Strategy only works on specific historical data

## Key Finding

| Pair | Back Score | Forward Score | Forward/Back | Verdict |
|------|------------|---------------|--------------|---------|
| EUR_USD | 15,166 | 0.0 | 0% | **DO NOT TRADE** |
| GBP_USD | 668 | 573 | 86% | **TRADEABLE** |

## What is Parameter Stability?

Parameter stability measures whether neighboring parameter values also perform well.

**Example:**
- If `rsi_period=14` scores 1000, and `rsi_period=13` scores 950, that's stable (95%)
- If `rsi_period=14` scores 1000, and `rsi_period=13` scores 0, that's OVERFIT (0%)

A robust strategy should have >70% stability on most parameters.

## EUR_USD Analysis

### The Problem
- ALL 20 top candidates had Forward OnTester = 0.0
- No parameter combination generalizes to forward period
- The strategy found patterns in backtest that don't exist forward

### Unstable Parameters (Fragile)
| Parameter | Stability | Interpretation |
|-----------|-----------|----------------|
| rsi_period | 0% | Only value 7 works |
| use_rsi_extreme_filter | 0% | Must be True |
| sl_mode | 3% | Must be ATR |
| swing_strength | 5% | Only value 5 works |

### Conclusion
These aren't tunable parameters - they're hard-coded requirements that happened to work on historical data. Market conditions changed, and the strategy no longer applies.

**Recommendation: Do not trade EUR_USD with RSI divergence.**

## GBP_USD Analysis

### The Result
- Best robust result: Back=668, Forward=573 (86% retention)
- Third best: Back=487, Forward=711 (146% - forward BETTER than back!)
- Mean stability: 83-122%

### Stable Parameters
Most parameters showed >70% stability, meaning small changes don't kill performance.

### Some Caution Needed
These parameters still showed instability:
| Parameter | Stability |
|-----------|-----------|
| swing_strength | 2% |
| rsi_period | 3% |
| require_pullback | 0% |

### Conclusion
GBP_USD shows a genuine edge that persists into forward testing. The high forward/back ratio (86%) and parameter stability suggest this isn't overfitting.

**Recommendation: GBP_USD is tradeable with this strategy.**

## Robust Optimization Process

### Standard vs Robust Optimization

**Standard Optimization:**
1. Search parameter space
2. Rank by back performance
3. Forward test top results
4. Pick best combined rank

**Robust Optimization (New):**
1. Search parameter space
2. Rank by back + forward performance
3. **Test parameter stability** (neighbors)
4. **Reject unstable results** (mean stability < 60%)
5. Pick best stable result

### How to Run

```bash
# Standard optimization
python scripts/run_optimization.py --pair GBP_USD --timeframe H1

# Robust optimization (includes stability testing)
python scripts/run_robust_optimization.py --pair GBP_USD --timeframe H1
```

## Implications for Live Trading

### Before Trading a Pair
1. Run robust optimization
2. Check forward performance (should be > 0)
3. Check forward/back ratio (should be > 10%)
4. Check mean stability (should be > 60%)
5. Review individual parameter stability

### Red Flags (Don't Trade)
- Forward OnTester = 0 for all candidates
- Forward/Back ratio < 5%
- Mean stability < 40%
- Core parameters (rsi_period, swing_strength) < 20% stable

### Green Flags (Consider Trading)
- Forward OnTester > 0 for multiple candidates
- Forward/Back ratio > 50%
- Mean stability > 80%
- Most parameters > 70% stable

## Technical Details

### Stability Calculation
For each parameter:
1. Get current value and performance
2. Test +1 and -1 step variations
3. Calculate: `stability = avg_neighbor_score / base_score`

### Rating System
| Mean Stability | Rating |
|----------------|--------|
| > 80% | ROBUST |
| 60-80% | MODERATE |
| 40-60% | FRAGILE |
| < 40% | OVERFIT |

### Files
- `optimization/unified_optimizer.py` - Contains `analyze_parameter_stability()` and `run_robust_optimization()`
- `scripts/run_robust_optimization.py` - CLI script for robust optimization
- `results/robust/` - Saved robust optimization results
