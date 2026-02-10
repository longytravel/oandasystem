# Optimization System - Future Improvements

Ideas discussed 2026-02-02 for making optimization faster without losing power.

## Current Speed Optimizations (Already Implemented)

1. **Numba JIT Compilation** - Backtesting engine uses `@njit` for near-C speed (50-100x faster than pure Python)
2. **Vectorized Data Loading** - Parquet format, NumPy arrays, contiguous memory
3. **No Object Overhead** - Arrays and indices only, no Trade/Position classes
4. **Staged Optimization** - 5-6 dimensions at a time instead of 30-dimensional search
5. **TPE Sampler** - Bayesian optimization learns promising regions vs brute force grid search

## Future Improvements

### Easy Wins
- [ ] **Parallel trials** - Optuna `n_jobs=-1` to use all CPU cores
- [ ] **Pruning** - Kill unpromising trials early (Optuna MedianPruner)
- [ ] **Warm-start stages** - Use best params from previous stage as TPE starting point

### Medium Effort
- [ ] **Cache indicator calculations** - Pre-compute RSI, ATR for all parameter values
- [ ] **Reduce data for early stages** - 50% data for stages 1-2, full data for final only
- [ ] **Smarter parameter ranges** - ±10% narrowing instead of ±20% after each stage
- [ ] **Configurable forward split** - `--forward-months` or `--forward-pct` CLI options

### Nuclear Options (Major Effort)
- [ ] **GPU acceleration** - Numba CUDA for massively parallel backtests
- [ ] **Distributed optimization** - Optuna PostgreSQL storage for multi-machine runs

## Configuration Ideas

```bash
# Current
python scripts/run_optimization.py --trials-per-stage 3000 --final-trials 5000

# Future options
--forward-months 6        # Configurable forward test period
--forward-pct 0.25        # Or percentage-based
--n-jobs -1               # Parallel trials
--use-pruning             # Early stopping of bad trials
--early-stage-data 0.5    # Use 50% data for stages 1-3
```

## Scoring Formula Reference

```
OnTester Score = Net_Profit * sqrt(Trades) * R² * (1 - MaxDD)²
```

- Rewards profit AND consistency (R² of equity curve)
- Penalizes large drawdowns exponentially
- sqrt(trades) balances trade frequency without over-rewarding high frequency
