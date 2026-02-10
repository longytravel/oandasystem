# Performance Optimization Plan - OANDA Trading System

## Problem Statement
Pipeline runs take ~30+ minutes on a slow laptop. Goal: Speed up backtesting/optimization without losing quality.

## Bottleneck Analysis

| Stage | Current Time (est.) | % of Total | Root Cause |
|-------|-------------------|------------|------------|
| S2 Optimization | 15-30 min | 50-60% | Python-loop signal filtering called 35K times |
| S3 Walk-Forward | 10-20 min | 25-35% | Redundant precompute + sequential window testing |
| S4 Stability | 3-8 min | 10-15% | 480 backtest runs with Python overhead |
| S5 Monte Carlo | 1-3 min | 3-5% | Already parallelized with numba |
| S1/S6/S7 | <10s | <1% | Not a bottleneck |

**Root cause:** `filter_signals()`, `compute_sl_tp()`, and `get_management_arrays()` are Python for-loops over NamedTuples. With 35K optimization trials each iterating ~200 signals, this creates ~7M slow Python iterations.

---

## Phase 1: Vectorize Signal Processing (3-5x speedup)

### OPT-1: Vectorize signal filtering with numpy arrays

**Files:** `optimization/fast_strategy.py`, `strategies/rsi_full_v3.py` (and v1/v2)

**Change:** During `precompute()`, store signal attributes as parallel numpy arrays (not a list of NamedTuples). In `filter_signals()`, replace Python for-loop with vectorized boolean mask operations.

```python
# BEFORE (slow - Python loop):
def filter_signals(self, signals, params):
    result = []
    for s in signals:
        if s.attributes['rsi_diff'] < params['min_rsi_diff']:
            continue
        result.append(s)

# AFTER (fast - numpy vectorized):
def filter_signals_vectorized(self, params):
    mask = self._rsi_diffs >= params['min_rsi_diff']
    mask &= self._bars_between >= params['min_bars_between']
    mask &= self._bars_between <= params['max_bars_between']
    # ... all filters as vectorized ops
    return np.where(mask)[0]  # return indices
```

The `precompute()` would create arrays like `self._rsi_diffs`, `self._bars_between`, `self._signal_bars`, `self._signal_prices`, `self._signal_dirs`, etc.

**Impact:** 10-50x faster per-trial filtering. Since S2 is 50-60% of pipeline time, this alone gives ~2-3x total speedup.
**Quality:** Identical results.

### OPT-2: Vectorize SL/TP and management array computation

**Files:** `optimization/fast_strategy.py`, `strategies/rsi_full_v3.py`

**Change:** Replace per-signal Python loops in `get_all_arrays()` with numpy vectorized SL/TP computation on the filtered array.

```python
# BEFORE (slow):
for i, signal in enumerate(signals):
    sl, tp = self.compute_sl_tp(signal, params, pip_size)

# AFTER (fast):
sl_pips = np.where(sl_mode_mask, fixed_sl, atr_pips * sl_atr_pct)
sl_prices = np.where(directions == 1, prices - sl_pips * pip_size, prices + sl_pips * pip_size)
```

**Impact:** 5-10x faster SL/TP. Combined with OPT-1, trial overhead drops from ~5ms to ~0.1ms.
**Quality:** Identical results.

### OPT-5: Add `fastmath=True` to numba functions

**Files:** `optimization/numba_backtest.py`

**Change:** Add `fastmath=True` to `@njit` decorators for `basic_backtest_numba`, `full_backtest_numba`, `full_backtest_with_trades`, `calculate_r_squared`, `calculate_ontester_score`.

**Impact:** 1.3-1.5x faster numba kernels. Free performance.
**Quality:** Sub-cent PnL differences. Does not affect rankings or scores meaningfully.
**Risk:** Very low for backtesting. `fastmath` allows float reordering which is fine here.

### OPT-10: Add `--fast` CLI preset

**Files:** `scripts/run_pipeline.py`

**Change:** Add `--fast` flag that sets: trials_per_stage=2000, final_trials=5000, mc_iterations=250.

**Impact:** ~2x fewer trials. TPE converges well within 2000 for 3-6 categorical params per stage.
**Quality:** Slightly less thorough exploration. Acceptable for development/testing.

---

## Phase 2: Walk-Forward Optimization (additional 1.5-2x speedup)

### OPT-4: Precompute signals once, filter by window range

**Files:** `pipeline/stages/s3_walkforward.py`

**Change:** Instead of calling `strategy.precompute_for_dataset(df_window)` per window, precompute ONCE on the full dataset, then for each window select signals where `signal_bar >= window_start_idx and signal_bar < window_end_idx`.

Add a config flag `walkforward.precompute_once = True` (default True) for easy rollback.

**Rationale:** In live trading, the strategy sees all historical data, not just a window. Precomputing per-window is actually LESS realistic.

**Impact:** Eliminates ~120 redundant precompute calls. 3-5x speedup on S3.
**Quality:** Near-identical. Signals near window edges may differ slightly (arguably more realistic). Flag allows comparison.

### OPT-3: Parallelize walk-forward window testing

**Files:** `pipeline/stages/s3_walkforward.py`

**Change:** After OPT-1/2 make filtering vectorized (numpy releases GIL), use `ThreadPoolExecutor` to test multiple candidates across windows in parallel.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
# Test all candidates across all windows in parallel
with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as pool:
    futures = [pool.submit(test_candidate, c, windows) for c in candidates]
```

**Impact:** 2-4x speedup on S3 (limited by CPU cores).
**Quality:** Identical.

### OPT-6: Optuna improvements

**Files:** `optimization/unified_optimizer.py`

**Changes:**
1. Enable `n_jobs=-1` (all cores) for Optuna after OPT-1/2 reduce Python overhead
2. Add Optuna `MedianPruner` for early stopping of bad trials in staged optimization

**Impact:** 1.2-1.5x faster S2.
**Quality:** Pruning is well-tested in Optuna. Identical sampler behavior otherwise.

---

## Phase 3: Polish (additional 1.2-1.5x speedup)

### OPT-7: Numba-ify indicator calculations

**Files:** `strategies/rsi_full_v3.py`

**Change:** Add `@njit` to `_calc_rsi`, `_calc_atr`, `_calc_ma` methods (or extract as module-level functions). The EMA-style loops in these functions are pure Python inside numpy arrays.

**Impact:** 2-3x faster precompute (minor overall since precompute is small % of total).
**Quality:** Identical.

### OPT-8: Parallelize stability testing

**Files:** `pipeline/stages/s4_stability.py`

**Change:** Test multiple candidates in parallel using ThreadPoolExecutor.

**Impact:** 2-4x speedup on S4 (10-15% of total).
**Quality:** Identical.

### OPT-9: Float32 for market data (optional)

**Files:** `optimization/unified_optimizer.py`, `optimization/numba_backtest.py`

**Change:** Use float32 for highs/lows/closes. Halves memory bandwidth.

**Impact:** 1.1-1.2x on numba kernels.
**Quality:** 6-7 significant digits is sufficient for forex.
**Risk:** Needs explicit numba type signatures. More testing required.

---

## Expected Results

| Phase | Pipeline Time | Speedup |
|-------|-------------|---------|
| Current | ~30 min | 1x |
| Phase 1 | ~6-10 min | 3-5x |
| Phase 1+2 | ~4-6 min | 5-8x |
| All | ~3-5 min | 6-10x |

## Quality Assurance

1. **Regression test:** Run GBP_USD H1 v3 pipeline before and after, compare confidence scores
2. **Exact match validation:** OPT-1/2 must produce BIT-IDENTICAL results to current code (same filtered signals, same SL/TP values)
3. **fastmath tolerance:** Verify score differences < 0.1 points
4. **Walk-forward precompute-once:** Compare window results with precompute_once=True vs False, document any differences

---
---

# Quality Assessment: OANDA Backtesting Pipeline (Task #2)

## Executive Summary

The system implements a 7-stage backtesting pipeline that is **above average for a personal/small-team trading system** but has **several significant gaps compared to professional/institutional standards**. The architecture is well-designed with good separation of concerns, but the statistical rigor needs improvement in key areas.

**Overall Grade: B- (56/100 weighted average across components)**
- Strong architecture and workflow design
- Several meaningful overfitting protections
- Significant statistical methodology gaps
- Monte Carlo implementation is simplistic
- Scoring system has calibration issues

---

## 1. Walk-Forward Methodology -- 6/10

### Strengths:
- Rolling window approach with configurable train/test periods and roll step
- Testing fixed parameters (not re-optimizing per window) tests robustness across time
- Window pass rate threshold (75%) and minimum mean Sharpe (0.5) are reasonable gates
- Proper handling of insufficient windows (fails the stage)

### Critical Issues:

**WF windows overlap with optimization training data.** Stage 2 optimizes on the back 80% of data. Stage 3 then tests those SAME parameters across rolling windows that START within the optimization training period. Early WF windows are partially or fully in-sample, inflating pass rates.

**FIX:** WF windows should start after the optimization training period ends, or WF should only validate on the forward 20%.

**Per-window pass criteria are too lenient.** A window "passes" with ontester_score > 0 (even $1 profit). No per-window Sharpe minimum exists -- only the mean across all windows is checked.

---

## 2. Overfitting Protection -- 6.5/10

### Strengths:
- Forward/back ratio filter, forward rank weighting (2x), diversity selection
- +/-10% perturbation stability testing (industry standard range)
- Boolean/categorical parameter exemption from stability
- Stability is advisory (not a gate) -- pragmatic decision

### Critical Issues:

**min_forward_ratio = 0.15 is far too permissive.** A strategy at 15% of backtest performance is almost certainly overfit. Professional standard: 0.5-0.7 minimum.

**Stability tests only 2 neighbors per parameter.** Only +10% and -10% are tested. This misses parameters that are stable at exactly those points but fragile at other perturbation levels. Should test 4-6 levels.

**No multi-parameter perturbation.** One-at-a-time (OAT) testing misses parameter interaction fragility.

---

## 3. Monte Carlo Validity -- 4/10

### Strengths:
- Numba-parallelized, fast Fisher-Yates shuffle
- Calculates VaR, Expected Shortfall, probability metrics
- Unique per-iteration seeds for reproducibility

### Critical Issues:

**Only trade-order shuffling is implemented.** This is the weakest form of MC. Missing:
1. **Bootstrap resampling** (sample trades with replacement for CI estimation)
2. **Permutation testing** (shuffle entry times to test if edge is real vs. random)
3. **Parameter perturbation MC** (test robustness under parameter uncertainty)

Trade-order shuffling only answers "what if trades happened in different sequence" -- not "is this edge real?"

**500 iterations is below minimum.** For reliable 5th percentile estimation, 1,000 is minimum, 10,000 is recommended. The current 25th-of-500 observation has high variance.

---

## 4. Scoring System -- 5.5/10

### Strengths:
- Multi-factor scoring with 6 components, well-weighted
- Walk-forward gets highest weight (25%) -- correct priority
- Penalizes suspicious over-performance (>2.0x forward/back)

### Issues:

**GREEN threshold at 70 may be too low** given the quality of the component scores feeding into it (inflated Sharpe, weak MC, overlapping WF data).

**Sharpe component uses initial forward split Sharpe**, not the more meaningful walk-forward window Sharpe.

---

## 5. Optimization Approach -- 7/10

### Strengths:
- Optuna TPE sampler (state-of-the-art Bayesian optimization)
- Staged optimization reduces search space intelligently
- MT5-style OnTester score is a proven composite metric
- R2 of equity curve rewards smooth growth
- Combined back+forward ranking

### Issues:
- Greedy parameter locking in staged mode may miss global optima
- Optuna uses `suggest_categorical` for numeric params (loses ordering information)

---

## 6. Data Handling -- 7/10

### Strengths:
- M1-to-H1 aggregation is correct
- Data quality validation is thorough
- Smart caching avoids redundant downloads

### Issues:
- 3 years limits WF windows to 3-5 (but 5yr fails for this strategy)
- No regime detection or structural break handling
- Quote conversion rates are hardcoded approximations

---

## 7. Statistical Rigor -- 4/10

### Critical Issues:

**Sharpe ratio annualization is wrong.** Uses `sqrt(252)` assuming daily observations, but observations are TRADES (not days). For a strategy with 33 trades/year, this inflates Sharpe by ~2.8x. The reported Sharpe of 7.35 is almost certainly ~2.5 when correctly annualized.

**No multiple comparisons correction.** Testing thousands of parameter combinations without Bonferroni/FDR correction means the probability of finding "good" strategies by chance is high.

**Minimum 20 trades is too low for significance.** A 60% win rate with 20 trades has a 95% CI of [36%, 81%] -- not distinguishable from random.

**No confidence intervals on any metric.** All metrics are point estimates without uncertainty quantification.

---

## 8. Industry Comparison -- 5/10

### Missing from professional systems:
1. Slippage modeling (old engine had it, Numba engine doesn't)
2. Permutation/bootstrap significance testing
3. True holdout data set
4. Regime-aware performance analysis
5. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
6. White's Reality Check / SPA test for multiple comparisons

---

## Recommended Fixes (Priority Order)

### HIGH PRIORITY (affects result validity):
1. **Fix Sharpe annualization** in `numba_backtest.py:490,658,1005` -- use `sqrt(trades_per_year)` not `sqrt(252)` where `trades_per_year = n_trades / (n_bars / bars_per_year)`
2. **Increase MC to 1,000+ iterations** in `config.py:62` and add bootstrap resampling to `s5_montecarlo.py`
3. **Fix WF data overlap** in `s3_walkforward.py:187-228` -- windows should not overlap optimization training data
4. **Raise min_forward_ratio to 0.40-0.50** in `config.py:28`
5. **Add permutation significance test** -- verify strategy beats random entry at 95% level

### MEDIUM PRIORITY:
6. Add confidence intervals via bootstrap
7. Increase perturbation test depth to 4-6 levels in `unified_optimizer.py:903`
8. Add slippage to Numba backtest engine
9. Consider Deflated Sharpe Ratio in scoring
10. Add basic regime detection

### LOW PRIORITY:
11. Use Optuna native int/float suggestion
12. Add true holdout validation set
13. Add candidate correlation analysis
14. Use historical exchange rates for cross-currency conversion

---

## Component Scores Summary

| Component | Score | Weight | Weighted |
|---|---|---|---|
| Walk-Forward | 6/10 | 15% | 0.90 |
| Overfitting Protection | 6.5/10 | 15% | 0.98 |
| Monte Carlo | 4/10 | 15% | 0.60 |
| Scoring System | 5.5/10 | 10% | 0.55 |
| Optimization | 7/10 | 15% | 1.05 |
| Data Handling | 7/10 | 10% | 0.70 |
| Statistical Rigor | 4/10 | 15% | 0.60 |
| Industry Comparison | 5/10 | 5% | 0.25 |
| **Total** | | **100%** | **5.63/10** |

**Verdict: The system architecture is solid and the workflow is correct. The main deficiencies are statistical methodology issues that could make GREEN signals unreliable. The Sharpe ratio bug alone could turn a GREEN into a YELLOW. Adding proper significance testing would reveal whether strategy edges are real or artifacts of overfitting.**
