# Strategy Evolution - V1 to V6

**Last Updated:** 2026-02-07

This document tracks the evolution of trading strategies, their results, and the lessons learned at each stage. All results are on GBP_USD H1 with 3yr data, 6mo train/6mo test windows, 3mo roll.

---

## Summary Table

| Version | Score | WF Pass | Stability | MC | Key Change | Lesson |
|---------|-------|---------|-----------|-----|------------|--------|
| V1 | 93/100 GREEN | 100% | - | - | Original RSI divergence | Lucky candidate, fragile params |
| V2 | - | - | - | - | Regime + quality filters | Added complexity, no improvement |
| V3 | 89.8/100 GREEN | 100% | 87.9% | 77.4 | Multi-period RSI, hardened defaults | Simplicity wins, defaults matter |
| V4 | 81.5/100 GREEN | 83.3% | 95.2% | 59.7 | BE->trail chaining, partial close | Optimizer turned OFF headline feature |
| V5 | 71.8/100 GREEN | 100% | 97.2% | 0.0 | Chandelier, stale exit, wider mgmt | Optimizer disabled ALL management |
| V6 | Not tested | - | - | - | EMA cross entry + ML exit | Fundamentally different approach |

**Key finding: V3 (simplest hardened version) > V4 > V5 (most complex). More management params does NOT equal better results.**

---

## V1 - Original RSI Divergence (2026-02-05)

**File:** `strategies/rsi_full.py` | **Params:** 35

### Design
- RSI hidden divergence signals (price vs RSI swing comparison)
- 5 parameter groups: signal, filters, risk, management, time
- Full trade management: trailing, break-even, partial close, time filters

### Best Result
- **Score: 93/100 GREEN** - "Ready for live trading (small size)"
- Candidate #18: F/B ratio 6.318, WF 100%, Sharpe 7.35
- Params: rsi_period=7, rsi_oversold=25, sl_mode=atr(3.0), tp_mode=fixed(30pip)
- use_trailing=True, use_break_even=True, use_trend_filter=True(MA100)
- Run: `results/pipelines/GBP_USD_H1_20260206_011323/`

### Problems
- Only 1/20 WF windows passed initially (before pipeline fixes)
- Several fragile parameters (rsi_period, swing_strength)
- High score driven by single lucky candidate, not robust strategy

---

## V2 - Regime Detection + Quality Scoring (2026-02-03)

**File:** `strategies/rsi_full_v2.py` | **Params:** 48 (+13 new)

### Design Changes
- Added ADX trend strength filter
- Added volatility regime filter (low/normal/high)
- Added Bollinger Band squeeze detection
- Added signal quality scoring (divergence strength, swing quality, confluence)
- Added forward performance threshold to optimizer

### Result
- Never fully pipeline-tested with the 7-stage pipeline
- The additional parameters made the search space even larger
- Superseded by V3's simpler approach to the same problem

### Lesson
Adding more filters and scoring doesn't help if the core signal is fragile. Better to harden the signal itself.

---

## V3 - Stability-Hardened (2026-02-06)

**File:** `strategies/rsi_full_v3.py` | **Params:** 32 (-3 vs V1)

### Design Changes (vs V1)
- **Multi-period RSI consensus** [7, 14, 21] - replaces single `rsi_period` param
- **Adaptive swing union** [3, 5, 7] - replaces single `swing_strength` param
- **Dual-MA trend filter** (50 + 200) - replaces single `trend_ma_period` param
- **Fixed ATR multiplier** for SL - replaces `sl_atr_mult` param
- **Fixed BE trigger** - replaces `be_trigger_pips` param
- **Eliminated 5 fragile parameters** that V1 was sensitive to

### Critical Defaults
- `tp_mode='rr'`, `tp_rr_ratio=5.0`, `sl_mode='fixed'`, `sl_fixed_pips=50`
- All filters/management OFF by default (signal stage tests pure signal quality)
- Signal stage FAILS with `tp_mode='fixed'` (negative expectancy)

### Best Result
- **Score: 89.8/100 GREEN**
- Candidate #9: F/B 83.5%, WF 100%, Stability 87.9%, MC 77.4
- 14/20 WF windows passed (vs 1/20 for V1), ALL 14 GREEN, 0 fragile params
- Run: `results/pipelines/GBP_USD_H1_20260206_151217/`

### Lesson
Hardcoding known-good values (multi-period RSI, adaptive swings) and reducing the parameter space produces far more robust results than optimizing more parameters.

---

## V4 - Trade Management Optimization (2026-02-07)

**File:** `strategies/rsi_full_v4.py` | **Params:** 34 (+2 vs V3)

### Design Changes (vs V3)
- **BE->Trailing chaining** (`chain_be_to_trail`): when BE fires, auto-activate trailing
- **Partial at BE** (`partial_at_be`): partial close triggered when BE activates
- Lower TP ranges for more realistic targets
- Numba engine reordered: BE processed before trailing

### Files Modified
- `optimization/numba_backtest.py` (both functions)
- `optimization/unified_optimizer.py`
- `backtesting/engine.py`
- `pipeline/stages/s3_walkforward.py`
- `pipeline/stages/s5_montecarlo.py`
- `scripts/plot_equity.py`
- `scripts/plot_stability.py`

### Best Result
- **Score: 81.5/100 GREEN**
- Candidate #1: F/B 78.7%, WF 83.3%, Stability 95.2%, MC 59.7
- sl_mode=fixed(50pip), tp_mode=atr(6.0x), use_trailing=True(50/10), use_break_even=True(0.3x/+5pip)
- **chain_be_to_trail=FALSE**, use_partial_close=FALSE
- Back: 99 trades, 50.1% return, R2=0.926 | Forward: 93 trades, 14.7% return, Sharpe=3.86
- Run: `results/pipelines/GBP_USD_H1_20260207_135042/`

### V4 vs V3
| Metric | V3 | V4 |
|--------|-----|-----|
| Score | 89.8 | 81.5 |
| WF Pass | 100% | 83.3% |
| Stability | 87.9% | 95.2% |
| MC | 77.4 | 59.7 |
| F/B Ratio | 83.5% | 78.7% |
| Forward Trades | 30 | 93 |
| Forward Return | ~10% | 14.7% |

### Lesson
The optimizer turned OFF the headline feature (`chain_be_to_trail=FALSE`). Independent BE + trailing works better than chaining them. V4 has higher raw performance and more trades, but lower confidence due to weaker MC and WF scores.

---

## V5 - Chandelier Exit + Stale Exit (2026-02-07)

**File:** `strategies/rsi_full_v5.py` | **Params:** 37 (+3 vs V4)

### Design Changes (vs V4)
- **Chandelier exit** (ATR-based trailing from highest high)
- **Stale trade exit** (close after N bars with no progress)
- **Wider BE range** (more trigger options)
- **1R partial close** (partial at 1x risk distance)
- `chain_be_to_trail` replaced by `trail_mode` (standard/chandelier)

### Numba Signature Change
- Replaced `chain_be_to_trail` with `trail_mode, chandelier_atr_mult, atr_pips, stale_exit_bars`

### Best Result
- **Score: 71.8/100 GREEN**
- Candidate #6: WF 100%, Stability 97.2%, **MC 0.0** (p=0.11, not significant)
- sl_mode=fixed(35pip), tp_mode=atr(6.0x), ALL management OFF
- Optimizer disabled EVERY V5 feature: no trail, no BE, no partial, no stale exit
- Pure "set and forget" with 6:1 reward:risk ratio
- Run: `results/pipelines/GBP_USD_H1_20260207_165612/`

### V5 vs V3
| Metric | V3 | V5 |
|--------|-----|-----|
| Score | 89.8 | 71.8 |
| MC | 77.4 | 0.0 |
| F/B Ratio | 83.5% | 54.7% |
| Forward Trades | 30 | 46 |

### Lesson
**RSI divergence on GBP_USD H1 works best with ZERO trade management.** The optimizer consistently disables management features when given the choice. More management params = worse results. V3 (simplest) > V4 > V5 (most complex).

---

## V6 - EMA Cross + ML Exit (2026-02-07)

**File:** `strategies/ema_cross_ml.py` | **Params:** 17

### Design (Fundamentally Different)
- **Entry:** EMA fast/slow crossover (not RSI divergence)
  - Fast EMA options: [8, 13, 21]
  - Slow EMA options: [34, 55, 89]
  - All combos pre-computed for speed
- **Exit:** ML scoring with 8 OHLC features and Optuna-optimized weights
  - Features computed ONCE per dataset
  - Scores recomputed per trial (cheap vectorized multiply)
  - Direction-aware: long features flipped for short positions

### New Files
- `strategies/ema_cross_ml.py` - EMACrossMLStrategy
- `optimization/ml_features.py` - `compute_ml_features` (@njit) + `compute_ml_scores` (numpy)

### Parameter Groups
- Signal (3): ema_fast_period, ema_slow_period, min_ema_spread
- Risk (3): sl_mode, sl_fixed_pips, tp_mode + related
- ML Exit (11): 8 feature weights + use_ml_exit, ml_min_hold, ml_threshold

### Status
- **Implemented and verified** (imports, numba compilation, backward compat)
- **NOT pipeline-tested** - needs `run_pipeline.py --strategy ema_cross_ml`
- **Superseded by ML Exit Program** - the V6 approach (Optuna-optimized linear weights) is too simple. The ML Exit Program proposes proper supervised models (CatBoost/LightGBM) trained on labeled trade data.

### Lesson
V6 was a good experiment that confirmed the architecture works (ML arrays in numba, features pre-computed). But optimizing ML weights inside Optuna alongside entry params conflates two different problems. The ML Exit Program correctly separates model training from pipeline optimization.

---

## Overall Lessons

1. **Simplicity wins.** V3 (32 params, hardened defaults) outperformed V4 (34) and V5 (37).
2. **Defaults determine signal stage success.** If default SL/TP gives negative expectancy, signal optimization finds 0 valid results.
3. **The optimizer is the truth detector.** It consistently turns OFF features that don't help (V4 chaining, V5 everything).
4. **RSI divergence + GBP_USD H1** is the proven edge. EUR_USD has zero forward performance.
5. **ML exits need proper ML.** Linear weight optimization in Optuna is not enough. Need supervised models with train/predict separation.
6. **More trades != better strategy.** V1 had 30 forward trades but scored 93; V4 had 93 trades but scored 81.5.
