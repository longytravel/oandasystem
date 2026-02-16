# Strategy Evolution - V1 to V6

**Last Updated:** 2026-02-16

This document tracks the evolution of trading strategies, their results, and the lessons learned at each stage.

---

## Summary Table

| Version | Score (H1) | Score (M15) | WF Pass | Stability | MC | Key Change | Lesson |
|---------|-----------|-------------|---------|-----------|-----|------------|--------|
| V1 | 93/100 GREEN | - | 100% | - | - | Original RSI divergence | Lucky candidate, fragile params |
| V2 | - | - | - | - | - | Regime + quality filters | Added complexity, no improvement |
| V3 | 89.8/100 GREEN | 87.2/100 GREEN | 100% | 87.9% | 77.4 | Multi-period RSI, hardened defaults | **Best strategy** - simplicity wins |
| V4 | 81.5/100 GREEN | - | 83.3% | 95.2% | 59.7 | BE->trail chaining, partial close | Optimizer turned OFF headline feature |
| V5 | 71.8/100 GREEN | - | 100% | 97.2% | 0.0 | Chandelier, stale exit, wider mgmt | Optimizer disabled ALL management |
| V6.1 | 72.8/100 GREEN | - | 100% | - | - | EMA cross entry | Weaker signal than RSI divergence |
| Fair Price MA | - | - | - | - | - | MA mean-reversion | Deployed EUR_JPY H1, EUR_AUD H1 |

**Key finding: V3 (simplest hardened version) > V4 > V5 (most complex). More management params does NOT equal better results.**

**Important (Feb 16):** All results above were computed without exit slippage. Re-optimization with `slippage_pips=0.5` required. M15 strategies especially affected.

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

## V3 - Stability-Hardened (2026-02-06) - RECOMMENDED

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

### Best Result (H1)
- **Score: 89.8/100 GREEN**
- Candidate #9: F/B 83.5%, WF 100%, Stability 87.9%, MC 77.4
- 14/20 WF windows passed (vs 1/20 for V1), ALL 14 GREEN, 0 fragile params
- Run: `results/pipelines/GBP_USD_H1_20260206_151217/`

### Best Result (M15)
- **Score: 87.2/100 GREEN** - 50/50 candidates GREEN
- 4yr data, 30mo train / 12mo test, 99,506 M15 candles
- 2,100+ trades, 84-86% win rate, Forward Sharpe 1.15, F/B ratio 0.844
- Best params: sl_mode=atr(300%), tp_mode=rr(7.5:1), trailing=ON(50/8pip), BE=ON(0.3x/+5pip)
- Run: `results/pipelines/GBP_USD_M15_20260210_063223/`
- **Note:** Required Sharpe-based scoring fix (see M15 Timeframe section below)

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

## V6 - EMA Cross (2026-02-07, updated 2026-02-09)

**File:** `strategies/ema_cross_ml.py` | **Params:** 6 (V6.1, down from 17 in V6.0)

### Design (Fundamentally Different)
- **Entry:** EMA fast/slow crossover (not RSI divergence)
  - Fast EMA options: [8, 13, 21]
  - Slow EMA options: [34, 55, 89]
  - All combos pre-computed for speed
- **Risk:** Standard SL/TP (same as V3)

### V6.0 -> V6.1 Fixes
V6.0 was broken: `tp_atr_mult` defaulted to 0 (no TP), ml_exit group had 8 weights all=0.

V6.1 fixes:
- Removed ml_exit group entirely (17 -> 6 params)
- Fixed `tp_atr_mult` default to 4.0
- Added 5 crossover-quality features (ema_separation, ema_fast_slope, ema_slow_slope, cross_velocity, bars_since_last_cross)

### V6.1 Results

**Baseline (no ML):** 72.8/100 GREEN
- Run: `results/pipelines/GBP_USD_H1_20260209_182619/`

**With ML entry filter:** 71.7/100 GREEN (marginally worse)
- ML SKILL DETECTED on 4/7 windows (AUC 0.613-0.761) but NO SKILL on OOS window
- ML filter actively hurts by removing good candidates on out-of-sample data
- Run: `results/pipelines/GBP_USD_H1_20260209_190306/`

### Lesson
EMA crossover is fundamentally weaker than RSI divergence on GBP_USD H1 (72.8 vs 89.8). ML entry filter can't fix a weak underlying signal. V3 RSI divergence remains the best strategy.

---

## M15 Timeframe (2026-02-10)

Testing V3 RSI divergence on M15 (15-minute candles) required a critical scoring bug fix.

### Scoring Bug Fix
The `ontester_score` uses absolute dollar profit with compound position sizing. On M15 with 480+ trades, this compounds to billions, making the forward/back ratio ~0 regardless of actual edge. **Fix:** Changed forward filter and candidate ratio from `ontester_score` to **Sharpe ratio** in `unified_optimizer.py` and `s2_optimization.py`. This is essential for any strategy with >200 trades.

### V3 RSI M15 Results
- **Score: 87.2/100 GREEN** - 50/50 candidates GREEN
- 4yr data, 30mo train / 12mo test, 99,506 M15 candles
- 2,100+ trades, 84-86% win rate, Forward Sharpe 1.15, F/B ratio 0.844
- MC: all p=0.0000, Bootstrap Sharpe CI [2.84, 7.02], 95% DD 16-20%
- Best params: sl_mode=atr(300%), tp_mode=rr(7.5:1), trailing=ON(50/8pip), BE=ON(0.3x/+5pip)
- Run: `results/pipelines/GBP_USD_M15_20260210_063223/` (166 min)

### ML Entry Filter A/B (M15)
- **Baseline:** 87.2/100 GREEN
- **ML entry filter:** 87.2/100 GREEN (identical)
- CatBoost detected marginal skill (val_AUC 0.572-0.578), kept 61-67% of signals
- Every metric identical: same scores, same Sharpe, same trades, same F/B ratio
- 7th consecutive A/B test showing ML is neutral
- Run: `results/pipelines/GBP_USD_M15_20260210_063230/` (177 min)

### Lesson
Sharpe ratio comparison is essential for high-frequency strategies. With 85% base win rate, ML entry filter can't improve — there's too little room for a classifier to add value.

---

## Fair Price MA (2026-02-11)

**File:** `strategies/fair_price_ma.py`

A moving-average-based mean-reversion strategy, separate from the RSI divergence lineage. Deployed to VPS for live paper trading on EUR_JPY H1 and EUR_AUD H1 (92.9/100 GREEN).

---

## Exit Slippage Model (2026-02-16)

### Problem
Backtest previously assumed exact SL fills. In live trading, SL orders execute as stop-to-market and slip 0.5+ pips. Combined with tight break-even triggers on M15 (ATR x 0.3 ~ 1.4 pips), BE "wins" became live losses because spread (1.5 pip) + exit slippage (0.5 pip) exceeded the locked profit.

### Fix
Added `slippage_pips` parameter to all 7 numba backtest functions (14 SL exit sites). SL exits now apply unfavorable slippage: `exit_price = pos_sl -/+ slippage_pips * pip_size`. TP exits are unaffected (limit orders fill at price). Entry spread was already modeled; this completes the cost model.

Pipeline config: `spread_pips=1.5`, `slippage_pips=0.5`. All callers updated: s2, s3, s4, s5, unified_optimizer, plot_equity, plot_stability, verify_live, ml_exit/dataset_builder.

### Impact
- **M15 severely affected:** WR 90% -> 54%, Return +32% -> -6.3% for tight BE strategies
- **H1 less affected:** Wider stops absorb 0.5 pip slippage with minimal impact
- **All pre-slippage pipeline results are invalid** and need re-optimization with slippage enabled

### Lesson
Cost modeling must match live execution. Entry slippage (spread) was modeled from the start, but exit slippage on stop orders was overlooked. Tight trade management (small BE triggers, trailing on M15) is especially vulnerable to execution costs.

---

## Overall Lessons

1. **Simplicity wins.** V3 (32 params, hardened defaults) outperformed V4 (34) and V5 (37).
2. **Defaults determine signal stage success.** If default SL/TP gives negative expectancy, signal optimization finds 0 valid results.
3. **The optimizer is the truth detector.** It consistently turns OFF features that don't help (V4 chaining, V5 everything).
4. **RSI divergence + GBP_USD** is the proven edge. EUR_USD has zero forward performance.
5. **ML can't fix a weak signal.** 7 A/B tests (entry filter) and 5 A/B tests (exit model) all neutral.
6. **More trades != better strategy.** V1 had 30 forward trades but scored 93; V4 had 93 trades but scored 81.5.
7. **Use Sharpe for scoring, not dollar profit.** Compound position sizing breaks with >200 trades.
8. **M15 works** (pre-slippage). Same strategy (V3) validates well on both H1 (89.8) and M15 (87.2). However, M15 results are invalidated by exit slippage modeling (Feb 16).
9. **Model all execution costs.** Exit slippage on SL orders (0.5+ pips) destroyed M15 profitability. Tight BE triggers are especially vulnerable.
