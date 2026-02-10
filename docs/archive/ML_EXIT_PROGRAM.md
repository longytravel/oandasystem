# ML Exit Program - Implementation Status

**Last Updated:** 2026-02-08
**Based on:** [EXIT_FIRST_ML_DEVELOPMENT_WRITEUP.md](EXIT_FIRST_ML_DEVELOPMENT_WRITEUP.md) (original reference)
**Status:** Sprints 1-4 COMPLETE. Sprint 4.5 (correctness fixes) COMPLETE. Sprint 5 (OOS+risk-only) COMPLETE. Sprint 6 (V2 optimal stopping) COMPLETE. A/B testing in progress.
**V2 Documentation:** See [ML_EXIT_V2.md](ML_EXIT_V2.md) for the research-backed V2 system.

---

## Overview

Shift strategy complexity from entries to exits using a constrained ML exit layer. Keep entries simple and deterministic (RSI divergence or EMA cross). Build a supervised ML model that decides WHEN to exit trades.

### Core Principle
```
Layer 1: Hard Risk Envelope (always on - SL, max DD, max loss)
Layer 2: Deterministic Exit (fallback - fixed SL/TP, time exit)
Layer 3: ML Exit Policy (conditional override within constraints)
```

The ML layer can ONLY tighten stops, trigger exits, or suggest partial closes. It can NEVER widen a stop or increase risk.

---

## What We Learned From V1-V6

1. RSI divergence entry on GBP_USD H1 has a proven edge (V3: 89.8/100 GREEN)
2. Adding rule-based exit management made things WORSE (V5: 71.8, all management disabled)
3. Our V6 attempt (Optuna-optimized linear weights) was too simple for real ML
4. The optimizer consistently turns OFF management features - suggesting exits should be data-driven, not rule-based

---

## Sprint 1: Integrity Foundation -- COMPLETE

**Completed:** 2026-02-07

### What was done:
- KI-2, KI-3, KI-4, KI-5, KI-9, KI-10, KI-11, KI-13 verified as FIXED in code
- Added `full_backtest_with_telemetry()` to `optimization/numba_backtest.py`
  - Returns: exit_reasons, bars_held, entry/exit bar indices, mfe_r, mae_r per trade
- Updated `s5_montecarlo.py` `_collect_trade_details()` to use telemetry function
- Regression test: V3 pipeline → **90.5/100 GREEN** (20/20 candidates)
  - Run dir: `results/pipelines/GBP_USD_H1_20260207_221100/`

---

## Sprint 2: Exit Dataset Pipeline -- COMPLETE

**Completed:** 2026-02-07

### Files Created:
| File | Purpose |
|------|---------|
| `pipeline/ml_exit/__init__.py` | Package init |
| `pipeline/ml_exit/features.py` | 14 trade-state + market + session features |
| `pipeline/ml_exit/labeling.py` | 5 supervised targets |
| `pipeline/ml_exit/dataset_builder.py` | Per-(trade, bar) decision DataFrame builder |

### Feature Set (16 features, V2):

**Trade-state (9):** direction, age_bars, unrealized_r, distance_to_sl_r, distance_to_tp_r, mfe_r_running, mae_r_running, **mfe_drawdown_r** (V2), **age_ratio** (V2)

**Market (5):** atr_norm, trend_slope_short (5-bar), trend_slope_long (20-bar), momentum_short (3-bar RSI change), momentum_long (10-bar RSI change)

**Session (2):** hour_of_day, day_of_week

### Labels (6):
- `future_r_change_5bar` / `future_r_change_10bar` (V1 regression targets)
- `hit_sl_before_tp` (binary classification)
- `optimal_exit_bar`, `bars_to_exit` (analysis)
- **`remaining_pnl_r`** (V2 optimal stopping label - primary regression target)

### Dataset Stats:
- Typical size: 800-2300 rows per WF window (from 88-221 trades)
- Each row = one (trade, bar) decision point
- Deterministic rebuild via `build_dataset_hash()`

---

## Sprint 3: Supervised Baseline Models -- COMPLETE

**Completed:** 2026-02-07

### Files Created:
| File | Purpose |
|------|---------|
| `pipeline/ml_exit/train.py` | CatBoost > LightGBM > sklearn training with Optuna |
| `pipeline/ml_exit/inference.py` | Model prediction with confidence scoring |
| `pipeline/ml_exit/policy.py` | HOLD/EXIT policy + per-bar score array generation |

### Architecture:
- **Two models** trained per WF window:
  1. `hold_value_model`: GBM regression on `future_r_change_5bar`
  2. `adverse_risk_model`: GBM classification on `hit_sl_before_tp`
- **Backend priority**: CatBoost > LightGBM > sklearn GradientBoosting
- **Hyperparameter tuning**: Optuna with time-series CV (expanding window)
- **Policy**: EXIT when `hold_value < 0.0` AND `adverse_risk > 0.5` (both conditions must be met)

### Critical Bug Fix: 0 Exit Signals

**First run produced 0 exit signals.** Root causes:
1. `min_hold_value=-0.5` but regressor predicts in [-0.38, +0.47] — threshold unreachable
2. sklearn GBM `predict_proba` outputs binary 0/1, not calibrated probabilities (AUC=0.999)
3. AND policy with too-strict thresholds = zero intersection

**Fix applied:**
- Changed thresholds: `min_hold_value=0.0, max_adverse_risk=0.5`
- After fix: ~28% of decision rows trigger exits, ~47% of trades affected
- **Key lesson**: Always check prediction distributions before setting thresholds!

---

## Sprint 4: Pipeline Integration -- COMPLETE

**Completed:** 2026-02-08

### Files Modified:
| File | Changes |
|------|---------|
| `pipeline/config.py` | Added `MLExitConfig` dataclass (10 fields) |
| `scripts/run_pipeline.py` | Added `--ml-exit` CLI flag |
| `pipeline/stages/s3_walkforward.py` | ML training per WF window, two-pass inference |
| `pipeline/report/data_collector.py` | ML diagnostics data collection |
| `pipeline/report/chart_generators.py` | Feature importance + window metrics charts |
| `pipeline/report/html_builder.py` | Conditional ML Exit tab |

### How It Works:
```
For each WF window:
  1. Train ML models on all data before test window
  2. Pass 1: Run backtest WITHOUT ML → get trade entry/exit positions
  3. Pass 2: For each (trade, bar), compute 14 features → predict → apply policy
  4. Map per-decision scores to per-bar ML score arrays
  5. Pass 3: Re-run backtest WITH ML score arrays → numba engine exits when score > threshold
```

### CLI Usage:
```bash
# V3 with ML exit
python scripts/run_pipeline.py --pair GBP_USD --timeframe H1 --strategy rsi_v3 --years 3 --ml-exit

# Fast mode for testing
python scripts/run_pipeline.py --pair GBP_USD --timeframe H1 --strategy rsi_v3 --years 3 --ml-exit --fast --top-n 5
```

### Pipeline Test Results:

| Run | Backend | Score | Rating | Candidates | Time | Run Dir |
|-----|---------|-------|--------|------------|------|---------|
| V3 Regression (no ML) | - | 90.5/100 | GREEN | 20/20 GREEN | 17 min | `GBP_USD_H1_20260207_221100` |
| V3 + ML Exit | sklearn | 90.5/100 | GREEN | 5/5 GREEN | 82 min | `GBP_USD_H1_20260207_234046` |
| V3 + ML Exit | CatBoost | 90.5/100 | GREEN | 19/20 GREEN | 201 min | `GBP_USD_H1_20260208_071943` |

### CatBoost 20-Candidate Results (Latest):

| Tier | Score | Candidates | WF Rate | F/B Ratio | Notes |
|------|-------|------------|---------|-----------|-------|
| Top | 90.5 | #2,3,5,6,7,8,11 (7) | 100% | 1.355 | Sharpe 1.56-1.73 |
| High | 86.3 | #4, 9 (2) | 83% | 1.355 | 1 WF window missed |
| Mid | 83.0 | #14 (1) | 83% | 1.749 | Different param cluster |
| Lower | 80.2 | #10, 12, 13 (3) | 83% | 0.719 | Weaker forward perf |
| Base | 79.5 | #15-20 (6) | 83% | 0.623 | Weakest F/B ratio |
| Failed | - | #1 (1) | 50% | - | Optuna NaN bug (low trades) |

### ML Model Observations:
- **Hold value R²**: -0.27 to +0.15 (weak — model struggles to predict future R-change)
- **Risk AUC**: 0.95-1.0 (strong but may be overfitting)
- **Exit signal rate**: 0-340 per window (highly variable by candidate and market regime)
- **Impact**: ML exit produces same top score as no-ML (90.5) across all three runs — ML signals not yet adding measurable value but also not hurting
- **CatBoost vs sklearn**: CatBoost produces calibrated probabilities (not binary 0/1) but outcomes are identical
- **Known bug**: Optuna NaN failures for low-trade-count candidates (all 30 trials fail during hold_value tuning)

---

## Sprint 4.5: ML Exit Correctness Fixes -- COMPLETE

**Completed:** 2026-02-08

Six correctness fixes applied after analyzing why V1 ML exit was consistently neutral:

| Fix | Issue | Change |
|-----|-------|--------|
| 1. Signal alignment | trade_i != signal_i when signals skipped | Added `signal_indices` array to telemetry return |
| 2. hour_of_day | Inference used zeros, training used real hours | Fixed inference to use `df.index.hour` |
| 3. min_confidence | Config value defined but never passed | Wired `config.ml_exit.min_confidence` into policy |
| 4. Single threshold | Dual gating (policy 0.5 + engine 0.6) | Policy outputs binary 1.0/0.0 |
| 5. Diagnostic | No visibility into ML impact | Pass1 vs Pass3 trade count comparison |
| 6. Optuna guard | <50 samples caused NaN failures | Returns None for <50 samples |

**Result:** Still neutral (86.3 baseline = 86.3 ML). Root cause: hold_value R^2 near 0.

---

## Sprint 5: OOS Holdout + Risk-Only Policy -- COMPLETE

**Completed:** 2026-02-08

| Work Stream | Change |
|-------------|--------|
| WS1: holdout_months | `config.data.holdout_months` computes `back_ratio`. With `--years 4 --holdout-months 12`, 1 true OOS window. |
| WS2: policy_mode | `risk_only`/`hold_only`/`dual_model` options. Risk-only skips dead hold_value model. |
| Window tolerance | `_generate_windows()` allows 7-day tolerance, clamps test_end to data_end. |

**Result:** Baseline 82.9 GREEN = ML risk_only 82.9 GREEN. Identical. ML exit signals fire but natural exits (7.5:1 R:R + ATR SL) dominate.

**CONCLUSION after 5 V1 A/B tests:** ML exit V1 is a dead end for RSI divergence / GBP_USD H1. Root cause: wrong labeling (5-bar horizon too noisy, R^2 near 0).

---

## Sprint 6: V2 Optimal Stopping Formulation -- COMPLETE

**Completed:** 2026-02-08

Based on literature review (RL, meta-labeling, supervised classification), implemented research-backed V2 system. See [ML_EXIT_V2.md](ML_EXIT_V2.md) for full documentation.

### Key Changes:
1. **New label**: `remaining_pnl_r = final_unrealized_r - current_unrealized_r` (optimal stopping)
2. **Cooldown**: Block new entries for 10 bars after ML exit (prevents trade inflation)
3. **New features**: `mfe_drawdown_r` (pullback from peak), `age_ratio` (trade progress)
4. **Feature count**: 14 -> 16

### A/B Test: In Progress
- Config: `--years 4 --holdout-months 12 --train-months 12 --test-months 12`
- Baseline vs ML V2 with cooldown

---

## Future Work

### Shadow/Paper Trading
- Live paper trading integration with ML exit
- Monitoring dashboard for ML model drift
- A/B comparison framework (ML vs deterministic)

---

## MLExitConfig Defaults (pipeline/config.py)

```python
@dataclass
class MLExitConfig:
    enabled: bool = False
    n_optuna_trials: int = 30
    cv_folds: int = 5
    early_stopping_rounds: int = 20
    min_hold_value: float = 0.0      # Exit if hold_value < this
    max_adverse_risk: float = 0.5    # Exit if adverse_risk > this
    min_confidence: float = 0.3
    ml_min_hold_bars: int = 3
    ml_exit_threshold: float = 0.5
    retrain_per_window: bool = True
    policy_mode: str = 'dual_model'  # 'dual_model', 'risk_only', 'hold_only'
    ml_exit_cooldown_bars: int = 10  # V2: blocks entries after ML exit
```

---

## File Inventory

### New Files (7)
```
pipeline/ml_exit/__init__.py         # Package init
pipeline/ml_exit/features.py         # 16 features (9 trade-state + 5 market + 2 session)
pipeline/ml_exit/labeling.py         # 6 supervised targets (incl. remaining_pnl_r)
pipeline/ml_exit/dataset_builder.py  # Decision-row dataset extraction
pipeline/ml_exit/train.py            # GBM training (CatBoost > LightGBM > sklearn)
pipeline/ml_exit/inference.py        # Prediction + confidence scoring
pipeline/ml_exit/policy.py           # HOLD/EXIT policy + score array generation
```

### Modified Files (8)
```
optimization/numba_backtest.py       # Added full_backtest_with_telemetry()
pipeline/stages/s5_montecarlo.py     # Uses telemetry in trade details
pipeline/stages/s3_walkforward.py    # ML exit integration (3 new methods)
pipeline/config.py                   # MLExitConfig dataclass
scripts/run_pipeline.py              # --ml-exit flag
pipeline/report/data_collector.py    # ML data collection
pipeline/report/chart_generators.py  # ML charts (feature importance, window metrics)
pipeline/report/html_builder.py      # ML Exit tab (conditional)
```

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-07 | Adapted 12-week plan to 5-week | Original scoped for 5-person team |
| 2026-02-07 | Deferred sequence/RL models | Need supervised baseline proof first |
| 2026-02-07 | Keep V6 ML arrays in numba | Already built, backward compatible |
| 2026-02-07 | Start with ~14 features | Simple first, add features based on importance |
| 2026-02-07 | CatBoost/LightGBM over deep learning | Better with small datasets |
| 2026-02-08 | AND policy (not OR) with reachable thresholds | OR triggers 78% exits (too aggressive). AND with hold<0 AND risk>0.5 gives 28% (selective) |
| 2026-02-08 | Two-pass backtest for inference | Can't compute trade-state features without knowing trade positions first |
| 2026-02-08 | sklearn backend (fallback) works but is suboptimal | Binary predict_proba, no calibration. CatBoost would be better |
| 2026-02-08 | CatBoost 1.2.8 installed and tested | Calibrated probabilities, 20-candidate run, same 90.5 top score |
| 2026-02-08 | ML exit is neutral (not helping, not hurting) | hold_value R² near 0 means regression model can't predict future R-change. Need better features or different approach |
| 2026-02-08 | V2: Switch to remaining_pnl_r label | Research shows 5-bar horizon is too noisy. Full remaining PnL is the correct optimal stopping formulation. |
| 2026-02-08 | V2: Add cooldown after ML exit | Without cooldown, ML exit frees position → new signals fire → trade inflation 20-40%. 10-bar default. |
| 2026-02-08 | V2: Add mfe_drawdown_r + age_ratio features | Pullback from peak and trade progress are key signals for exit timing (14→16 features). |
| 2026-02-08 | Keep dual model infrastructure | V2 changes training target, not model architecture. Keeps backward compat with V1 datasets. |
