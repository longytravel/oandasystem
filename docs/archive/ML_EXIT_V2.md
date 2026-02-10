# ML Exit V2 - Research-Backed Optimal Exit System

## Status: COMPLETE - A/B Testing Phase
- **Created**: 2026-02-08
- **Implementation Complete**: 2026-02-08
- **Strategy**: RSI Divergence V3 (proven GREEN baseline for fair comparison)
- **Approach**: Supervised regression with optimal stopping formulation

---

## 1. Research Basis

### Literature Review

Three proven approaches to ML-based trade exits exist in the quantitative finance literature:

**1. Reinforcement Learning (RL)**
- Per-bar actions: hold/exit, learned through reward optimization
- Used by Freqtrade (Base5ActionRLEnv), academic implementations (DDQN, PPO)
- State includes unrealized P&L, position duration, market features
- Pros: Learns directly from experience, adapts to market conditions
- Cons: Complex to implement, requires massive data, unstable training
- *Reference: [Freqtrade RL](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/), [tr8dr](https://tr8dr.github.io/RLp1/)*

**2. Meta-Labeling (Lopez de Prado)**
- Triple barrier method: labels trades by which barrier (TP/SL/time) is hit first
- Secondary model learns to filter primary model's signals
- Proven in quantitative fund research
- *Reference: "Advances in Financial Machine Learning" Ch.3, [mlfinlab docs](https://www.mlfinlab.com/en/latest/labeling/tb_meta_labeling.html)*

**3. Supervised Exit Classification**
- Per-bar prediction: "Should I exit NOW or hold?"
- Direct optimal stopping formulation
- Simplest to implement, most interpretable
- **This is our approach (V2)**

### Why V1 Failed

Our V1 ML exit had three critical bugs:

| Bug | Impact | V2 Fix |
|-----|--------|--------|
| **Wrong labeling**: `hit_sl_before_tp` (whole-trade binary) and `future_r_change_5bar` (5-bar horizon) | Model can't learn per-bar exit decisions. R^2 consistently near 0. | `remaining_pnl_r`: actual value of holding vs exiting now |
| **Trade count inflation**: ML closes trade early -> position freed -> new signals fire -> more bad trades | Pass1=90 trades becomes Pass3=123. Extra 33 trades are garbage entries that drag performance down. | Cooldown period after ML exit blocks new entries |
| **Dual model gating**: Two models (regression + classification) must agree | Over-constrained when one model is weak. Calibration issues compound. | Single regression target (`remaining_pnl_r`), dual model infrastructure retained for backward compat |

### V1 Test History (5 A/B Tests, All Neutral)

| Test | Config | Baseline | ML | Conclusion |
|------|--------|----------|-----|------------|
| V3 + ML sklearn | 3yr, 5 candidates | 90.5 GREEN | 90.5 GREEN | Neutral |
| V3 + ML CatBoost | 3yr, 20 candidates | 90.5 GREEN | 90.5 GREEN | Neutral |
| V3 + ML (6 correctness fixes) | 3yr, 20 candidates | 86.3 GREEN | 86.3 GREEN | Neutral |
| V3 + ML risk_only | 4yr, holdout 12mo | 82.9 GREEN | 82.9 GREEN | Neutral |
| EMA V6 + ML | 2yr | 59.2 YELLOW | Killed (all negative) | Worse |

---

## 2. V2 Approach: Optimal Stopping Formulation

### Core Label: `remaining_pnl_r`

For each decision bar during a trade:
```
remaining_pnl_r = final_unrealized_r - current_unrealized_r
```

Where:
- `current_unrealized_r = (close[bar] - entry_price) * direction / initial_sl_dist`
- `final_unrealized_r = (close[exit_bar] - entry_price) * direction / initial_sl_dist`
- Both normalized to R-multiples (risk units)

**Interpretation:**
- `remaining_pnl_r > 0`: Holding improves the trade (price moves in our favor)
- `remaining_pnl_r < 0`: Holding makes it worse (price reverses against us)
- `remaining_pnl_r = 0`: Trade ends at current level (neutral)

**Why this works:** At each bar, the model answers: "Given the current trade state and market conditions, will this trade get better or worse from here?" This is the exact question a human trader asks when deciding to hold or exit.

### Model Architecture

- **Model**: CatBoost/sklearn regressor predicting `remaining_pnl_r`
- **Training**: Optuna-tuned hyperparameters, time-series CV (expanding window)
- **Dual model retained**: Regressor (now on `remaining_pnl_r`) + classifier (on `hit_sl_before_tp`)
  - V1 infrastructure kept for backward compatibility
  - Both models benefit from the correct regression target
  - `policy_mode` supports `dual_model`, `risk_only`, `hold_only`
- **Cooldown**: 10 bars after ML exit (configurable via `--ml-cooldown`)

### Cooldown Mechanism

After ML exit, block new entries for `cooldown_bars` (default: 10):
```
if ml_exit_triggered:
    cooldown_remaining = cooldown_bars

# In entry logic:
if cooldown_remaining > 0:
    skip_entry = True
    cooldown_remaining -= 1
```

**Why this matters:** Without cooldown, ML exit replaces one losing trade with multiple new trades. The signals that fire during the freed position were originally blocked - they are NOT selected by the optimizer and have no quality guarantee.

---

## 3. Features (16 total)

### Trade State (9 features)
| # | Feature | Description | V1? |
|---|---------|-------------|-----|
| 0 | direction | 1=long, -1=short | Yes |
| 1 | age_bars | Bars since entry | Yes |
| 2 | unrealized_r | Current P&L in R-multiples | Yes |
| 3 | distance_to_sl_r | Distance to SL in R | Yes |
| 4 | distance_to_tp_r | Distance to TP in R | Yes |
| 5 | mfe_r_running | Max favorable excursion | Yes |
| 6 | mae_r_running | Max adverse excursion | Yes |
| 7 | **mfe_drawdown_r** | mfe_r - unrealized_r (pullback from peak) | **NEW** |
| 8 | **age_ratio** | age_bars / max_hold_bars (trade progress 0-1) | **NEW** |

### Market State (5 features)
| # | Feature | Description |
|---|---------|-------------|
| 9 | atr_norm | Current ATR / entry ATR |
| 10 | trend_slope_short | 5-bar price change / ATR |
| 11 | trend_slope_long | 20-bar price change / ATR |
| 12 | momentum_short | 3-bar RSI change |
| 13 | momentum_long | 10-bar RSI change |

### Session (2 features)
| # | Feature | Description |
|---|---------|-------------|
| 14 | hour_of_day | 0-23 |
| 15 | day_of_week | 0-4 (Mon-Fri) |

### New Features Rationale

**mfe_drawdown_r** (MFE pullback): Measures how far the trade has fallen from its peak. A trade at +3R that pulls back to +1R has a drawdown of 2R. High drawdown suggests momentum reversal - the trade's best days may be behind it. This is the single most informative feature for exit timing.

**age_ratio** (trade progress): Normalizes trade age by max hold time. A trade at 80% of its max hold with negative P&L is very different from a trade at 20%. Helps the model learn time-dependent exit patterns.

---

## 4. Implementation Details

### Files Modified

| File | Change | Status |
|------|--------|--------|
| `pipeline/ml_exit/labeling.py` | Added `remaining_pnl_r` label (6th label). Fixed edge case when current_bar >= exit_bar. | DONE |
| `pipeline/ml_exit/features.py` | Added `mfe_drawdown_r` (idx 7), `age_ratio` (idx 8). 14 -> 16 features. Updated docstrings. | DONE |
| `pipeline/ml_exit/dataset_builder.py` | Pass `max_hold_bars` to trade_state dict for age_ratio feature. | DONE |
| `pipeline/ml_exit/train.py` | Train regression on `remaining_pnl_r` (falls back to `future_r_change_5bar` for V1 datasets). | DONE |
| `optimization/numba_backtest.py` | Added `ml_exit_cooldown_bars` param to `full_backtest_numba` and `full_backtest_with_telemetry`. Cooldown logic blocks entries for N bars after ML exit. | DONE |
| `pipeline/config.py` | Added `ml_exit_cooldown_bars: int = 10` to `MLExitConfig`. Added to serialization. | DONE |
| `scripts/run_pipeline.py` | Added `--ml-cooldown` CLI arg (default: 10). | DONE |
| `pipeline/stages/s3_walkforward.py` | Pass cooldown to `full_backtest_numba` when ML active. Pass `max_hold_bars` in inference trade_state. | DONE |

### Files NOT Modified (V1 infrastructure retained)

| File | Reason |
|------|--------|
| `pipeline/ml_exit/inference.py` | Still predicts dual model (hold_value + adverse_risk). V2 changes are in the training target, not inference. |
| `pipeline/ml_exit/policy.py` | Still supports `dual_model`, `risk_only`, `hold_only` modes. V2 benefits from existing policy infrastructure. |

### CLI Usage
```bash
# V2 A/B Test: Baseline (no ML)
python scripts/run_pipeline.py --pair GBP_USD --timeframe H1 --strategy rsi_v3 \
    --years 4 --holdout-months 12 --train-months 12 --test-months 12 \
    -d "V3 baseline (no ML) - V2 A/B test"

# V2 A/B Test: ML V2
python scripts/run_pipeline.py --pair GBP_USD --timeframe H1 --strategy rsi_v3 \
    --years 4 --holdout-months 12 --train-months 12 --test-months 12 \
    --ml-exit --ml-cooldown 10 \
    -d "V3 + ML V2 (remaining_pnl + cooldown) - V2 A/B test"
```

---

## 5. A/B Test Plan

**Strategy**: RSI Divergence V3 (proven 89.8 GREEN baseline)
**Data**: GBP_USD H1, 4yr with 12mo OOS holdout
**Config**: 12mo train / 12mo test windows

| Run | Config | Expected |
|-----|--------|----------|
| A (Baseline) | V3, standard SL/TP exits, no ML | ~82-86 GREEN |
| B (ML V2) | Same + ML exit with `remaining_pnl_r` labels + cooldown | Should beat A |

**Success criteria**: Run B scores higher than Run A. This proves ML exit adds value.

**Key differences from V1 tests:**
1. Correct label (`remaining_pnl_r` vs `future_r_change_5bar`) - model learns optimal stopping
2. Cooldown prevents trade inflation (10 bars default)
3. Two new features (`mfe_drawdown_r`, `age_ratio`) give model pullback and progress info

---

## 6. V1 vs V2 Summary

| Aspect | V1 (5 neutral tests) | V2 (research-backed) |
|--------|---------------------|---------------------|
| **Label** | `future_r_change_5bar` (5-bar horizon) | `remaining_pnl_r` (full remaining trade) |
| **Features** | 14 (7 trade + 5 market + 2 session) | 16 (+mfe_drawdown_r, +age_ratio) |
| **Model** | Dual: regressor + classifier | Same dual model, but regressor now predicts `remaining_pnl_r` |
| **Target** | Predict 5-bar R change AND SL hit probability | Predict total remaining value of holding |
| **Cooldown** | None (trade inflation bug) | 10 bars after ML exit (configurable) |
| **Trade count** | Inflated 20-40% (garbage entries) | Constant (cooldown blocks new entries) |
| **R^2** | Near 0 (wrong target, 5-bar too noisy) | Expected: higher (full remaining PnL is more predictable) |
