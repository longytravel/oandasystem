# ML Entry Filter - Design Document

## The Problem

Our strategies generate signals (e.g., EMA crossovers). Some are profitable, some aren't.
A trailing stop or ML exit tries to fix bad trades AFTER entry - but by then you're already
in a losing position. Better to never take the bad trade in the first place.

## The Idea

```
Without ML filter:
  Signal -> Open Trade -> Win or Lose (50/50)

With ML filter:
  Signal -> ML: "Good trade?" -> YES -> Open Trade -> Win (65%+)
                                -> NO  -> Skip
```

Train a model on historical trades to predict: **"Will this signal be profitable?"**
Skip signals where the model says no. Fewer trades, but higher win rate and expectancy.

## Why This Is Better Than ML Exit

| Aspect          | ML Exit (current)              | ML Entry Filter (proposed)     |
|-----------------|--------------------------------|--------------------------------|
| Prediction      | Per-bar while trade is open    | Once at entry time             |
| Training rows   | ~50-200 bars per trade         | 1 row per trade                |
| Label quality   | Noisy (future R-change)        | Clean (trade won or lost)      |
| Risk            | May kill good trades early     | Only skips trades, no harm     |
| Complexity      | Two-pass backtest, per-bar     | Simple pre-filter              |
| Failure mode    | Closes winners too early       | Trades less (still safe)       |

## Features (Available at Signal Time)

These are all computed from data available at the bar when the signal fires.
No future information is used.

### Market Context (8 features)

| # | Feature              | What It Measures                        | Calculation                          |
|---|----------------------|-----------------------------------------|--------------------------------------|
| 1 | `atr_norm`           | Current volatility level                | ATR(14) / close                      |
| 2 | `atr_ratio`          | Volatility expansion/contraction        | ATR(14) / SMA(ATR(14), 50)           |
| 3 | `trend_strength`     | How strongly price is trending          | ADX(14) / 100                        |
| 4 | `trend_direction`    | Direction of trend (long/short aligned) | direction * (close - EMA200) / ATR   |
| 5 | `momentum`           | Recent price movement                   | (close - close[20]) / ATR            |
| 6 | `rsi`                | Overbought/oversold level               | RSI(14) / 100                        |
| 7 | `bb_width`           | Bollinger squeeze (low = breakout soon) | (upper - lower) / close              |
| 8 | `range_position`     | Where price sits in recent range        | (close - low20) / (high20 - low20)   |

### Signal Quality (4 features)

| # | Feature              | What It Measures                        | Calculation                          |
|---|----------------------|-----------------------------------------|--------------------------------------|
| 9 | `bars_since_cross`   | Signal freshness (late entries worse)   | Bars since last EMA cross opposite   |
| 10| `cross_strength`     | How decisive the crossover was          | abs(EMA_fast - EMA_slow) / ATR       |
| 11| `bar_range_ratio`    | Signal bar size relative to normal      | (high - low) / ATR                   |
| 12| `close_vs_open`      | Signal bar direction matches trade dir  | direction * (close - open) / ATR     |

### Session (2 features)

| # | Feature              | What It Measures                        | Calculation                          |
|---|----------------------|-----------------------------------------|--------------------------------------|
| 13| `hour`               | Time of day (London/NY session matters) | hour / 24                            |
| 14| `day_of_week`        | Day effects (Monday gaps, Friday close) | day / 7                              |

**Total: 14 features**, all normalized to roughly [-3, +3] range.

## Label

**Binary classification: Was the trade profitable?**

```
label = 1 if trade_pnl > 0 else 0
```

Simple, clean, unambiguous. The model learns the mapping:
`market conditions at entry -> probability of profitable trade`

## Training & Inference Flow

### Within Each Walk-Forward Window

```
Window N:
  ┌─────────────────────────┬──────────────────┐
  │     Training Period      │   Test Period     │
  │     (6 months)           │   (6 months)      │
  └─────────────────────────┴──────────────────┘

  Step 1: Run backtest on Training Period (all signals, no filter)
          -> Produces ~100-200 trades with known outcomes
          -> Extract 14 features at each trade's entry bar
          -> Label: profitable (1) or not (0)

  Step 2: Train classifier (CatBoost/LightGBM) on training trades

  Step 3: Score Test Period signals
          -> For each signal in test period, compute 14 features
          -> Model outputs probability of profit: P(win)
          -> Keep signals where P(win) > threshold (e.g., 0.55)

  Step 4: Run backtest on Test Period with ONLY filtered signals
          -> Fewer trades, but higher quality
```

### Threshold Optimization

The `min_win_probability` threshold is optimized per window via Optuna:
- Range: [0.45, 0.75]
- Objective: maximize OnTester score of filtered trades
- Low threshold (0.45) = keep most signals, small improvement
- High threshold (0.75) = very selective, few but high-quality trades
- Guard: minimum 10 trades must remain after filtering

## Integration with Pipeline

### Where It Fits

```
Pipeline Stages:
  1. Data         (unchanged)
  2. Optimization (unchanged - finds best strategy params)
  3. Walk-Forward (MODIFIED - adds ML filter per window)
  4. Stability    (unchanged)
  5. Monte Carlo  (unchanged)
  6. Confidence   (unchanged)
  7. Report       (add ML Filter tab)
```

### Walk-Forward Modification (Stage 3)

For each candidate, for each WF window:

```python
# PASS 1: Backtest WITHOUT filter (training data)
train_trades = backtest(train_data, params)

# EXTRACT: Features + labels from training trades
X_train = compute_entry_features(train_data, train_signals)
y_train = (train_trades.pnl > 0).astype(int)

# TRAIN: Classifier
model = CatBoost().fit(X_train, y_train)

# SCORE: Test period signals
X_test = compute_entry_features(test_data, test_signals)
probs = model.predict_proba(X_test)[:, 1]

# FILTER: Keep high-probability signals
keep_mask = probs >= threshold
filtered_signals = test_signals[keep_mask]

# PASS 2: Backtest WITH only filtered signals
test_results = backtest(test_data, params, signals=filtered_signals)
```

### File Structure

```
pipeline/ml_filter/
    __init__.py
    features.py        # compute_entry_features() - 14 features at signal bar
    train.py           # train_entry_filter() - CatBoost classifier
    inference.py       # score_signals() - predict P(win) for new signals
```

Reuses existing infrastructure:
- `pipeline/ml_exit/train.py` patterns (Optuna CV, CatBoost backend detection)
- `s3_walkforward.py` two-pass pattern (already does Pass 1 / Pass 2 for ML exit)
- `pipeline/ml_exit/dataset_builder.py` patterns (feature extraction per trade)

## Expected Impact

### Best Case
- Win rate: 50% -> 60-65%
- Trade count: 200 -> 120-150 (fewer but better)
- Profit factor: 1.5 -> 2.0+
- Sharpe: meaningful improvement

### Worst Case (model has no edge)
- Model predicts ~50/50 for everything
- Threshold filters randomly -> fewer trades, same win rate
- Score stays similar or drops slightly due to fewer trades
- **Safe failure mode**: just trading less, not making bad decisions

### Why It Should Work
1. **Clean learning problem**: binary classification with clear labels
2. **Enough data**: 100-200 trades per training window with EMA cross
3. **Meaningful features**: volatility regime, trend alignment, momentum
4. **Known market effects**: some hours/days/conditions are genuinely better
5. **No information leakage**: all features computed from data before entry

## CLI Usage (Proposed)

```bash
# Run with ML entry filter
python scripts/run_pipeline.py --strategy ema_cross_ml --ml-filter

# Compare
python scripts/run_pipeline.py --strategy ema_cross_ml -d "baseline"
python scripts/run_pipeline.py --strategy ema_cross_ml --ml-filter -d "with ML entry filter"
```

## Metrics to Track

Per walk-forward window:
- `n_signals_total`: Total signals before filter
- `n_signals_kept`: Signals that passed filter
- `filter_rate`: % of signals removed
- `model_auc`: AUC on hold-out fold (does model discriminate?)
- `win_rate_before`: Win rate without filter
- `win_rate_after`: Win rate with filter
- `threshold_used`: Optimized probability threshold

## Success Criteria

The ML entry filter is working if:
1. Model AUC > 0.55 consistently (better than random)
2. Win rate improves by 5%+ after filtering
3. Pipeline confidence score improves vs baseline
4. Results hold in OOS windows (not just in-sample)
