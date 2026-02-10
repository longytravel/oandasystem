# Calculation Audit - Findings & Proposed Fixes

## Audit Scope
All mathematical calculations in the backtesting, optimization, walk-forward, stability, Monte Carlo, confidence scoring, and RSI divergence signal generation.

---

## CRITICAL ISSUES (Affect P&L / trading decisions)

### C1. Sharpe Ratio uses Population Std Dev instead of Sample Std Dev
**Files:** `optimization/numba_backtest.py` lines 486-490, 654-658, 1000-1005
**Issue:** The variance calculation uses `var / n_trades` (population variance) but should use `var / (n_trades - 1)` (sample variance / Bessel's correction) for the Sharpe ratio. With small trade counts (20-50), this underestimates standard deviation by up to 5%, artificially inflating the Sharpe ratio.
**Impact:** MEDIUM - Sharpe is used for candidate filtering (min_back_sharpe=1.0). Inflated Sharpe lets borderline candidates through.
**Fix:** Change `var / n_trades` to `var / (n_trades - 1)`. The guard `if n_trades > 1` is already there.

### C2. Walk-Forward Stage Missing `quote_conversion_rate`
**File:** `pipeline/stages/s3_walkforward.py` line 339-356
**Issue:** The `_backtest_params` method calls `full_backtest_numba()` without passing `quote_conversion_rate`. It defaults to 1.0, which is correct for USD-quoted pairs (GBP/USD) but WRONG for cross-currency pairs (USD/JPY, EUR/JPY, EUR/GBP). This means walk-forward P&L calculations are incorrect for non-USD-quoted pairs.
**Impact:** HIGH for non-USD-quoted pairs; NO impact for GBP/USD (current use case).
**Fix:** Pass `quote_conversion_rate` from config or calculate it via `get_quote_conversion_rate(self.config.pair, 'USD')`.

### C3. Monte Carlo Stage Missing `quote_conversion_rate`
**File:** `pipeline/stages/s5_montecarlo.py` lines 297-314
**Issue:** Same as C2 - `_get_trade_pnls` calls `full_backtest_with_trades()` without `quote_conversion_rate`. Defaults to 1.0.
**Impact:** HIGH for non-USD-quoted pairs; NO impact for GBP/USD.
**Fix:** Same as C2.

### C4. BacktestEngine Hardcoded `pip_size = 0.0001`
**File:** `backtesting/engine.py` line 149
**Issue:** `pip_size` is hardcoded to 0.0001 with a comment "Could adjust for JPY pairs etc." but never does. JPY pairs use 0.01. All P&L, position sizing, and metric calculations in this engine are wrong for JPY pairs.
**Impact:** HIGH for JPY pairs. The BacktestEngine appears to be the older non-numba engine (not used in the pipeline), so practical impact depends on usage.
**Fix:** Detect JPY pairs from the pair name or pass pip_size as a parameter.

### C5. RSI v1 `compute_sl_tp` Fixed TP Mode Bug
**File:** `strategies/rsi_full.py` lines 596-604
**Issue:** When `tp_mode == 'fixed'`, the code falls through to the `else` branch which uses `sl_pips * params['tp_rr_ratio']` instead of `params.get('tp_fixed_pips', 50)`. The `_get_tp_pips` helper method at line 226-237 handles this correctly, but `compute_sl_tp` at line 603 does NOT use the helper and has the wrong fallthrough.
**Impact:** MEDIUM - Fixed TP mode doesn't actually use fixed pips, it uses RR ratio instead. V3 strategy correctly handles this.
**Fix:** Change the `else` branch in `compute_sl_tp` to use `params.get('tp_fixed_pips', 50)`.

### C6. Partial Close PnL Calculation is Redundant
**File:** `optimization/numba_backtest.py` lines 322, 342 (and 843, 862)
**Issue:** `partial_pnl = (pos_entry + partial_target - pos_entry) * partial_close_size * pip_value` simplifies to `partial_target * partial_close_size * pip_value`. The code is correct (not a bug) but the subtraction is confusing and unnecessary. The short side at line 342 has the same pattern: `(pos_entry - (pos_entry - partial_target))` = `partial_target`. Both are mathematically correct, just misleading.
**Impact:** NONE (cosmetic).
**Fix:** Simplify to `partial_target * partial_close_size * pip_value` for clarity.

---

## MODERATE ISSUES (Edge cases / numerical)

### M1. BacktestEngine Sharpe Ratio Annualization Wrong
**File:** `backtesting/engine.py` line 392
**Issue:** `sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()` uses `sqrt(252)` which assumes daily returns. But `returns` is per-trade PnL, not daily returns. For H1 data with ~5 trades/month, this massively inflates the Sharpe. The numba backtests have the same issue (lines 490, 658) but are at least consistently comparable within the system.
**Impact:** MEDIUM - The absolute Sharpe values are not meaningful in industry-standard terms. However, since the same formula is used everywhere consistently, relative comparisons are valid. The threshold `min_back_sharpe=1.0` is calibrated against this formula.
**Fix (optional):** Either (a) note this is a "per-trade Sharpe scaled by sqrt(252)" not a true annualized Sharpe, or (b) convert to daily returns before annualizing. Changing this would require recalibrating thresholds, so this is a low-priority fix.

### M2. RSI Initial Period Off-by-One
**Files:** `strategies/rsi_full.py` line 251, `strategies/rsi_full_v3.py` line 230
**Issue:** `avg_gain[period] = np.mean(gain[1:period+1])` - The initial SMA is computed on indices 1 through period (inclusive), which is `period` values. This is correct for Wilder's RSI since gain[0] is meaningless (diff with prepended value). However, the `delta = np.diff(close, prepend=close[0])` at line 241 means `delta[0] = 0` always, and `gain[0] = 0, loss[0] = 0`. So starting from index 1 is correct. No bug here, but worth documenting the intent.
**Impact:** NONE - calculation is correct.

### M3. ATR First Bar Initialization
**Files:** `strategies/rsi_full.py` line 270, `strategies/rsi_full_v3.py` line 249
**Issue:** `tr[0] = high[0] - low[0]` is correct. But `np.roll(close, 1)` at the bar 0 wraps around to close[-1] (last value). The TR formula `max(H-L, |H-prevC|, |L-prevC|)` uses a garbage value for prevClose at bar 0. However, this is immediately overwritten by `tr[0] = high[0] - low[0]`, so it's fine.
**Impact:** NONE - already handled.

### M4. Equity Curve for R-Squared Uses Trade-Based (Not Time-Based) Indexing
**File:** `optimization/numba_backtest.py` lines 496, 664
**Issue:** `calculate_r_squared(equity_curve)` receives equity values indexed by trade number (not bar/time). A strategy with uneven trade frequency will have its R-squared measured on a trade-count axis, which compresses periods with many trades and stretches periods with few trades. This could give misleadingly high R-squared if many trades cluster in a trending period.
**Impact:** LOW-MEDIUM - R-squared is one factor in OnTester score. The bias favors strategies with consistent trade frequency, which is arguably a good property.
**Fix (optional):** Consider bar-indexed equity curves for R-squared. Would require storing equity per bar rather than per trade, which adds memory cost.

### M5. `basic_backtest_numba` Missing Drawdown Update After Force Close
**File:** `optimization/numba_backtest.py` lines 616-625
**Issue:** When a position is force-closed at the end of data, the code adds the PnL to equity and records it, but does NOT update `peak_equity` or `max_dd`. The `full_backtest_numba` has the same issue at lines 446-456. If the last trade is a large loss, `max_dd` will be understated.
**Impact:** LOW-MEDIUM - Only affects the final trade if it's forced closed, and only if it creates a new drawdown peak.
**Fix:** Add drawdown tracking after the force-close block.

### M6. Daily Reset Uses Day-of-Week Not Calendar Date
**File:** `optimization/numba_backtest.py` lines 244-248, 770-774
**Issue:** Daily limits/loss tracking resets when `days[bar] != current_day`, using day-of-week (0-6). This means limits reset when transitioning from Monday to Tuesday, but also incorrectly if data has gaps that skip a day - e.g., jumping from Friday (4) to Monday (0) correctly resets, but TWO consecutive Mondays (holiday gap) would NOT reset because both are day=0.
**Impact:** LOW - Multi-week gaps are rare in H1 forex data. The daily limit feature isn't heavily used (max_daily_trades defaults to 0).
**Fix:** Use actual date comparison instead of day-of-week comparison. Would require passing timestamps array.

### M7. Walk-Forward Anchor Offset of 6 Months Wastes Data
**File:** `pipeline/stages/s3_walkforward.py` lines 206
**Issue:** `anchor_months = 6` skips the first 6 months of data as "buffer". With 3 years of data, train=6mo, test=6mo, this reduces available window count. The comment says buffer but there's no technical reason - the first window already has a full train period.
**Impact:** LOW - Reduces window count, which might prevent finding enough windows for validation. Not a calculation error, just a design choice that wastes data.
**Fix:** Remove or reduce anchor offset, or make it configurable.

---

## LOW ISSUES (Minor / cosmetic)

### L1. BacktestEngine Position Sizing Division
**File:** `backtesting/engine.py` line 194
**Issue:** `size = risk_amount / (sl_pips * self.pip_value / 10)` - the `/ 10` converts to "mini lots" but this is confusing and undocumented. The numba engine uses a cleaner formula: `pos_size = risk_amt / (sl_dist * pip_value)`.
**Impact:** NONE for pipeline (uses numba engine). Affects standalone BacktestEngine usage.

### L2. Confidence Score Weights Should Sum to 1.0
**File:** `pipeline/config.py` lines 69-74
**Issue:** The weights are 0.15 + 0.15 + 0.25 + 0.15 + 0.15 + 0.15 = 1.00. Verified correct. No issue.

### L3. Monte Carlo Pips Calculation in Trade Details is Approximate
**File:** `pipeline/stages/s5_montecarlo.py` lines 460-463
**Issue:** `trade['pips'] = float(pnls[i]) / 10.0` for non-JPY pairs assumes fixed pip value of $10 per pip per standard lot. This is a rough approximation that doesn't account for actual position size. The pips field is only used for display in the report.
**Impact:** VERY LOW - Display only, not used in any calculations.

### L4. BacktestEngine Does Not Track Drawdown After Force Close
**File:** `backtesting/engine.py` lines 216-222
**Issue:** Force close adds trade and updates equity, but equity_history only gets one append. Drawdown could be understated if force close causes new peak drawdown. Same class of issue as M5.
**Impact:** LOW - Same as M5 but in the standalone engine.

### L5. Swing Detection Sensitivity in V3
**File:** `strategies/rsi_full_v3.py` lines 269-296
**Issue:** `max_breaks = max(1, strength // 4)` allows 1 break for strength=3 (33% tolerance), 1 break for strength=5 (20% tolerance), but 1 break for strength=7 too (14% tolerance). This is inconsistent tolerance. For strength=3, allowing 1 break out of 6 checks (3 left + 3 right) is very permissive.
**Impact:** LOW - Design choice affecting signal generation. More permissive swing detection generates more signals, which is the V3 design intent.

---

## SUMMARY TABLE

| ID | Severity | File | Description | Impact for GBP/USD |
|----|----------|------|-------------|-------------------|
| C1 | CRITICAL | numba_backtest.py | Population vs sample std dev in Sharpe | ~5% Sharpe inflation |
| C2 | CRITICAL | s3_walkforward.py | Missing quote_conversion_rate | NONE (USD-quoted) |
| C3 | CRITICAL | s5_montecarlo.py | Missing quote_conversion_rate | NONE (USD-quoted) |
| C4 | CRITICAL | engine.py | Hardcoded pip_size | NONE (not in pipeline) |
| C5 | CRITICAL | rsi_full.py (v1) | Fixed TP mode uses RR instead | Depends on params |
| C6 | COSMETIC | numba_backtest.py | Redundant partial PnL arithmetic | NONE |
| M1 | MODERATE | engine.py + numba | Sharpe annualization assumes daily | Inflated but consistent |
| M2 | NONE | rsi_full*.py | RSI initial period (actually correct) | NONE |
| M3 | NONE | rsi_full*.py | ATR bar 0 (already handled) | NONE |
| M4 | LOW-MED | numba_backtest.py | R-squared trade-indexed not time-indexed | Slight bias |
| M5 | LOW-MED | numba_backtest.py | No DD update after force close | Understated DD |
| M6 | LOW | numba_backtest.py | Daily reset by DOW not date | Rare edge case |
| M7 | LOW | s3_walkforward.py | 6-month anchor wastes data | Fewer windows |
| L1 | LOW | engine.py | Position sizing /10 undocumented | NONE for pipeline |
| L3 | VERY LOW | s5_montecarlo.py | Approx pips in trade details | Display only |
| L5 | LOW | rsi_full_v3.py | Swing tolerance inconsistent | Design choice |

---

## PROPOSED FIXES (Priority Order)

### Fix 1: Sharpe Ratio Sample Std Dev (C1)
In `optimization/numba_backtest.py`, three locations:
- Line 488: `std = np.sqrt(var / n_trades)` -> `std = np.sqrt(var / (n_trades - 1))`
- Line 656: same fix
- Line 1003: same fix

### Fix 2: Walk-Forward quote_conversion_rate (C2)
In `pipeline/stages/s3_walkforward.py`:
- Import `get_quote_conversion_rate` from `optimization.numba_backtest`
- In `_backtest_params`, after calling `full_backtest_numba`, pass `quote_conversion_rate=get_quote_conversion_rate(self.config.pair, 'USD')`

### Fix 3: Monte Carlo quote_conversion_rate (C3)
In `pipeline/stages/s5_montecarlo.py`:
- Same pattern as Fix 2 for `_get_trade_pnls`

### Fix 4: RSI v1 Fixed TP Mode (C5)
In `strategies/rsi_full.py` line 603:
- Change `else:` branch to `tp_pips = params.get('tp_fixed_pips', 50)`

### Fix 5: Force-Close Drawdown Tracking (M5)
In `optimization/numba_backtest.py`, add DD tracking after force close blocks (3 locations):
```python
if equity > peak_equity:
    peak_equity = equity
dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
if dd > max_dd:
    max_dd = dd
```

### Fix 6 (Optional): Walk-Forward Anchor Reduction (M7)
In `pipeline/stages/s3_walkforward.py` line 206:
- Change `anchor_months = 6` to `anchor_months = 0` or make configurable

---

## NOT PROPOSED FOR FIX (Acceptable as-is)

1. **Sharpe annualization (M1)**: Changing would require recalibrating all thresholds. The current approach is internally consistent.
2. **R-squared trade indexing (M4)**: Trade-based R-squared is a defensible choice and matches MT5 OnTester behavior.
3. **Daily reset by DOW (M6)**: Edge case too rare to justify adding timestamp arrays to numba functions.
4. **BacktestEngine issues (C4, L1, L4)**: The standalone engine isn't used in the pipeline. Fixing it is low priority unless it will be used for live trading.
