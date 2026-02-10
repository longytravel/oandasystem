# Known Issues Tracker

**Last Updated:** 2026-02-08
**Sources:** Consolidated from REVIEW_FEEDBACK.md, PLAN_CALC_AUDIT.md, PLAN-dashboard-fixes.md

---

## Status Key
- OPEN - Not yet fixed
- FIXED - Fix applied and verified
- WONTFIX - Accepted limitation, documented
- DEFERRED - Will fix as part of ML Exit Program

---

## CRITICAL Issues

### KI-1: Monte Carlo return distribution is degenerate
**Source:** REVIEW_FEEDBACK Finding 1 | **Status:** OPEN -> DEFERRED (ML Sprint 1)
**File:** `pipeline/stages/s5_montecarlo.py`
**Problem:** Shuffle-only MC preserves sum(pnls), so all iterations produce identical final returns. `pct_5_return`, `std_return`, `prob_positive` etc. are all meaningless (N copies of same value). DD distribution IS valid (different paths = different drawdowns).
**Impact:** MC return percentiles fed to confidence scoring provide zero information.
**Fix:** Use bootstrap resampling for return distribution. Keep shuffle for DD. See ML Exit Program Sprint 1.

### KI-2: Partial closes inflate trade count
**Source:** REVIEW_FEEDBACK Finding 2 + 9 | **Status:** FIXED
**File:** `optimization/numba_backtest.py`
**Problem:** Partial close records a separate PnL entry and increments n_trades. One signal with partial close = 2 trade records. Inflates `sqrt(trades)` in OnTester score, distorts Sharpe.
**Fix applied:** "Fix Finding 2" comments throughout numba_backtest.py (lines 252, 399, 447, 523, 870, 1012, 1059, 1127). Partial close PnL is now accumulated into parent trade and recorded as a single combined trade.

### KI-3: Intrabar management has look-ahead bias
**Source:** REVIEW_FEEDBACK Finding 3 | **Status:** FIXED
**File:** `optimization/numba_backtest.py`
**Problem:** Within each bar: trailing/BE/partial use bar_high/bar_low BEFORE SL/TP check. For a long trade, if bar_high triggers trailing tightening, then bar_low hits the new tighter SL, code assumes high came first. Could be optimistic.
**Fix applied:** "Fix Finding 3" comments at lines 311, 925. SL/TP exits are now checked BEFORE management adjustments. Management changes apply to the NEXT bar only.

---

## HIGH Issues

### KI-4: Confidence scorer PF key mismatch
**Source:** REVIEW_FEEDBACK Finding 4 | **Status:** FIXED
**File:** `pipeline/stages/s6_confidence.py` line 165
**Problem:** Candidates store `back_profit_factor` but scorer reads `back_pf`. Falls back to 1.0.
**Fix applied:** "Fix Finding 4" comment at line 165. Key now correctly reads `back_profit_factor`.

### KI-5: Sharpe uses population std dev (not sample)
**Source:** PLAN_CALC_AUDIT C1 | **Status:** FIXED
**File:** `optimization/numba_backtest.py` lines 569, 747, 1171
**Problem:** `var / n_trades` instead of `var / (n_trades - 1)`. With 20-50 trades, ~5% Sharpe inflation.
**Fix applied:** All three backtest functions now use `var / (n_trades - 1)` (sample std dev) with guard `if n_trades > 1`.

### KI-6: WF + MC missing quote_conversion_rate
**Source:** PLAN_CALC_AUDIT C2, C3 | **Status:** OPEN
**Files:** `pipeline/stages/s3_walkforward.py` line 339, `pipeline/stages/s5_montecarlo.py` line 297
**Problem:** Defaults to 1.0. Correct for GBP_USD but wrong for JPY pairs.
**Impact:** ZERO for current GBP_USD use case.
**Fix:** Pass `quote_conversion_rate` from pair config.

### KI-7: Pipeline continues after data validation failure
**Source:** REVIEW_FEEDBACK Finding 7 | **Status:** OPEN
**File:** `pipeline/pipeline.py` line 256
**Problem:** `_should_abort` only checks optimization stage, not data `validation_passed`.
**Fix:** Add data validation check to `_should_abort()`.

### KI-8: Missing save_stage_output on early returns
**Source:** REVIEW_FEEDBACK Finding 12 | **Status:** OPEN
**Files:** `pipeline/stages/s4_stability.py`, `s5_montecarlo.py`, `s6_confidence.py`
**Problem:** Early returns (empty candidates) don't call `save_stage_output()`. Breaks resume persistence.
**Fix:** Add save calls to early return paths.

---

## MEDIUM Issues

### KI-9: Stability chart empty - key name mismatch
**Source:** PLAN-dashboard-fixes Issue 1 | **Status:** FIXED
**Files:** `pipeline/report/chart_generators.py` line 657
**Problem:** Stability data stored under `params` key, report reads `per_param`. Chart always empty.
**Fix applied:** `chart_generators.py` line 657 uses `stability.get('per_param', stability.get('params', {}))` fallback.

### KI-10: Empty stage summaries after pipeline resume
**Source:** PLAN-dashboard-fixes Issue 2 | **Status:** FIXED
**File:** `pipeline/report/data_collector.py` lines 12-19
**Problem:** Resumed runs have empty `{}` for skipped stage results. Report shows 0 for summaries.
**Fix applied:** `_get_stage_summary()` helper falls back to `state.stages[stage_name].summary` when result dict is empty. All stage summaries use this helper (lines 84-90).

### KI-11: Sharpe annualization hardcoded for H1
**Source:** REVIEW_FEEDBACK Finding 11, PLAN_CALC_AUDIT M1 | **Status:** FIXED
**File:** `optimization/numba_backtest.py` lines 198-199, 571-572, 601, 749-750, 818-819, 1173-1174
**Problem:** `bars_per_year = 5544.0` (H1 assumption). Wrong for M1, H4, D1.
**Fix applied:** "Fix Finding 11" comments throughout. `bars_per_year` is now a configurable function parameter (default 5544.0 for H1) in all three backtest functions. Callers can pass the correct value for any timeframe.

### KI-12: RSI v1 fixed TP mode bug
**Source:** PLAN_CALC_AUDIT C5 | **Status:** OPEN (low priority)
**File:** `strategies/rsi_full.py` lines 596-604
**Problem:** `tp_mode='fixed'` falls through to RR calculation instead of using `tp_fixed_pips`.
**Impact:** V1 only. V3+ handles this correctly.

### KI-13: Force-close missing drawdown update
**Source:** PLAN_CALC_AUDIT M5 | **Status:** FIXED
**File:** `optimization/numba_backtest.py` lines 529-534, 711-716, 1133-1138
**Problem:** Force-close at end of data doesn't update peak_equity/max_dd. Final DD could be understated.
**Fix applied:** All three backtest functions now track drawdown after force close with peak_equity/max_dd update.

---

## LOW Issues

### KI-14: Legacy engine.py `df` reference bug
**Source:** REVIEW_FEEDBACK Finding 10 | **Status:** OPEN (legacy, unused)
**File:** `backtesting/engine.py` line 394
**Problem:** `df` not in scope in `_calculate_results()`. NameError at runtime.
**Impact:** Legacy engine not used by pipeline. Low priority.

### KI-15: BacktestEngine hardcoded pip_size
**Source:** PLAN_CALC_AUDIT C4 | **Status:** WONTFIX (legacy)
**File:** `backtesting/engine.py` line 149
**Problem:** `pip_size = 0.0001` hardcoded. Wrong for JPY pairs.

### KI-16: Daily reset uses day-of-week not calendar date
**Source:** PLAN_CALC_AUDIT M6 | **Status:** WONTFIX
**File:** `optimization/numba_backtest.py`
**Problem:** Multi-week gaps (rare) wouldn't reset daily limits correctly.

### KI-17: Partial close PnL arithmetic is redundant
**Source:** PLAN_CALC_AUDIT C6 | **Status:** OPEN (cosmetic)
**File:** `optimization/numba_backtest.py` lines 322, 342
**Problem:** `(pos_entry + partial_target - pos_entry)` simplifies to `partial_target`. Not a bug.

---

### KI-18: ML exit trade count inflation
**Source:** ML Exit V2 analysis | **Status:** FIXED
**File:** `optimization/numba_backtest.py`
**Problem:** When ML closes a trade early, position is freed, new signals fire on bars that were previously blocked. Pass1=90 trades becomes Pass3=123. Extra trades are unoptimized garbage entries that drag performance down.
**Fix applied:** Added `ml_exit_cooldown_bars` parameter (default 10). After ML exit, new entries are blocked for N bars. Wired through config.py → run_pipeline.py → s3_walkforward.py.

### KI-19: ML exit wrong labeling (V1)
**Source:** ML Exit V2 research | **Status:** FIXED
**File:** `pipeline/ml_exit/train.py`, `pipeline/ml_exit/labeling.py`
**Problem:** V1 used `future_r_change_5bar` (5-bar lookahead) as regression target. Too noisy — R² consistently near 0. Model couldn't learn per-bar exit decisions.
**Fix applied:** V2 trains on `remaining_pnl_r = final_R - current_R` (optimal stopping formulation). Answers "will this trade get better or worse from here?" Falls back to V1 target for old datasets.

---

## DESIGN DECISIONS (Not Bugs)

### DD-1: Walk-forward uses fixed params (not re-optimization)
**Source:** REVIEW_FEEDBACK Finding 6
**Decision:** By design. Tests parameter stability across time. Re-optimization WF is 10-100x slower. Config flag `reoptimize_per_window` exists for future use.

### DD-2: Overfit filter auto-disables on empty results
**Source:** REVIEW_FEEDBACK Finding 13
**Decision:** Intentional. Better to have candidates with poor forward performance than none (downstream stages will produce RED score).

### DD-3: Single-position architecture
**Source:** REVIEW_FEEDBACK Finding 8
**Decision:** System is designed for single-position strategies. Multi-position/portfolio is out of scope.

### DD-4: Static cross-currency conversion rates
**Source:** REVIEW_FEEDBACK Finding 5
**Decision:** Known approximation. Correct for GBP_USD (1.0). ~5-15% error for cross pairs. Historical rate lookup is a future enhancement.

### DD-5: R-squared uses trade-indexed equity (not time-indexed)
**Source:** PLAN_CALC_AUDIT M4
**Decision:** Matches MT5 OnTester behavior. Slightly favors consistent trade frequency (arguably desirable).
