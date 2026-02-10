# External Code Review Feedback - Assessment Document

## Purpose
This document tracks the external team's feedback, our assessment of each finding, and any actions taken.

---

## Finding 1: CRITICAL - Monte Carlo "return distribution" is mathematically degenerate
**Claim**: `run_monte_carlo_numba` only shuffles trade order; `simulate_equity_curve` computes final return from total sum, so every iteration has the same return. `pct_5_return` is not a real risk estimate. Claims `unique_returns = 1`.

**Assessment**: **ACCEPT** - The reviewer is mathematically correct about the return distribution.

**Verified against code** (assessor review 2026-02-06):
- `simulate_equity_curve` (s5_montecarlo.py:30-53) computes `final_return = (equity - initial_capital) / initial_capital * 100` where equity is built via `equity += pnls[i]` in a loop.
- The final value of `equity` always equals `initial_capital + sum(pnls)`. Shuffling changes the ORDER of additions but not the SUM. Therefore `final_return = sum(pnls) / initial_capital * 100` is identical for every shuffle iteration. The `returns` array from `run_monte_carlo_numba` (line 72-102) contains N copies of the same value.
- Consequently: `pct_5_return`, `std_return`, `prob_positive`, `prob_above_5pct`, `var_95`, `expected_shortfall` are all degenerate (computed from identical values).
- HOWEVER: `max_dd` DOES vary with shuffle order (different equity paths produce different drawdown sequences), so the DD distribution IS valid and useful.
- The bootstrap resampling (line 106-143) samples WITH REPLACEMENT, producing genuinely different trade sets. The permutation test (line 176-216) flips trade signs randomly. Both produce meaningful variation.

**Impact**: The MC return percentiles (`pct_5_return`) fed to confidence scoring (s6_confidence.py:227) provide no information. The DD percentiles, bootstrap CIs, and permutation test are all fine.

**Action Required**: YES - Fix MC to produce meaningful return variation. Options:
1. Use bootstrap resampling (sample with replacement) instead of shuffling for return distribution
2. Remove the degenerate return percentile from confidence scoring, rely on bootstrap CIs and permutation test instead

---

## Finding 2: CRITICAL - Partial closes counted as separate trades
**Claim**: Partial close books a trade (line 324) and final exit books another (line 382), inflating trade count and metrics via `sqrt(trades)`.

**Assessment**: **ACCEPT** - This is a valid concern.

**Verified against code** (assessor review 2026-02-06):
- `full_backtest_numba` (numba_backtest.py:324): Partial close does `pnls[n_trades] = partial_pnl; n_trades += 1`
- `full_backtest_with_trades` (numba_backtest.py:864-869): Same pattern -- partial close records separate PnL entry and increments n_trades.
- Final exit (numba_backtest.py:382): Also `pnls[n_trades] = pnl; n_trades += 1`
- OnTester formula (line 140): `profit * r_squared * pf * sqrt(trades)` -- inflated trade count boosts score
- Sharpe calculation uses `n_trades` which includes partial closes
- One signal with partial close produces 2 trade records, inflating trade count

**Severity assessment**: In practice, partial close is OFF by default in V3 strategy and was OFF in the winning GBP_USD configs. The impact is real but only when `use_partial_close=True`.

**Action Required**: YES - Partial close PnL should be accumulated into the parent trade's final PnL rather than being a separate trade record. This requires restructuring how partial close is tracked.

---

## Finding 3: CRITICAL - Intrabar management has look-ahead bias
**Claim**: For an open long, code updates trailing/BE/partial from bar high before SL/TP checks, which assumes favorable order-of-events inside a candle.

**Assessment**: **ACCEPT WITH CAVEAT** - Valid concern but inherent to OHLC backtesting.

**Verified against code** (assessor review 2026-02-06):
- `full_backtest_numba` execution order within `if in_pos:` block:
  1. Time-based exit check (line 262)
  2. Trailing stop update using `bar_high`/`bar_low` (line 270-293)
  3. Breakeven update using `bar_high`/`bar_low` (line 296-311)
  4. Partial close using `bar_high`/`bar_low` (line 314-357)
  5. SL/TP exit check against (potentially modified) `pos_sl`/`pos_tp` (line 360-373)
- The same pattern exists in `full_backtest_with_trades` (lines 814-914).
- For a long trade: if `bar_high` triggers trailing SL tightening at step 2, and `bar_low` hits the newly tightened SL at step 5, the code assumes the high came first. In reality, low could have come first (hitting the ORIGINAL, wider SL).
- This creates an optimistic bias: management adjustments that protect profits are applied before exit checks.

**Caveat**: This is a fundamental limitation of OHLC backtesting without tick data. ALL OHLC backtesters face this. The standard mitigation is:
1. Use conservative execution assumptions (check SL/TP BEFORE adjustments)
2. Or simply document the limitation

**Action Required**: YES - Reverse the order: check SL/TP exits FIRST using the CURRENT (pre-adjustment) SL/TP, then apply management adjustments for the NEXT bar. This is the conservative approach.

---

## Finding 4: HIGH - Confidence score ignores candidate profit factor due wrong key
**Claim**: Candidates store `back_profit_factor` (s2_optimization.py:366) but scorer reads `back_pf` (s6_confidence.py:162).

**Assessment**: **ACCEPT** - Key mismatch causes PF to silently default to 1.0.

**Verified against code** (assessor review 2026-02-06):
- `s2_optimization.py:366` stores `'back_profit_factor': back.profit_factor` on each candidate dict.
- `s6_confidence.py:162` reads: `profit_factor = back_stats.get('profit_factor', candidate.get('back_pf', 1.0))`
- Tracing the fallback chain:
  1. `back_stats = candidate.get('back_stats', {})` -- `back_stats` is never populated by the pipeline, so this is `{}`.
  2. `{}.get('profit_factor', ...)` returns the default.
  3. Default is `candidate.get('back_pf', 1.0)` -- but the key is `back_profit_factor`, not `back_pf`.
  4. `candidate.get('back_pf', 1.0)` returns 1.0.
- The same pattern affects `back_sharpe` (line 166), `back_trades` (line 170), and `back_max_dd` (line 174). Checking these:
  - `back_sharpe`: `back_stats.get('sharpe', candidate.get('back_sharpe', 0))` -- `back_sharpe` IS the correct key stored by s2 (line 364). This one WORKS.
  - `back_trades`: `back_stats.get('trades', candidate.get('back_trades', 0))` -- `back_trades` IS correct (line 361). This WORKS.
  - `back_max_dd`: `back_stats.get('max_dd', candidate.get('back_max_dd', 40))` -- `back_max_dd` IS correct (line 367). This WORKS.
- **Only `profit_factor` has the key mismatch** (`back_pf` vs `back_profit_factor`).
- Impact: PF sub-score = `(1.0 - 1.0) * 50 = 0`. Since PF is 1/4 of the backtest quality component (which is 15% of total), the impact is: `0 / 4 * 0.15 = 0` lost points, max possible loss ~3.75 points (if PF were 3.0+).

**Action Required**: YES - Fix the fallback key from `back_pf` to `back_profit_factor` on line 162.

---

## Finding 5: HIGH - Cross-currency valuation is static and time-inaccurate
**Claim**: `get_quote_conversion_rate` uses hardcoded constants and returns 1.0 for non-USD accounts.

**Assessment**: **ACCEPT - KNOWN LIMITATION, LOW PRIORITY**

**Verified against code** (assessor review 2026-02-06):
- numba_backtest.py:1082 `conversion_rates` dict: JPY=0.0067, GBP=1.27, EUR=1.08, AUD=0.65, NZD=0.60, CAD=0.74, CHF=1.12
- numba_backtest.py:1093: returns 1.0 for non-USD accounts
- These are documented approximations, not bugs

**Caveat**: For the primary use case (GBP_USD with USD account), conversion rate is 1.0 (correct). For cross pairs, this introduces ~5-15% error in position sizing/PnL. Using historical rates would require downloading additional data and passing time-varying rates into the numba engine.

**Action Required**: NO immediate action - document as known limitation. Historical rate lookup is a future enhancement.

---

## Finding 6: HIGH - Walk-forward is not true re-optimization walk-forward
**Claim**: Windows evaluate fixed candidate params only on test slices. `reoptimize_per_window` exists in config but is unused.

**Assessment**: **ACCEPT - BY DESIGN, NOT A BUG**

**Verified against code** (assessor review 2026-02-06):
- s3_walkforward.py:290 `_test_candidate_windows_fast` evaluates fixed params across windows. No re-optimization occurs.
- config.py:47 `reoptimize_per_window: bool = False` exists but is never read by the WF stage code.
- The current approach IS a valid walk-forward validation method: testing parameter stability across time windows
- True re-optimization WF (re-optimize per window) is more rigorous but 10-100x slower

**Action Required**: NO immediate action - this is a design choice, not a bug. The config flag exists for future implementation. The current fixed-param WF is standard practice and works well for our use case.

---

## Finding 7: HIGH - Pipeline continues after failing core validation gates
**Claim**: Data stage can fail requirements but pipeline doesn't abort. Walk-forward can return insufficient evidence yet overall run still reports complete.

**Assessment**: **PARTIALLY ACCEPT**

**Verified against code** (assessor review 2026-02-06):
- s1_data.py:287 `_check_requirements` returns bool, stored in result dict as `validation_passed` (line 121).
- pipeline.py:256 `_should_abort` only checks `optimization` stage for zero candidates. It does NOT check `data.validation_passed`.
- pipeline.py:131 calls `_should_abort()` AFTER each stage, so if data validation fails but returns data, the pipeline proceeds to optimization.
- s3_walkforward.py:84-101 handles insufficient windows correctly: returns `{'candidates': []}`, which propagates empty candidate lists downstream.
- Downstream stages (s4, s5, s6) all handle empty candidates gracefully with early returns.
- The pipeline IS designed to generate a report even with failures (so the user sees what happened via RED score).

**Action Required**: PARTIAL - Add data validation failure check to `_should_abort` to abort early when data is fundamentally inadequate. Walk-forward already handles its own failure correctly.

---

## Finding 8: HIGH - Strategy-agnostic claim is limited by architecture
**Claim**: Strategy loading is hardcoded, staged optimization requires parameter groups, backtest supports only one open position.

**Assessment**: **ACCEPT - KNOWN LIMITATION, BY DESIGN**

**Evidence**: All claims are factually correct. The system is designed for single-position RSI divergence strategies. Multi-position/portfolio support was never a design goal.

**Action Required**: NO - This is a scope limitation, not a bug. The system doesn't claim to support arbitrary strategies.

---

## Finding 9: HIGH - Trade-detail reporting is structurally inaccurate for management features
**Claim**: Report trade reconstruction maps trade index to signal index, which breaks with partial closes.

**Assessment**: **ACCEPT** - Directly related to Finding 2.

**Evidence**:
- s5_montecarlo.py:620: `if i < len(entry_bars)` - maps trade[i] to signal[i]
- With partial closes, trade count > signal count, so later trades get no metadata
- Pips calculation (line 631-632) uses a heuristic, not actual executed pip difference

**Action Required**: YES - Will be fixed as part of Finding 2 (partial close restructuring). Once partial close PnL is merged into parent trade, the 1:1 mapping holds again.

---

## Finding 10: HIGH - Legacy backtesting engine has runtime bug
**Claim**: engine.py line 394 references `df` not in scope, causing NameError.

**Assessment**: **ACCEPT**

**Verified against code** (assessor review 2026-02-06):
- engine.py:394: `duration_days = (df.index[-1] - df.index[0]).days if len(df) > 1 else 365`
- Method signature at line 344-348: `def _calculate_results(self, trades, equity_history, params)` -- no `df` parameter.
- `df` is not in scope. This would cause a `NameError` at runtime when `len(trades) > 1`.
- The Sharpe calculation at line 396 also depends on this broken `duration_days`.

**Action Required**: YES - Fix the variable reference (e.g., estimate duration from trade timestamps or equity_history). However, this is the LEGACY engine (not used by the pipeline). The pipeline uses `numba_backtest.py`. Low priority but should still be fixed.

---

## Finding 11: MEDIUM - Sharpe annualization is timeframe-hardcoded
**Claim**: `bars_per_year = 5544.0` regardless of timeframe.

**Assessment**: **ACCEPT**

**Verified against code** (assessor review 2026-02-06):
- `full_backtest_numba` line 500: `bars_per_year = 5544.0`
- `basic_backtest_numba` line 677: `bars_per_year = 5544.0`
- `full_backtest_with_trades` line 1033: `bars_per_year = 5544.0`
- 5544 = 252 trading days * ~22 trading hours = H1 assumption
- For M1: should be ~332,640; for H4: ~1,386; for D1: ~252

**Action Required**: YES - Accept timeframe as a parameter and calculate bars_per_year dynamically. Currently the pipeline only uses H1 so impact is zero, but this should be fixed for correctness.

---

## Finding 12: MEDIUM - Stage artifact persistence is inconsistent on early returns
**Claim**: Early returns in s4_stability.py, s5_montecarlo.py, s6_confidence.py skip `save_stage_output`.

**Assessment**: **ACCEPT**

**Verified against code** (assessor review 2026-02-06):
- s4_stability.py:57-62: `if not candidates:` returns dict without calling `state.save_stage_output()`. Normal path calls it at line 157.
- s5_montecarlo.py:254-259: `if not candidates:` returns dict without calling `state.save_stage_output()`. Normal path calls it at line 362.
- s6_confidence.py:57-63: `if not candidates:` returns dict without calling `state.save_stage_output()`. Normal path calls it at line 131.
- These early returns produce valid result dicts but don't persist to JSON, which means resume from later stages may not find the output artifacts.

**Action Required**: YES - Add save_stage_output calls to early return paths.

---

## Finding 13: MEDIUM - Overfit filter can auto-disable
**Claim**: `unified_optimizer.py:601` disables forward-ratio filter if all results are rejected.

**Assessment**: **ACCEPT - BY DESIGN**

**Verified against code** (assessor review 2026-02-06):
- unified_optimizer.py:596-601: After filtering, `if filtered:` uses filtered results; `else:` logs warning `"Forward threshold filter rejected ALL results - disabling filter"` and keeps the unfiltered `valid` list.
- This is intentional: better to have candidates with poor forward performance than no candidates at all (the pipeline's later stages -- WF, stability, MC, confidence -- will catch overfitting and produce a RED score).

**Action Required**: NO - This is intentional behavior with proper logging. The walk-forward and stability stages serve as secondary overfit detection.

---

## Finding 14: MEDIUM - Major bottleneck is optimization stage
**Claim**: Optimization took ~9902s while other stages were seconds to tens of seconds.

**Assessment**: **ACCEPT - EXPECTED**

**Evidence**: Optimization runs 5000-10000+ trials with full backtests. This is inherently the expensive stage.

**Action Required**: NO - This is expected behavior. Optimization is computationally intensive by nature.

---

## Summary of Actions

*All findings independently verified against source code on 2026-02-06.*

### MUST FIX (Code changes needed):
1. **Finding 1** (CRITICAL): MC return distribution is degenerate -- shuffling preserves sum, so all 1000 iterations produce identical returns. Use bootstrap for return CIs or remove degenerate percentile from scoring.
2. **Finding 2 + 9** (CRITICAL): Partial close inflates trade count (1 signal = 2 trade records). Merge partial PnL into parent trade to restore 1:1 signal-to-trade mapping.
3. **Finding 3** (CRITICAL): Intrabar look-ahead -- management adjustments applied before SL/TP checks on same bar. Reverse order for conservative execution.
4. **Finding 4** (HIGH): PF key mismatch in confidence scorer -- `back_pf` should be `back_profit_factor` (s6_confidence.py:162). Impact: ~3.75 points max.
5. **Finding 10** (HIGH): Legacy engine.py `df` reference bug -- NameError in `_calculate_results()`. Low priority (legacy, unused by pipeline).
6. **Finding 11** (MEDIUM): Hardcoded `bars_per_year = 5544.0` in Sharpe calculation. Zero impact on H1 (current use), but incorrect for other timeframes.
7. **Finding 12** (MEDIUM): Missing `save_stage_output` on early returns in s4, s5, s6. Breaks resume persistence for edge cases.

### ACCEPT BUT NO ACTION (Design choices / known limitations):
- **Finding 5**: Static cross-currency rates (known, documented, correct for primary GBP_USD use case)
- **Finding 6**: Fixed-param WF (by design, `reoptimize_per_window` config exists for future work)
- **Finding 8**: Single-position architecture (by design, not claimed as multi-strategy)
- **Finding 13**: Overfit filter auto-disable (intentional with warning, downstream stages catch overfitting)
- **Finding 14**: Optimization is slow (expected -- 5000-10000 trials with full backtests)

### PARTIALLY ACCEPT:
- **Finding 7** (HIGH): Add data `validation_passed` check to `pipeline.py:_should_abort()`. Walk-forward already handles its own failure correctly.
