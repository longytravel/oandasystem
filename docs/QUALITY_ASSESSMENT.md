# Pipeline Quality Assessment

**Date:** 2026-02-07
**Overall Grade: B- (56/100)**

This is a summary extracted from the full quality audit. The system architecture is solid but has statistical methodology gaps.

---

## Component Scores

| Component | Score | Weight | Notes |
|-----------|-------|--------|-------|
| Optimization | 7/10 | 15% | Optuna TPE, staged, OnTester score - proven approach |
| Data Handling | 7/10 | 10% | M1->H1 aggregation, quality validation, smart caching |
| Overfitting Protection | 6.5/10 | 15% | Forward ratio filter, perturbation stability, WF validation |
| Walk-Forward | 6/10 | 15% | Rolling windows, but WF data overlaps optimization training |
| Scoring System | 5.5/10 | 10% | Multi-factor, well-weighted, but inputs are sometimes flawed |
| Industry Comparison | 5/10 | 5% | Missing slippage, permutation testing, deflated Sharpe |
| Monte Carlo | 4/10 | 15% | Shuffle-only returns are degenerate (see KI-1) |
| Statistical Rigor | 4/10 | 15% | Sharpe bugs, no multiple comparisons, low trade counts |

---

## HIGH Priority Fixes (Affects Result Validity)

1. **Fix MC return distribution** - Use bootstrap resampling (KI-1)
2. **Fix intrabar event ordering** - Conservative SL/TP check first (KI-3)
3. **Fix Sharpe sample std dev** - Bessel's correction (KI-5)
4. **Fix PF key mismatch in scorer** - `back_pf` -> `back_profit_factor` (KI-4)
5. **Add permutation significance test** - Verify edge beats random entry

## MEDIUM Priority Fixes

6. Add confidence intervals via bootstrap
7. Increase perturbation test depth (4-6 levels, not just 2)
8. Add slippage to numba backtest engine
9. Fix WF data overlap with optimization training data
10. Raise min_forward_ratio from 0.15 to 0.40-0.50

## LOW Priority

11. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
12. True holdout validation set
13. Historical exchange rates for cross-currency conversion
14. Use Optuna native int/float suggestion (not categorical for numerics)

---

## Key Insight

The Sharpe ratio bug alone (KI-5 + annualization) could turn a GREEN into a YELLOW. Adding proper significance testing would reveal whether strategy edges are real or artifacts of overfitting.

**The first 4 HIGH priority fixes are all preconditions for the ML Exit Program (Sprint 1).**

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the full issue tracker with status.
