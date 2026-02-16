# Pipeline Quality Assessment

**Date:** 2026-02-07 (updated 2026-02-16)
**Overall Grade: B (64/100)**

This is a summary extracted from the full quality audit. The system architecture is solid but has statistical methodology gaps.

---

## Component Scores

| Component | Score | Weight | Notes |
|-----------|-------|--------|-------|
| Optimization | 7/10 | 15% | Optuna TPE, staged, Quality Score objective - proven approach |
| Data Handling | 7/10 | 10% | M1->H1 aggregation, quality validation, smart caching |
| Overfitting Protection | 6.5/10 | 15% | Forward ratio filter, perturbation stability, WF validation |
| Walk-Forward | 6/10 | 15% | Rolling windows, but WF data overlaps optimization training |
| Scoring System | 7/10 | 10% | Quality Score (Sortino, R², PF, trades, return, Ulcer+DD), per-bar Ulcer |
| Industry Comparison | 6.5/10 | 5% | Slippage model added (SL exit slippage), bars_per_year from timeframe. Still missing permutation testing, deflated Sharpe |
| Monte Carlo | 4/10 | 15% | Shuffle-only returns are degenerate (see KI-1) |
| Statistical Rigor | 5/10 | 15% | Sharpe fixed (Bessel's correction), Sortino replaces Sharpe in scoring. Still no multiple comparisons, low trade counts |

---

## HIGH Priority Fixes (Affects Result Validity)

1. **Fix MC return distribution** - Use bootstrap resampling (KI-1)
2. ~~**Fix intrabar event ordering**~~ - FIXED: SL/TP checked before management (KI-3)
3. ~~**Fix Sharpe sample std dev**~~ - FIXED: Bessel's correction applied (KI-5)
4. ~~**Fix PF key mismatch in scorer**~~ - FIXED: correct key name (KI-4)
5. **Add permutation significance test** - Verify edge beats random entry

## MEDIUM Priority Fixes

6. Add confidence intervals via bootstrap
7. Increase perturbation test depth (4-6 levels, not just 2)
8. ~~Add slippage to numba backtest engine~~ - FIXED: `slippage_pips` param on all 7 numba functions, SL exits adjusted unfavorably
9. Fix WF data overlap with optimization training data
10. Raise min_forward_ratio from 0.15 to 0.40-0.50

## LOW Priority

11. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
12. True holdout validation set
13. Historical exchange rates for cross-currency conversion
14. ~~Use Optuna native int/float suggestion~~ - Low impact, deferred

---

## Other Fixes Since Original Assessment

- **Scoring system overhaul**: OnTester score replaced by Quality Score (Sortino, R², PF, trades, return, Ulcer+DD). Per-bar Ulcer Index captures time underwater. Return% capped at 200% to prevent compound inflation.
- **bars_per_year computed from timeframe** (was hardcoded H1). KI-11 FIXED.
- **Dead code removed**: `backtesting/`, `visualization/`, 7 dead scripts deleted. ML exit code archived.
- **requirements.txt fixed**: Accurate dependency list.
- **Dashboard security**: HTTP Basic auth added to POST endpoints (service start/stop/restart).
- **Confidence scoring fixes**: WF Sharpe consistency blending, Sortino-based BT quality sub-components, blended DD (Ulcer+MaxDD/2).

## Key Insight

The remaining HIGH priority items are MC bootstrap resampling (KI-1) and permutation significance testing. These are the main gaps between current grade (B) and a proper A-tier pipeline.

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the full issue tracker with status.
