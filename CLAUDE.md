# OANDA Trading System

## Project Overview

Python-based algorithmic trading system using OANDA API with a 7-stage optimization pipeline (Data -> Optimization -> Walk-Forward -> Stability -> Monte Carlo -> Confidence -> Report).

Key directories:
- `strategies/` - Strategy implementations (V3-V5 RSI divergence, V6 EMA cross, Fair Price MA, RSI Fast). Archived: `strategies/archive/` (V1, V2, trend_simple)
- `pipeline/` - Pipeline stages (`stages/s1-s7_*.py`), config, state, report generation. Archived: `pipeline/archive/ml_exit/` (ML exit code, concluded as dead end)
- `optimization/` - Optuna optimizer (`unified_optimizer.py`), numba backtesting, ML features
- `live/` - Live/paper trading (trader.py, oanda_client.py, position_manager.py, pipeline_adapter.py)
- `data/` - Data download and caching (M1 parquet -> any timeframe)
- `results/pipelines/` - Pipeline run outputs (each run gets a unique timestamped directory)
- `config/` - Settings, exported parameter files
- `docs/` - Documentation (README, SYSTEM, STRATEGY_EVOLUTION, KNOWN_ISSUES). Archived: `docs/archive/`

Best strategy: V3 RSI Divergence (89.8/100 GREEN on H1, 87.2/100 GREEN on M15). Note: these scores are pre-exit-slippage-model (added Feb 16). M15 results are invalidated by slippage; H1 results less affected but need re-validation.

## Critical Rules

### Investigate suspicious results before reporting
When results look anomalous - all zeros, all identical scores, 100% failure rates, or identical A/B comparisons - STOP and investigate the underlying formula/logic before reporting. Never present suspicious results at face value. Ask: "Is this plausible given the inputs?"

Examples of past bugs this would have caught:
- 8,194 forward tests all scoring 0/100 (compound-inflated backtest scores broke the ratio)
- Identical A/B scores between baseline and ML runs (75% of confidence score used pre-WF metrics)
- All ML exit signals = 0 (thresholds set outside the model's prediction range)

### Plan before executing
When the user is discussing or exploring an approach, produce a plan first. Do not start running pipelines, writing code, or making changes until explicitly told to execute. If ambiguous, ask: "Should I plan this out first, or go ahead and implement?"

### Respect log warnings as potential blockers
When pipeline logs show recurring warnings (especially about data overlap, OOS windows, or validation), flag them immediately. Do not treat warnings that appear on every run as background noise - they often indicate fundamental issues.

### Verify thresholds against actual data distributions
Before setting any threshold, filter, or cutoff value, check what the actual data/model produces. ML model outputs, scoring formulas, and filter criteria must be validated against real distributions, not theoretical ranges.

## Quality Score (Universal Metric)

The entire pipeline uses one consistent scoring metric:

```
Quality Score = Sortino × R² × min(PF, 5) × √min(Trades, 200) × (1 + min(Return%, 200) / 100) / (Ulcer + MaxDD%/2 + 5)
```

**Sortino** replaces Sharpe — doesn't penalize upside volatility, better for asymmetric strategies (trend following, big R:R). **Ulcer Index** alongside MaxDD captures chronic drawdown pain (time underwater), not just worst-case depth. Two systems with identical MaxDD but different recovery times score differently.

This rewards smooth equity curves (R²), risk-adjusted profit (Sortino), absolute profit (return multiplier 1x-3x), efficiency (PF), statistical confidence (√trades), and penalizes both chronic and peak drawdown. Return% is capped at 200% to prevent compound sizing inflation on high-frequency timeframes. It is used for:
- Optimization objective (what Optuna maximizes)
- Combined ranking (back + forward)
- Walk-forward window pass/fail (quality_score > 0)
- Stability analysis (baseline and neighbor comparison)
- Confidence scoring (15% weight as "Quality Score" component)

**Hard pre-filters** in the Optuna objective reject garbage before scoring:
- MaxDD > 30% → instant reject (configurable: `optimization.max_dd_hard_limit`)
- R² < 0.5 → instant reject (configurable: `optimization.min_r2_hard`)

**Metrics namedtuple** has 10 fields: trades, win_rate, profit_factor, sharpe, max_dd, total_return, r_squared, ontester_score, sortino, ulcer.

Returns 0 when Sortino ≤ 0, PF ≤ 0, R² ≤ 0, or no trades. Defined in `optimization/numba_backtest.py`.

## Pipeline Execution Checklist

When running pipeline optimizations:
1. Output directory must be unique per run (timestamp-based, already handled by pipeline)
2. Verify `--test-months` CLI arg matches intended config (CLI overrides config.py defaults)
3. Quality Score handles all timeframes universally - no special treatment for M15/M5 vs H1
4. After each run: check OOS window count (`oos_n_windows`). If 0, all comparisons are in-sample
5. For A/B tests: verify the best candidate is actually affected by the treatment (e.g., ML filter may return None for some candidates)
6. State serialization: `state.json` must include `trade_details` for resume capability

## Strategy Defaults

Signal stage uses risk group defaults. If defaults produce negative expectancy, optimization finds 0 valid results regardless of signal quality. Always verify:
- `tp_atr_mult` > 0 (zero = no take profit = all trades hit SL)
- `tp_mode='rr'` with `tp_rr_ratio >= 3.0` is the safe default for signal-stage testing
- All filters/management OFF by default so signal stage tests pure signal quality
