# OANDA Trading System

## Project Overview

Python-based algorithmic trading system using OANDA API with a 7-stage optimization pipeline (Data -> Optimization -> Walk-Forward -> Stability -> Monte Carlo -> Confidence -> Report).

Key directories:
- `strategies/` - Strategy implementations (V1-V5 RSI divergence, V6 EMA cross)
- `pipeline/` - Pipeline stages (`stages/s1-s7_*.py`), config, state, ML exit, report generation
- `optimization/` - Optuna optimizer (`unified_optimizer.py`), numba backtesting, ML features
- `live/` - Live/paper trading (trader.py, oanda_client.py, position_manager.py, pipeline_adapter.py)
- `data/` - Data download and caching (M1 parquet -> any timeframe)
- `results/pipelines/` - Pipeline run outputs (each run gets a unique timestamped directory)
- `config/` - Settings, exported parameter files
- `docs/` - Documentation (README, SYSTEM, STRATEGY_EVOLUTION, KNOWN_ISSUES)

Best strategy: V3 RSI Divergence (89.8/100 GREEN on H1, 87.2/100 GREEN on M15).

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

## Pipeline Execution Checklist

When running pipeline optimizations:
1. Output directory must be unique per run (timestamp-based, already handled by pipeline)
2. Verify `--test-months` CLI arg matches intended config (CLI overrides config.py defaults)
3. Check scoring formulas handle edge cases: compound sizing inflation (use Sharpe not dollar profit for high-frequency), very few trades, zero-variance results
4. After each run: check OOS window count (`oos_n_windows`). If 0, all comparisons are in-sample
5. For A/B tests: verify the best candidate is actually affected by the treatment (e.g., ML filter may return None for some candidates)
6. State serialization: `state.json` must include `trade_details` for resume capability

## Strategy Defaults

Signal stage uses risk group defaults. If defaults produce negative expectancy, optimization finds 0 valid results regardless of signal quality. Always verify:
- `tp_atr_mult` > 0 (zero = no take profit = all trades hit SL)
- `tp_mode='rr'` with `tp_rr_ratio >= 3.0` is the safe default for signal-stage testing
- All filters/management OFF by default so signal stage tests pure signal quality
