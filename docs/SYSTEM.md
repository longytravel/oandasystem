# OANDA Trading System - Developer Guide

**Last Updated:** 2026-02-07

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up OANDA credentials
cp .env.example .env
# Edit .env with your OANDA API key

# Run a pipeline (recommended: V3 strategy)
python scripts/run_pipeline.py --pair GBP_USD --timeframe H1 --strategy rsi_full_v3

# Run with custom config
python scripts/run_pipeline.py --pair GBP_USD --timeframe H1 --strategy rsi_full_v3 \
  --trials 5000 --test-months 6
```

**IMPORTANT:** `--test-months` defaults to 3 in CLI, overriding config.py's default of 6. Always specify explicitly.

---

## System Architecture

```
+-----------------------------------------------------------------------+
|                           7-STAGE PIPELINE                             |
+-----------------------------------------------------------------------+
|  Stage 1: DATA          Download & validate OANDA data                 |
|  Stage 2: OPTIMIZATION  Staged Optuna optimization (TPE sampler)       |
|  Stage 3: WALK-FORWARD  Rolling window validation (fixed params)       |
|  Stage 4: STABILITY     +-10% parameter perturbation testing           |
|  Stage 5: MONTE CARLO   Shuffle + bootstrap + permutation testing      |
|  Stage 6: CONFIDENCE    Scoring (0-100) -> RED/YELLOW/GREEN            |
|  Stage 7: REPORT        7-tab interactive HTML dashboard               |
+-----------------------------------------------------------------------+
```

---

## Directory Structure

```
oandasystem/
|
+-- pipeline/                    # CORE: 7-stage validation pipeline
|   +-- pipeline.py              # Main orchestrator
|   +-- config.py                # Pipeline configuration
|   +-- state.py                 # Resumable state management
|   +-- stages/
|   |   +-- s1_data.py           # Data download & validation
|   |   +-- s2_optimization.py   # Staged Optuna optimization
|   |   +-- s3_walkforward.py    # Walk-forward validation
|   |   +-- s4_stability.py      # Parameter stability testing
|   |   +-- s5_montecarlo.py     # Monte Carlo simulation
|   |   +-- s6_confidence.py     # Confidence scoring
|   |   +-- s7_report.py         # Report orchestrator
|   +-- report/                  # Report generation package
|       +-- style.py             # CSS/theme (dark theme)
|       +-- data_collector.py    # Gathers data from all stages
|       +-- chart_generators.py  # Plotly chart generation
|       +-- html_builder.py      # 7-tab HTML dashboard builder
|
+-- optimization/                # CORE: Fast backtesting engine
|   +-- numba_backtest.py        # Numba JIT backtester (3 functions)
|   +-- unified_optimizer.py     # Optuna + staged optimization
|   +-- fast_strategy.py         # Strategy interface for optimization
|   +-- ml_features.py           # ML feature computation (8 OHLC features)
|
+-- data/                        # Data management
|   +-- download.py              # OANDA data downloader (M1 -> higher TF)
|   +-- oanda/                   # Cached parquet files
|
+-- strategies/                  # Trading strategies
|   +-- __init__.py              # Strategy registry
|   +-- rsi_full.py              # V1: RSI Divergence (35 params)
|   +-- rsi_full_v2.py           # V2: + regime/quality filters (35 params)
|   +-- rsi_full_v3.py           # V3: Stability-hardened (32 params) RECOMMENDED
|   +-- rsi_full_v4.py           # V4: + BE/trail chaining (34 params)
|   +-- rsi_full_v5.py           # V5: + chandelier/stale exit (37 params)
|   +-- ema_cross_ml.py          # V6: EMA cross + ML exit (17 params)
|   +-- rsi_divergence.py        # Legacy (not used by pipeline)
|   +-- rsi_fast.py              # Strategy registry helper
|   +-- trend_simple.py          # Simple trend (6 params, baseline)
|
+-- live/                        # Live/paper trading
|   +-- oanda_client.py          # OANDA v20 API wrapper
|   +-- trader.py                # Live execution engine
|   +-- position_manager.py      # Position tracking
|   +-- risk_manager.py          # Risk controls
|   +-- alerts.py                # Notifications
|
+-- scripts/                     # CLI entry points
|   +-- run_pipeline.py          # Single pair pipeline (MAIN)
|   +-- run_multi_symbol.py      # Multi-symbol parallel testing
|   +-- run_live.py              # Live trading
|   +-- download_data.py         # Data management
|   +-- plot_equity.py           # Equity curve visualization
|   +-- plot_stability.py        # Stability chart visualization
|   +-- run_optimization.py      # Standalone optimization (legacy)
|   +-- run_robust_optimization.py # Robust optimization (legacy)
|
+-- backtesting/                 # Legacy backtest engine (NOT used by pipeline)
|   +-- engine.py                # Old Python backtester
|
+-- results/                     # Output directory
|   +-- pipelines/               # Pipeline run artifacts
|       +-- {pair}_{tf}_{timestamp}/
|           +-- state.json       # Resumable state
|           +-- stage_*.json     # Per-stage outputs
|           +-- report.html      # Final HTML report
|           +-- report_data.json # Report data (JSON)
|
+-- docs/                        # Project documentation
|   +-- README.md                # Project overview
|   +-- SYSTEM.md                # This file
|   +-- STRATEGY_EVOLUTION.md    # V1-V6 history and results
|   +-- KNOWN_ISSUES.md          # Bug tracker
|   +-- ML_EXIT_PROGRAM.md       # ML exit development plan
|   +-- QUALITY_ASSESSMENT.md    # Pipeline quality grade
|   +-- EXIT_FIRST_ML_DEVELOPMENT_WRITEUP.md  # Original ML writeup
|   +-- archive/                 # Historical docs
|
+-- .claude/                     # AI agent configuration
    +-- skills/                  # User-invocable workflows
    +-- agents/                  # Autonomous agents
```

---

## Key Files for Review

### 1. Pipeline Core
| File | Purpose | Key |
|------|---------|-----|
| `pipeline/pipeline.py` | Main orchestrator - runs all 7 stages | Resumable, stage-by-stage |
| `pipeline/config.py` | All configuration (data, optimization, WF, stability, MC, confidence) | Defaults here, CLI overrides |
| `pipeline/state.py` | State management for resume | JSON-based persistence |

### 2. Optimization Engine
| File | Purpose | Key |
|------|---------|-----|
| `optimization/numba_backtest.py` | Numba JIT backtester | 3 functions: basic, full, full_with_trades |
| `optimization/unified_optimizer.py` | Optuna + staged optimization + stability testing | 5 stages + final |
| `optimization/fast_strategy.py` | Strategy interface | precompute, filter_signals, compute_sl_tp |
| `optimization/ml_features.py` | ML feature computation | 8 OHLC features, direction-aware scoring |

### 3. Numba Backtest Functions
| Function | Used By | Returns |
|----------|---------|---------|
| `basic_backtest_numba` | S2 optimization (fast) | Metrics only |
| `full_backtest_numba` | S3 walk-forward, S4 stability | Metrics + equity curve |
| `full_backtest_with_trades` | S5 Monte Carlo, reporting | Metrics + equity + trade details |

### 4. Report Package
| File | Purpose |
|------|---------|
| `pipeline/report/data_collector.py` | Gathers all stage data for report |
| `pipeline/report/chart_generators.py` | Plotly charts (equity, WF, stability, MC) |
| `pipeline/report/html_builder.py` | 7-tab HTML dashboard with lazy rendering |
| `pipeline/report/style.py` | Dark theme CSS |

---

## Configuration

### Environment Variables (.env)
```
OANDA_ACCOUNT_ID=xxx-xxx-xxxxxxx-xxx
OANDA_ACCESS_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OANDA_ENVIRONMENT=practice
```

### Pipeline Config (pipeline/config.py) - Key Values
```python
# Data
data.years = 3.0         # 3yr sweet spot (5yr too long, 2yr too few windows)
data.back_pct = 0.8      # 80% back, 20% forward

# Optimization
optimization.trials_per_stage = 5000
optimization.final_trials = 10000
optimization.top_n_candidates = 20

# Walk-Forward
walkforward.train_months = 6
walkforward.test_months = 6     # CLI default is 3! Always specify --test-months 6
walkforward.roll_months = 3
walkforward.min_trades_per_window = 5  # Low-freq strategies need lower threshold

# Stability
stability.perturbation_pct = 0.10  # +-10% perturbation

# Confidence
confidence.green_threshold = 70
confidence.yellow_threshold = 40

# Candidate Filtering
min_forward_ratio = 0.15   # Forward/back minimum (see QUALITY_ASSESSMENT for discussion)
forward_rank_weight = 2.0  # Forward weighted 2x in ranking
```

---

## Adding a New Strategy

1. Create `strategies/my_strategy.py` implementing `FastStrategy`:
   - `get_parameter_groups()` - Define staged optimization groups
   - `precompute(df)` - Generate all possible signals (called ONCE)
   - `filter_signals(signals, params)` - Filter by params (called PER TRIAL)
   - `compute_sl_tp(signal, params, pip_size)` - Calculate SL/TP

2. Register in `strategies/__init__.py`

3. Run: `python scripts/run_pipeline.py --strategy my_strategy --pair GBP_USD`

See [STRATEGY_EVOLUTION.md](STRATEGY_EVOLUTION.md) for detailed examples and lessons.

---

## Numba Backtest Signature

The numba functions accept a large parameter signature. V4+ added trade management chaining. V5 added chandelier/stale exit. V6 added ML arrays. All are backward compatible (pass zeros/False to disable).

Key parameters after the standard ones:
```
# V4 additions
chain_be_to_trail    # Replaced in V5 by trail_mode

# V5 additions
trail_mode           # 0=standard, 1=chandelier
chandelier_atr_mult  # ATR multiplier for chandelier
atr_pips             # Pre-computed ATR in pip units
stale_exit_bars      # Close after N bars with no progress

# V6 additions
ml_long_scores       # ML exit scores for long positions (array)
ml_short_scores      # ML exit scores for short positions (array)
use_ml_exit          # Enable ML exit (0/1)
ml_min_hold          # Minimum bars before ML can exit
ml_threshold         # ML score threshold for exit
```

---

## Key Concepts

### OnTester Score (MT5-style)
```
OnTester = Profit * R-squared * ProfitFactor * sqrt(Trades) / (MaxDrawdown + 5)
```

### Confidence Score (0-100)
Weighted combination of 6 components:
- Walk-forward pass rate (25%)
- Forward/back performance ratio (15%)
- Stability score (15%)
- Monte Carlo results (15%)
- Backtest quality (15%)
- Forward Sharpe (15%)

Rating: RED (0-40), YELLOW (40-70), GREEN (70+)

### Known Issues
See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for current bug tracker.
See [QUALITY_ASSESSMENT.md](QUALITY_ASSESSMENT.md) for pipeline quality grade.
