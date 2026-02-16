# OANDA Trading System

Automated trading strategy validation pipeline using OANDA data, with live/paper trading support.

## What It Does

Tests trading strategies through a rigorous 7-stage pipeline:
1. **Data** - Download and validate OANDA data (M1 -> higher TF)
2. **Optimization** - Find best parameters with staged Optuna (TPE sampler)
3. **Walk-Forward** - Validate across rolling time windows
4. **Stability** - Test parameter robustness (+-10% perturbation)
5. **Monte Carlo** - Stress test with trade randomization + bootstrap
6. **Confidence** - Score 0-100 (RED/YELLOW/GREEN)
7. **Report** - Generate interactive 7-tab HTML dashboard

Then deploys validated strategies to paper or live trading with full risk management.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure OANDA credentials
cp .env.example .env
# Edit .env with your API key

# Run pipeline on single pair
python scripts/run_pipeline.py --pair GBP_USD --timeframe H1

# Specify strategy version
python scripts/run_pipeline.py --pair GBP_USD --timeframe H1 --strategy rsi_full_v3

# Run paper trading with optimized params
python scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_M15_20260210_063223

# Run multi-symbol test
python scripts/run_multi_symbol.py --strategy Simple_Trend
```

## Documentation

| Doc | Purpose |
|-----|---------|
| **[SYSTEM.md](SYSTEM.md)** | Developer guide - architecture, code structure, key files |
| **[STRATEGY_EVOLUTION.md](STRATEGY_EVOLUTION.md)** | Strategy V1-V6 history, results, lessons learned |
| **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)** | Consolidated bug tracker with status |
| **[LIVE_TRADING.md](LIVE_TRADING.md)** | Live/paper trading setup, VPS deployment, monitoring |
| **[QUALITY_ASSESSMENT.md](QUALITY_ASSESSMENT.md)** | Pipeline quality grade (56/100) and improvement priorities |
| **[ML_TRADING_RESEARCH_BRIEF.md](ML_TRADING_RESEARCH_BRIEF.md)** | ML research summary and conclusions |

### Archived (Historical Reference)
| Doc | Purpose |
|-----|---------|
| [archive/ML_EXIT_PROGRAM.md](archive/ML_EXIT_PROGRAM.md) | ML exit development program (concluded: dead end) |
| [archive/EXIT_FIRST_ML_DEVELOPMENT_WRITEUP.md](archive/EXIT_FIRST_ML_DEVELOPMENT_WRITEUP.md) | Original ML exit writeup (reference) |
| [archive/chatGPTMLresearch.txt](archive/chatGPTMLresearch.txt) | ChatGPT ML research notes |
| [archive/PROJECT_PLAN.md](archive/PROJECT_PLAN.md) | Original project plan (2026-02-01) |
| [archive/HANDOVER.md](archive/HANDOVER.md) | Development session notes (2026-02-02/03) |

## Key Directories

- `pipeline/` - Core 7-stage validation pipeline
- `optimization/` - Numba backtester + Optuna optimizer + ML features
- `strategies/` - Trading strategies (6 versions)
- `scripts/` - CLI entry points
- `live/` - Paper/live trading infrastructure
- `config/` - Settings and exported parameter files
- `docs/` - All project documentation

## Current Strategies

> **Note:** All scores below are pre-exit-slippage-model (slippage added Feb 16, 2026). M15 results are particularly affected and considered invalid without re-optimization. H1 results are less affected but need re-validation.

| Strategy | File | Params | Best Score (H1) | Best Score (M15) | Notes |
|----------|------|--------|-----------------|-------------------|-------|
| RSI Divergence v1 | archived | 35 | 93/100 GREEN | - | Archived to `strategies/archive/` |
| RSI Divergence v2 | archived | 35 | - | - | Archived to `strategies/archive/` |
| RSI Divergence v3 | `rsi_full_v3.py` | 32 | 89.8/100 GREEN | 87.2/100 GREEN | **Recommended** - stability-hardened |
| RSI Divergence v4 | `rsi_full_v4.py` | 34 | 81.5/100 GREEN | - | Trade management optimization |
| RSI Divergence v5 | `rsi_full_v5.py` | 37 | 71.8/100 GREEN | - | Chandelier + stale exit |
| EMA Cross v6 | `ema_cross_ml.py` | 6 | 72.8/100 GREEN | - | EMA crossover, weaker than RSI |

## Status

- [x] OANDA data download (M1 preferred, auto-builds higher TF)
- [x] Numba JIT backtester (fast, ~50-100x vs pure Python)
- [x] Staged Optuna optimization (5 stages + final)
- [x] Walk-forward validation (rolling windows)
- [x] Parameter stability testing (+-10% perturbation)
- [x] Monte Carlo simulation (shuffle + bootstrap + permutation)
- [x] Confidence scoring (0-100, RED/YELLOW/GREEN)
- [x] HTML report generation (7-tab interactive dashboard)
- [x] Multi-symbol parallel testing
- [x] Paper/live trading infrastructure
- [x] ML exit model (concluded: dead end, code archived to `pipeline/archive/ml_exit/`)
- [x] ML entry filter (concluded: dead end, 8 A/B tests all neutral)
- [x] VPS deployment (Windows Server)
- [ ] Telegram alerts (implemented, not wired in)
- [ ] Multi-pair simultaneous trading
- [ ] News/session time filters
- [ ] Auto-reconnection on API errors

## License

Private - Internal use only
