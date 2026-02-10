# OANDA Trading System

Automated trading strategy validation pipeline using OANDA data.

## What It Does

Tests trading strategies through a rigorous 7-stage pipeline:
1. **Data** - Download and validate OANDA data (M1 -> higher TF)
2. **Optimization** - Find best parameters with staged Optuna (TPE sampler)
3. **Walk-Forward** - Validate across rolling time windows
4. **Stability** - Test parameter robustness (+-10% perturbation)
5. **Monte Carlo** - Stress test with trade randomization + bootstrap
6. **Confidence** - Score 0-100 (RED/YELLOW/GREEN)
7. **Report** - Generate interactive 7-tab HTML dashboard

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

# Run multi-symbol test
python scripts/run_multi_symbol.py --strategy Simple_Trend
```

## Documentation

| Doc | Purpose |
|-----|---------|
| **[SYSTEM.md](SYSTEM.md)** | Developer guide - architecture, code structure, key files |
| **[STRATEGY_EVOLUTION.md](STRATEGY_EVOLUTION.md)** | Strategy V1-V6 history, results, lessons learned |
| **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)** | Consolidated bug tracker with status |
| **[ML_EXIT_PROGRAM.md](ML_EXIT_PROGRAM.md)** | ML exit development program (our adapted plan) |
| **[HANDOVER.md](HANDOVER.md)** | Sprint 5 execution handover for RSI V3 ML exits |
| **[EXIT_FIRST_ML_DEVELOPMENT_WRITEUP.md](EXIT_FIRST_ML_DEVELOPMENT_WRITEUP.md)** | Original ML exit writeup (reference) |
| **[QUALITY_ASSESSMENT.md](QUALITY_ASSESSMENT.md)** | Pipeline quality grade (56/100) and improvement priorities |

### Archived (Historical Reference)
| Doc | Purpose |
|-----|---------|
| [archive/PROJECT_PLAN.md](archive/PROJECT_PLAN.md) | Original project plan (2026-02-01) |
| [archive/HANDOVER.md](archive/HANDOVER.md) | Development session notes (2026-02-02/03) |
| [archive/V2_IMPROVEMENTS.md](archive/V2_IMPROVEMENTS.md) | V2 strategy features (superseded by V3+) |
| [archive/ROBUSTNESS_ANALYSIS.md](archive/ROBUSTNESS_ANALYSIS.md) | EUR vs GBP pair analysis |
| [archive/OPTIMIZATION_IMPROVEMENTS.md](archive/OPTIMIZATION_IMPROVEMENTS.md) | Speed improvement ideas |
| [archive/PLAN-team-dashboard.md](archive/PLAN-team-dashboard.md) | Agent team dashboard proposal |

## Key Directories

- `pipeline/` - Core 7-stage validation pipeline
- `optimization/` - Numba backtester + Optuna optimizer + ML features
- `strategies/` - Trading strategies (6 versions)
- `scripts/` - CLI entry points
- `live/` - Paper/live trading infrastructure
- `docs/` - All project documentation

## Current Strategies

| Strategy | File | Params | Best Score | Notes |
|----------|------|--------|------------|-------|
| RSI Divergence v1 | `rsi_full.py` | 35 | 93/100 GREEN | Original, best single score |
| RSI Divergence v2 | `rsi_full_v2.py` | 35 | - | Stability fixes |
| RSI Divergence v3 | `rsi_full_v3.py` | 32 | 89.8/100 GREEN | Stability-hardened, recommended |
| RSI Divergence v4 | `rsi_full_v4.py` | 34 | 81.5/100 GREEN | Trade management optimization |
| RSI Divergence v5 | `rsi_full_v5.py` | 37 | 71.8/100 GREEN | Chandelier + stale exit |
| EMA Cross + ML Exit | `ema_cross_ml.py` | 17 | Not tested | ML-based exit signals |

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
- [x] Paper trading infrastructure
- [ ] ML exit model (in development - see ML_EXIT_PROGRAM.md)
- [ ] Production live trading

## License

Private - Internal use only
