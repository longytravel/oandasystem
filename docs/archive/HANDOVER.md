# OANDA Trading System - Developer Handover

**Last Updated:** 2026-02-03 (Night)
**Status:** ROBUSTNESS ANALYSIS COMPLETE - Pair selection validated

---

## Session Summary (2026-02-03)

### Critical Finding: Pair Selection Matters Most

After implementing **parameter stability analysis**, we discovered:

| Pair | Forward Score | Forward/Back | Verdict |
|------|---------------|--------------|---------|
| EUR_USD | **0.0** | 0% | DO NOT TRADE |
| GBP_USD | 572.9 | 86% | TRADEABLE |

**EUR_USD does not work with RSI divergence** - zero forward performance across ALL parameter combinations. No amount of optimization will fix this.

**GBP_USD has a real edge** - forward performance validates backtest, most parameters are stable.

### What Was Done
1. Implemented V2 improvements (regime filters, quality scoring)
2. Found V1 vs V2 comparison was misleading (different optimizer settings)
3. Created **parameter stability analysis** - tests if neighbors also perform well
4. Discovered EUR_USD has NO forward edge (0.0 for all candidates)
5. Validated GBP_USD shows genuine edge (86% forward retention)
6. Created robust optimization mode and production config

### New Files Created
- `optimization/unified_optimizer.py` - Added `analyze_parameter_stability()`, `run_robust_optimization()`
- `scripts/run_robust_optimization.py` - CLI for robust optimization
- `docs/ROBUSTNESS_ANALYSIS.md` - Full analysis and methodology
- `config/GBP_USD_H1_robust.json` - Production-ready config for GBP_USD
- `strategies/rsi_divergence_v2.py` - V2 strategy with regime/quality filters
- `docs/V2_IMPROVEMENTS.md` - V2 feature documentation

### Key Insight
The problem isn't finding better parameters - it's choosing the right PAIR. The stability analysis proves this: EUR_USD parameters are "stable" on backtest (80%) but have ZERO forward performance. GBP_USD parameters are stable AND have forward performance.

---

## Session Summary (2026-02-02)

### What Was Done
1. Ran staged optimization on EUR_USD M1 (737k candles, 1000 trials/stage)
2. Created `scripts/plot_equity.py` for validation backtests
3. Fixed spread handling (was missing - caused inflated metrics)
4. Validated results match optimization (within expected variance)
5. Created `/oanda-optimize` Claude skill for interactive workflow

### Validated Results (EUR_USD M1)
| Metric | Optimization | Validation Backtest |
|--------|--------------|---------------------|
| Trades | 3,272 | 3,273 ✓ |
| Win Rate | 69%/68% | 68.7% ✓ |
| Profit Factor | 1.34/1.16 | 1.24 ✓ |
| Sharpe | 1.52/0.88 | 1.06 ✓ |
| Max DD | 12.7%/15.4% | 15.4% ✓ |

### Files Created/Modified
- `scripts/plot_equity.py` - Validation backtest with equity curve
- `scripts/download_data.py` - Now uses append logic (was creating separate files)
- `~/.claude/skills/oanda-optimize/SKILL.md` - Interactive optimization skill

### Known Gaps (Not Yet Implemented)
- **Slippage**: Not modeled (add 0.5-1 pip for realism)
- **Variable spread**: Fixed at 1.5 pips
- **Walk-forward**: No rolling windows
- **Monte Carlo**: No trade order randomization

---

## Quick Reference

### Run Optimization (Main Workflow)

```bash
# Standard staged optimization with MT5-style ranking
python scripts/run_optimization.py --staged --trials-per-stage 5000

# Quick test run (faster, fewer params)
python scripts/run_optimization.py --quick --trials 5000

# Show what parameters exist
python scripts/run_optimization.py --show-groups --strategy rsi_full

# List available strategies
python scripts/run_optimization.py --list-strategies
```

### Robust Optimization (NEW - Recommended)

```bash
# Run robust optimization with stability analysis
python scripts/run_robust_optimization.py --pair GBP_USD --timeframe H1 --trials 3000

# Test more candidates for stability
python scripts/run_robust_optimization.py --pair GBP_USD --candidates 50 --min-stability 0.7
```

**This is the preferred method** - filters out overfit results by testing parameter stability.

### Download Data

```bash
python scripts/download_data.py --pair GBP_USD --granularity M1 --years 2
python scripts/download_data.py --pair EUR_USD --granularity M1 --years 2
# Data auto-appends - no need to re-download if updating
```

### Validate Results (NEW)

```bash
# Run validation backtest with spread, show equity curve
python scripts/plot_equity.py \
  --results results/optimization/RSI_Divergence_Full_LATEST.json \
  --pair EUR_USD \
  --timeframe M1 \
  --use-best \
  --save results/equity_EUR_USD.png
```

### Interactive Skill (NEW)

```bash
# Use the Claude skill for guided workflow
/oanda-optimize EUR_USD
```

---

## Ranking System (MT5-Style)

### How Results Are Ranked

This system mirrors MT5's optimization approach:

1. **Run back optimization** - get OnTester scores for all parameter combinations
2. **Forward test ALL results** - not just top N
3. **Rank by back OnTester** - #1 = highest back score
4. **Rank by forward OnTester** - #1 = highest forward score
5. **Combined Rank = back_rank + forward_rank**
6. **Sort by combined rank** - lower is better

### Why Combined Ranking?

| Result | Back Rank | Forward Rank | Combined | Verdict |
|--------|-----------|--------------|----------|---------|
| A | #1 | #100 | 101 | BAD - unbalanced |
| B | #15 | #12 | 27 | GOOD - consistent |

Result B is "good in both" - more reliable than Result A which dominates back but fails forward.

### OnTester Score Formula

```
Score = Profit × R² × ProfitFactor × √Trades / (MaxDrawdown + 5)
```

**Components:**
- **Profit**: Total profit in account currency
- **R²**: Equity curve smoothness (0-1, higher = smoother)
- **ProfitFactor**: Gross profit / gross loss
- **√Trades**: Square root of trade count (rewards more trades)
- **DD + 5**: Penalizes drawdown (the +5 prevents division by zero)

### R² (Equity Curve Smoothness)

R² measures how linear the equity curve is:
- **R² = 1.0**: Perfectly smooth, steady growth
- **R² = 0.5**: Choppy, inconsistent
- **R² = 0.0**: Random walk

High R² prevents:
- Lucky trades dominating results
- Curve-fitting to a few big wins
- Strategies that are stressful to trade

---

## System Overview

### What This System Does

1. **Downloads** OANDA price data (H1, M1 candles)
2. **Pre-computes** all possible RSI divergence signals
3. **Optimizes** 36 parameters using Optuna TPE sampling
4. **Calculates** R² and OnTester score for each result
5. **Forward tests** ALL valid results
6. **Ranks** by combined back+forward performance

### Core Components

| File | Purpose | Key Functions |
|------|---------|---------------|
| `optimization/numba_backtest.py` | Numba backtester with R² | `basic_backtest_numba()`, `full_backtest_numba()`, `calculate_r_squared()` |
| `optimization/unified_optimizer.py` | Optimizer with combined ranking | `UnifiedOptimizer.run()`, `_apply_combined_ranking()` |
| `optimization/fast_strategy.py` | Strategy interface | `FastStrategy` base class |
| `strategies/rsi_full.py` | 36-param RSI divergence | `RSIDivergenceFullFast` |
| `strategies/rsi_fast.py` | Strategy registry | `get_strategy()`, `FAST_STRATEGIES` |
| `scripts/run_optimization.py` | CLI entry point | Main script |

---

## How Optimization Works

### Staged Mode (Recommended)

The optimizer runs in 5 stages, locking parameters at each stage:

```
Stage 1: SIGNAL (8 params)     -> Lock best signal params
Stage 2: FILTERS (7 params)    -> Lock best filter params
Stage 3: RISK (7 params)       -> Lock best SL/TP params
Stage 4: MANAGEMENT (8 params) -> Lock trailing/BE/partial
Stage 5: TIME (6 params)       -> Lock trading hours/days
FINAL:   All 36 params         -> Fine-tune with tight ranges (±20%)
```

Each stage optimizes for **OnTester score**, locks the best values, then moves on.

### Forward Validation

After back-testing:
- ALL valid results are forward tested (not just top N)
- Combined rank calculated for each
- Results sorted by combined rank (lower = better)

### Output

```
COMBINED RANKING RESULTS - RSI_Divergence_Full
==============================================================================
Comb   Back#  Fwd#   BackScore   FwdScore    BackR²  FwdR²   BackTr  FwdTr
------------------------------------------------------------------------------
27     15     12     8234.5      2156.3      0.892   0.845   156     42
38     12     26     9456.2      1834.5      0.856   0.812   143     38
```

---

## Parameter Groups

### 1. Signal (8 params)
```
rsi_period:        [7, 10, 14, 21]
rsi_overbought:    [70, 75, 80]
rsi_oversold:      [20, 25, 30]
min_rsi_diff:      [3, 5, 8, 12]
swing_strength:    [3, 5, 7]
min_bars_between:  [5, 10, 15, 20]
max_bars_between:  [40, 60, 80, 100]
require_pullback:  [True, False]
```

### 2. Filters (7 params)
```
use_slope_filter:      [True, False]
min_price_slope:       [5, 15, 25]
max_price_slope:       [50, 65, 80]
use_rsi_extreme_filter:[True, False]
use_trend_filter:      [True, False]
trend_ma_period:       [50, 100, 200]
max_spread_pips:       [2, 3, 5]
```

### 3. Risk (7 params)
```
sl_mode:        ['fixed', 'atr', 'swing']
sl_fixed_pips:  [25, 35, 50, 75]
sl_atr_mult:    [1.5, 2.0, 2.5, 3.0]
sl_swing_buffer:[5, 10, 15]
tp_mode:        ['rr', 'atr', 'fixed']
tp_rr_ratio:    [1.0, 1.5, 2.0, 2.5, 3.0]
tp_atr_mult:    [2.0, 3.0, 4.0]
```

### 4. Management (8 params)
```
use_trailing:      [True, False]
trail_start_pips:  [20, 30, 50]
trail_step_pips:   [10, 15, 20]
use_break_even:    [True, False]
be_trigger_pips:   [15, 20, 30, 40]
be_offset_pips:    [0, 2, 5]
use_partial_close: [True, False]
partial_close_pct: [0.3, 0.5, 0.7]
```

### 5. Time (6 params)
```
use_time_filter:   [True, False]
trade_start_hour:  [0, 2, 4, 6, 8]
trade_end_hour:    [18, 20, 22, 23]
trade_monday:      [True, False]
trade_friday:      [True, False]
friday_close_hour: [18, 20]
```

---

## File Structure

```
oandasystem/
├── config/
│   └── settings.py           # OANDA API keys, paths
├── data/
│   ├── download.py           # OANDA data download
│   └── oanda/                # Cached parquet files
│       ├── GBP_USD_H1.parquet
│       ├── GBP_USD_M1.parquet
│       └── EUR_USD_H1.parquet
├── optimization/
│   ├── fast_strategy.py      # FastStrategy interface
│   ├── numba_backtest.py     # Numba engines with R² and OnTester
│   └── unified_optimizer.py  # Combined ranking optimizer
├── strategies/
│   ├── base.py               # Base strategy (unused for optimization)
│   ├── rsi_divergence.py     # Original strategy (unused for optimization)
│   ├── rsi_fast.py           # Strategy registry + simple 9-param version
│   └── rsi_full.py           # Full 36-param version
├── backtesting/
│   └── engine.py             # Main backtest engine (not used by optimizer)
├── scripts/
│   ├── download_data.py      # Download OANDA data
│   ├── run_backtest.py       # Single backtest (not optimization)
│   └── run_optimization.py   # MAIN ENTRY POINT
├── results/
│   └── optimization/         # JSON results files
├── live/                     # Live trading (not implemented)
├── logs/                     # Log files
└── tests/                    # Unit tests (empty)
```

---

## Known Issues / TODO

### Current State
- [x] Staged optimization working
- [x] R² equity curve smoothness
- [x] OnTester score (MT5-style)
- [x] Combined back+forward ranking
- [x] Forward tests ALL results
- [x] Results saved to JSON
- [ ] Unit tests not written
- [ ] Live trading not implemented
- [ ] Walk-forward analysis not implemented

### Potential Issues to Watch
1. **Speed slows during optimization** - Optuna TPE slows as it learns. First 1000 ~40/sec, last 1000 ~3/sec
2. **Low OnTester scores** - Means low R² or high drawdown. Check equity curve smoothness
3. **Combined rank ties** - Multiple results can have same combined rank
4. **Memory on forward test** - Now tests ALL results, not just top N
5. **Validation metrics don't match** - Usually means spread not applied. The optimizer uses 1.5 pip spread. If validation backtest shows BETTER metrics, spread is likely missing.

### To Fix/Improve
- [ ] Add walk-forward analysis (rolling optimization windows)
- [ ] Cross-pair validation (optimize on GBP, test on EUR)
- [ ] Add more strategies beyond RSI divergence
- [ ] Unit tests for Numba functions
- [ ] Better error handling in download script
- [ ] Progress persistence (resume interrupted optimization)

---

## Adding a New Strategy

### Step 1: Create Strategy File

```python
# strategies/my_strategy.py
from optimization.fast_strategy import FastStrategy, FastSignal, ParameterGroup, ParameterDef
from typing import Dict, List, Any
import numpy as np

class MyStrategyFast(FastStrategy):
    name = "My_Strategy"
    version = "1.0"

    def get_parameter_groups(self) -> Dict[str, ParameterGroup]:
        """Define parameter groups for staged optimization"""
        groups = {}

        signal = ParameterGroup('signal', 'Signal parameters')
        signal.add_param('period', [10, 20, 50], default=20)
        signal.add_param('threshold', [0.5, 1.0, 1.5], default=1.0)
        groups['signal'] = signal

        return groups

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Flat parameter space for non-staged optimization"""
        space = {}
        for group in self.get_parameter_groups().values():
            for name, param in group.parameters.items():
                space[name] = param.values
        return space

    def precompute(self, df) -> List[FastSignal]:
        """Pre-compute ALL possible signals (called ONCE)"""
        signals = []
        # Your signal detection logic here
        return signals

    def filter_signals(self, signals: List[FastSignal], params: Dict) -> List[FastSignal]:
        """Filter signals by params (called PER TRIAL)"""
        return [s for s in signals if self._passes_filter(s, params)]

    def compute_sl_tp(self, signal: FastSignal, params: Dict, pip_size: float) -> tuple:
        """Compute SL and TP prices"""
        sl_pips = params.get('sl_pips', 50)
        tp_pips = params.get('tp_pips', 100)

        if signal.direction == 1:  # Long
            sl = signal.price - sl_pips * pip_size
            tp = signal.price + tp_pips * pip_size
        else:  # Short
            sl = signal.price + sl_pips * pip_size
            tp = signal.price - tp_pips * pip_size

        return sl, tp
```

### Step 2: Register Strategy

```python
# In strategies/rsi_fast.py, add:
from strategies.my_strategy import MyStrategyFast

FAST_STRATEGIES['my_strategy'] = MyStrategyFast
```

### Step 3: Run

```bash
python scripts/run_optimization.py --strategy my_strategy --staged --trials-per-stage 3000
```

---

## Data

### Available Data
| Pair | Timeframe | Candles | Period |
|------|-----------|---------|--------|
| GBP_USD | H1 | 12,389 | 2024-02-02 to 2026-01-30 |
| GBP_USD | M1 | 737,930 | 2024-02-02 to 2026-01-30 |
| EUR_USD | H1 | 12,388 | 2024-02-02 to 2026-01-30 |
| EUR_USD | M1 | 737,954 | 2024-02-04 to 2026-02-02 | **NEW**

### Download More Data
```bash
# Download 2 years of hourly data
python scripts/download_data.py --pair USD_JPY --granularity H1 --years 2

# Download minute data (large!)
python scripts/download_data.py --pair GBP_USD --granularity M1 --years 1
```

---

## Typical Workflow

### 1. First Time Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OANDA API credentials
python scripts/download_data.py --pair EUR_USD --granularity M1 --years 2
```

### 2. Run Optimization
```bash
# Quick test (verify setup works)
python scripts/run_optimization.py --strategy rsi_full --pair EUR_USD --timeframe M1 \
  --staged --trials-per-stage 500 --final-trials 500

# Production run
python scripts/run_optimization.py --strategy rsi_full --pair EUR_USD --timeframe M1 \
  --staged --trials-per-stage 3000 --final-trials 5000
```

### 3. Validate Results (IMPORTANT)
```bash
# Run validation backtest - metrics should MATCH optimization
python scripts/plot_equity.py \
  --results results/optimization/RSI_Divergence_Full_LATEST.json \
  --pair EUR_USD --timeframe M1 --use-best \
  --save results/equity_EUR_USD.png
```

**Validation checks:**
- Trade count should match (back + forward)
- Win rate should match (~69%)
- PF and Sharpe should be BETWEEN back and forward values
- If backtest looks BETTER than optimization → something is wrong (likely missing spread)

### 4. Analyze Results
- Results saved to `results/optimization/`
- JSON contains combined rank, back rank, forward rank
- Look for results with **low combined rank** (consistent in both periods)
- Check R² values - higher = smoother equity curve

### 5. Iterate
- If OnTester scores low: check R² and drawdown
- If too few trades: relax filters, widen signal criteria
- If optimization slow: reduce trials, use --quick mode first

---

## Troubleshooting

### "No valid results"
- Minimum trades not met (default 20)
- OnTester score is zero (negative profit or high drawdown)
- Try: lower `--min-trades` or relax signal parameters

### Low OnTester scores
- R² is low (choppy equity curve)
- High drawdown penalizing the score
- Few trades (√Trades factor)

### Optimization very slow
- Optuna TPE slows as it learns (~40/sec -> ~3/sec)
- This is normal, wait or reduce trials

### KeyError for parameter
- Parameter missing from strategy's `get_parameter_space()`
- Check all params are defined in all groups

### Memory error
- Too many signals pre-computed
- Now forward tests ALL results - may need more RAM
- Use H1 data instead of M1, or reduce date range

---

## Validation Checklist

Before going live with optimized parameters:

### Basic Checks
- [ ] Combined rank is low (balanced back/forward performance)
- [ ] Both back AND forward OnTester scores are positive
- [ ] R² is acceptable (>0.5 for both periods)
- [ ] Drawdown is within your risk tolerance
- [ ] Trade count is sufficient (not just lucky few trades)
- [ ] Tested on demo account before live

### Robustness Checks (NEW)
- [ ] Run `run_robust_optimization.py` on the pair
- [ ] Forward OnTester > 0 for multiple candidates
- [ ] Forward/Back ratio > 10% (ideally > 50%)
- [ ] Mean stability > 60%
- [ ] Core params (rsi_period, swing_strength) > 20% stable
- [ ] Rating is MODERATE or ROBUST (not FRAGILE/OVERFIT)

---

## Contact / History

- **Built:** 2026-02-01/02
- **Purpose:** Automated RSI divergence strategy optimization
- **Methodology:** MT5-style combined ranking with R² and OnTester score
- **State:** Testing - finding and fixing issues as we go

---

*Remember: Combined rank is what matters. A result consistent in BOTH back and forward beats one that's amazing in just one period.*
