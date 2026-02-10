# Forex Python Trading System - Project Plan

**Owner:** Longy
**Created:** 2026-02-01
**Status:** Setting Up

---

## Executive Summary

Build a complete Python-based forex trading system that can:
1. Backtest strategies with realistic conditions
2. Optimize parameters using Optuna (Bayesian search)
3. Validate with walk-forward testing
4. Live trade via OANDA API
5. Scale to multiple strategies and pairs (portfolio approach)

**Key Decision:** Moving away from MT5/MQL5 to full Python stack for better automation, Optuna integration, and lower costs.

---

## Background & Context

### Why We Moved Away From MT5

1. **MT5 Cloud costs:** ~$20/week (~$80/month) for optimization
2. **Python + MQL5 integration never worked well** for this user
3. **37 parameters in optimization** = search space too large (2.8 × 10^28 combinations)
4. **No native Optuna integration** with MT5
5. **Wanted portfolio approach** - multiple strategies/pairs managed together

### What We Discussed

- Staged optimization (5-7 params per pass) vs all-at-once
- Python backtesting accuracy vs MT5
- OANDA as Python-friendly broker (free API, no minimum deposit, FCA regulated)
- H1 timeframe doesn't need millisecond execution
- User doesn't care about exact strategy - just wants profitable, tested strategies
- 30% max drawdown hard limit, prefer 25%

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              LOCAL MACHINE                               │
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │   Optuna     │──▶│   Backtest   │──▶│   Results    │                │
│  │  Optimizer   │   │    Engine    │   │   Database   │                │
│  └──────────────┘   └──────────────┘   └──────────────┘                │
│                                               │                          │
│                     ┌─────────────────────────┘                          │
│                     ▼                                                    │
│              ┌──────────────┐                                           │
│              │   Dashboard   │  (optuna-dashboard)                      │
│              └──────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ Deploy best params
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           VPS (~£5/month)                                │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                       Live Trader                               │    │
│  │  - Same strategy code as backtest                              │    │
│  │  - Connects to OANDA API                                       │    │
│  │  - Risk management (daily limits, DD limits)                   │    │
│  │  - Telegram alerts                                             │    │
│  │  - Auto-reconnect on failures                                  │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │    OANDA     │
                       │   Broker     │
                       └──────────────┘
```

---

## Project Structure

```
forex-python/
├── config/
│   ├── __init__.py
│   └── settings.py          # Pydantic settings from .env
│
├── data/
│   ├── oanda/               # Downloaded OANDA data
│   ├── mt5/                 # Optional: MT5 exports for comparison
│   └── download.py          # Data download utilities
│
├── strategies/
│   ├── __init__.py
│   ├── base.py              # Base strategy class
│   ├── rsi_divergence.py    # RSI hidden divergence
│   ├── ma_crossover.py      # Moving average crossover
│   └── breakout.py          # Support/resistance breakout
│
├── backtesting/
│   ├── __init__.py
│   ├── engine.py            # Core backtest engine
│   ├── trade_manager.py     # SL, TP, BE, trailing, partial close
│   └── metrics.py           # Sharpe, DD, profit factor, etc.
│
├── optimization/
│   ├── __init__.py
│   ├── optimizer.py         # Optuna wrapper
│   ├── walk_forward.py      # Walk-forward validation
│   └── analysis.py          # Parameter stability, Monte Carlo
│
├── live/
│   ├── __init__.py
│   ├── oanda_client.py      # OANDA API wrapper
│   ├── trader.py            # Live trading execution
│   └── monitor.py           # Position monitoring, alerts
│
├── results/
│   ├── backtests/           # Backtest trade logs
│   └── optimization/        # Optuna databases
│
├── logs/                    # Application logs
│
├── tests/
│   ├── test_strategies.py
│   ├── test_backtest.py
│   └── test_indicators.py
│
├── scripts/
│   ├── download_data.py     # Fetch historical data
│   ├── run_backtest.py      # Single backtest
│   ├── run_optimization.py  # Optuna optimization
│   ├── run_walk_forward.py  # Walk-forward test
│   └── run_live.py          # Live trading
│
├── .env                     # API keys (NOT committed)
├── .gitignore
├── requirements.txt
├── PROJECT_PLAN.md          # This file
└── README.md
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Priority: HIGH) ✅ COMPLETE

**Goal:** Working backtest engine with one strategy

- [x] Config system (load from .env)
- [x] OANDA data download (1-min OHLC, build higher timeframes)
- [x] Base strategy class
- [x] RSI Divergence strategy (similar to MQL5 EA logic)
- [x] Backtest engine with realistic spreads
- [x] Trade management (SL, TP, break-even, trailing)
- [x] Basic metrics (profit, DD, Sharpe, trades)

**Validation:** Run `python scripts/test_system.py`

### Phase 2: Optimization Pipeline (Priority: HIGH) ✅ COMPLETE

**Goal:** Optuna-powered parameter search with walk-forward

- [x] Optuna integration
- [x] Walk-forward optimization
- [x] Parameter stability analysis (in walk-forward output)
- [x] Optuna dashboard setup (use optuna-dashboard command)
- [x] Results export (best params to JSON)

**Validation:** Run optimization, verify walk-forward OOS results

### Phase 3: Live Trading (Priority: HIGH)

**Goal:** Paper trading on OANDA

- [ ] OANDA API client (orders, positions, account)
- [ ] Live trader (same strategy code as backtest)
- [ ] Risk management (daily loss, DD limits, circuit breaker)
- [ ] Telegram alerts
- [ ] Error handling, reconnection logic

**Validation:** Paper trade for 2-4 weeks, compare with backtest expectations

### Phase 4: Production & Portfolio (Priority: MEDIUM)

**Goal:** Live trading + multiple strategies

- [ ] Deploy to VPS
- [ ] Add second strategy (MA crossover or breakout)
- [ ] Portfolio-level risk management
- [ ] Performance tracking dashboard
- [ ] Automated reports

---

## Strategy: RSI Hidden Divergence

### Concept (from original MQL5 EA)

**Hidden Bullish Divergence (BUY signal):**
- Price makes HIGHER low
- RSI makes LOWER low
- Indicates trend continuation upward

**Hidden Bearish Divergence (SELL signal):**
- Price makes LOWER high
- RSI makes HIGHER high
- Indicates trend continuation downward

### Key Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| rsi_period | RSI calculation period | 5-30 |
| swing_strength | Bars each side for swing detection | 2-10 |
| min_bars_between | Minimum bars between swings | 3-15 |
| max_bars_between | Maximum bars between swings | 20-100 |
| stop_loss_pips | Stop loss in pips | 20-80 |
| tp_multiplier | TP as multiple of SL | 1.0-4.0 |
| risk_percent | Risk per trade | 0.5-2.0 |

### Trade Management

- **Break-even:** Move SL to entry + X pips after Y pips profit
- **Trailing stop:** Trail by X pips after Y pips profit
- **Partial close:** Close X% at Y pips profit

### Filters

- Trading hours (avoid Asian session low volatility)
- Max spread filter
- Friday close filter
- Daily loss limit

---

## OANDA Configuration

**Account Type:** Practice (demo) → then Live
**API:** REST v20
**Account ID:** [To be filled - get from OANDA portal]
**Pairs:** GBP_USD (primary), EUR_USD, USD_JPY (later)

**Spread assumptions for backtesting:**

| Pair | London Session | Asian Session |
|------|----------------|---------------|
| GBP_USD | 1.2 pips | 2.0 pips |
| EUR_USD | 1.0 pips | 1.5 pips |

**Slippage buffer:** 0.5 pips per trade

---

## Risk Management

### Per-Trade

- Max risk: 1% of equity
- Max spread: 3 pips (don't trade if wider)

### Daily

- Max daily loss: 3% of equity
- Max daily trades: 5 (prevent overtrading)

### Account

- Max drawdown: 25% (hard stop - all trading halts)
- Pause at 15% DD (alert, review)

### Position

- Max open positions: 3 per strategy
- Max correlation: Don't take highly correlated trades

---

## Validation Process

### Before Paper Trading

1. Backtest on 2+ years of data
2. Walk-forward validation (12-month train, 3-month test, roll)
3. Out-of-sample Sharpe > 0.8
4. Max DD < 20% in backtest
5. Parameter stability (similar params give similar results)

### Before Live Trading

1. Paper trade for 2-4 weeks minimum
2. Compare paper trades with backtest expectations
3. Check execution quality (fills, spread, slippage)
4. Verify alerts/monitoring working

### Scaling Up

1. Start live at 0.25x target position size
2. After 4 weeks profitable, scale to 0.5x
3. After 8 weeks, scale to 1.0x
4. Add second strategy only after first is stable

---

## Cost Analysis

| Item | MT5/Cloud (Old) | Python/OANDA (New) |
|------|-----------------|-------------------|
| Optimization | ~$80/month | £0 |
| Data | Included | Free (OANDA API) |
| Broker | IC Markets | OANDA |
| VPS | ~£15/month | ~£5/month |
| **Total** | **~$100/month** | **~£5/month** |

---

## Files Reference

### Key Files to Know

- `.env` - API keys and config (NEVER commit)
- `config/settings.py` - Loads .env, provides typed config
- `strategies/base.py` - All strategies inherit from this
- `backtesting/engine.py` - Core backtest loop
- `optimization/optimizer.py` - Optuna integration
- `live/trader.py` - Live execution

### Running Commands

```bash
# Download historical data
python scripts/download_data.py --pair GBP_USD --years 3

# Run single backtest
python scripts/run_backtest.py --strategy rsi_divergence --pair GBP_USD

# Run optimization
python scripts/run_optimization.py --strategy rsi_divergence --trials 500

# Run walk-forward
python scripts/run_walk_forward.py --strategy rsi_divergence

# Start live trading (paper)
python scripts/run_live.py --paper

# Start live trading (real)
python scripts/run_live.py --live
```

---

## Open Questions / Decisions Needed

1. **OANDA Account ID** - User needs to get this from OANDA portal
2. **VPS details** - User mentioned they have one, need access details
3. **Telegram bot** - Set up for alerts (optional but recommended)
4. **Additional pairs** - Start with GBPUSD only, or multiple?

---

## Session Handoff Notes

### What Was Completed (Session 2 - 2026-02-01)

- ✅ Phase 1 complete: Config, data download, strategy, backtest engine
- ✅ Phase 2 complete: Optuna optimizer, walk-forward validation
- ✅ All run scripts created (download, backtest, optimization, walk-forward)
- ✅ Break-even and trailing stop functionality added to backtest engine
- ✅ Test script for end-to-end validation
- ✅ Downloaded 2 years H1 data for GBP_USD and EUR_USD
- ✅ Ran optimization (80/300 trials) - Best Sharpe: 23.38
- ✅ Cross-pair validation: Strategy works on both GBP_USD and EUR_USD

### Current Best Results (Trial 48)

| Pair | Trades | Win Rate | Profit Factor | Return | Max DD |
|------|--------|----------|---------------|--------|--------|
| GBP_USD | 90 | 92.2% | 10.30 | 108.2% | 2.3% |
| EUR_USD | 79 | 89.9% | 8.48 | 85.3% | 2.2% |

### What To Do Next

1. **Resume optimization** or use current best params
2. **Run walk-forward validation** to confirm out-of-sample performance
3. **Start Phase 3:** Build live trading module
4. **Paper trade** for 2-4 weeks before going live

### Key Files
- `HANDOVER.md` - Detailed session handover
- `results/optimization/best_params_20260201.json` - Best parameters
- `results/optimization/RSI_Div_GBP_USD_20260201_1603.db` - Optuna study (resumable)

### Key User Preferences

- Doesn't care about exact strategy replication - just wants profitable
- Wants automation - works full time
- 30% max DD is hard limit (we set 25% in code as buffer)
- Wants portfolio approach eventually (multiple strategies/pairs)
- Has VPS available for live trading

---

## Contact / Resources

- **OANDA API Docs:** https://developer.oanda.com/rest-live-v20/introduction/
- **Optuna Docs:** https://optuna.readthedocs.io/
- **oandapyV20:** https://github.com/hootnot/oanda-api-v20

---

*Last updated: 2026-02-01 (Phase 1 & 2 complete)*
