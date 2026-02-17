# Live Trading Guide

**Last Updated:** 2026-02-17

This guide covers deploying and running the OANDA trading system in paper or live mode, including VPS setup and monitoring.

---

## Active Deployments

### Original H1 Strategies (5)

| # | Service ID | Strategy | Pair | TF | Risk | Score | Pipeline Run | Since |
|---|------------|----------|------|----|------|-------|--------------|-------|
| 1 | rsi_v3_GBP_USD_M15 | RSI Divergence V3 | GBP_USD | M15 | 1.0% | 72.8 GREEN | GBP_USD_M15_20260213_154233 | 2026-02-13 |
| 2 | rsi_v3_GBP_USD_H1 | RSI Divergence V3 | GBP_USD | H1 | 0.5% | 89.8 GREEN | GBP_USD_H1_20260206_151217 | 2026-02-06 |
| 3 | rsi_v1_USD_CHF_H1 | RSI Divergence V1 | USD_CHF | H1 | 1.0% | 97.4 GREEN | USD_CHF_H1_20260212_122528 | 2026-02-12 |
| 4 | fpma_EUR_JPY_H1 | Fair Price MA | EUR_JPY | H1 | 1.0% | 87.6 GREEN | EUR_JPY_H1_20260215_074906 | 2026-02-15 |
| 5 | fpma_EUR_AUD_H1 | Fair Price MA | EUR_AUD | H1 | 1.0% | 92.9 GREEN | EUR_AUD_H1_20260215_140708 | 2026-02-15 |

### Backtester Validation Fleet - M15 (12 strategies, added 2026-02-17)

Purpose: Generate trades across 7 different strategy types to compare backtest predictions vs actual OANDA fills. Not optimized for profit - optimized for trade volume and strategy diversity.

| # | Instance ID | Strategy | Pair | Source | Score |
|---|-------------|----------|------|--------|-------|
| 6 | rv3_EUR_USD_M15 | RSI V3 | EUR_USD | Pipeline optimized | 73.5 GREEN |
| 7 | rv3_EUR_JPY_M15 | RSI V3 | EUR_JPY | Pipeline optimized | 75.1 GREEN |
| 8 | rv3_USD_JPY_M15 | RSI V3 | USD_JPY | Pipeline optimized | 82.9 GREEN |
| 9 | ev6_USD_JPY_M15 | EMA V6 | USD_JPY | Pipeline optimized | 70.2 GREEN |
| 10 | dch_USD_JPY_M15 | Donchian Breakout | USD_JPY | Pipeline optimized | 82.0 GREEN |
| 11 | dch_NZD_USD_M15 | Donchian Breakout | NZD_USD | Default params | N/A |
| 12 | bbsq_EUR_JPY_M15 | Bollinger Squeeze | EUR_JPY | Pipeline optimized | 93.0 GREEN |
| 13 | bbsq_GBP_USD_M15 | Bollinger Squeeze | GBP_USD | Default params | N/A |
| 14 | ldn_USD_JPY_M15 | London Breakout | USD_JPY | Pipeline optimized | 85.3 GREEN |
| 15 | ldn_EUR_JPY_M15 | London Breakout | EUR_JPY | Pipeline optimized | 75.7 GREEN |
| 16 | stadx_GBP_JPY_M15 | Stochastic ADX | GBP_JPY | Pipeline optimized | 70.4 GREEN |
| 17 | stadx_USD_CHF_M15 | Stochastic ADX | USD_CHF | Default params | N/A |

**Total: 17 strategies** (~5-7 GB RAM estimated)

**VPS**: 104.128.63.239:5909 (VNC) | **Dashboard**: http://104.128.63.239:8080 | **Account**: 101-004-38418172-001 (practice)

### Fair Price MA (EUR_JPY H1) - Deployed 2026-02-15

Mean-reversion grid strategy. Fast EMA(200) + Slow EMA(400) detect trend. Entry on pullback 150 pips from fast EMA, grid of 5 orders across 50 pip range. SL fixed 75 pips, TP 1.5:1 R:R. Trailing at 50 pips (15 pip step). Break-even enabled. Session filter 06:00-22:00. Max hold 50 bars.

| Metric | Forward Test | Backtest |
|--------|-------------|----------|
| Win Rate | 83.3% | 56.7% |
| Profit Factor | 7.38 | 1.71 |
| Sharpe | 7.08 | 2.72 |
| Max Drawdown | 3.1% | 12.1% |
| Trades | 18 (6mo) | 187 |
| Return | 19.8% | 90.9% |

Confidence: WF 100%, Stability 92.3%, MC 100%, F/B ratio 4.72

### Fair Price MA (EUR_AUD H1) - Deployed 2026-02-15

Mean-reversion grid strategy. Fast EMA(50) + Slow EMA(400) detect trend. Entry on pullback 150 pips from fast EMA, grid of 10 orders across 75 pip range. SL ATR 2.0x, TP 1.5:1 R:R. Trailing at 30 pips (8 pip step). No break-even, no time filter, no max hold.

| Metric | Forward Test | Backtest |
|--------|-------------|----------|
| Win Rate | 81.8% | - |
| Profit Factor | 6.35 | - |
| Sharpe | 4.03 | - |
| Max Drawdown | 2.1% | 17.8% |
| Trades/Month | 1.8 | - |
| F/B Ratio | 2.06 | - |

Confidence: WF 91.4% (1 OOS window), Stability 100%, MC 91.7%, F/B ratio 2.06

### Slippage Model Warning (2026-02-16)

An exit slippage model (`slippage_pips=0.5`) was added to all backtest functions on 2026-02-16 (see KI-22 in [KNOWN_ISSUES.md](KNOWN_ISSUES.md)). Live SL orders (stop->market) slip ~0.5 pips, which the backtest previously ignored. **All strategies above were optimized before this fix and should be re-optimized with the slippage model enabled.**

**M15 most affected:** The rsi_v3 GBP_USD M15 strategy (#1) used tight break-even triggers (ATR x 0.3 ~ 1.4 pip). With `slippage_pips=0.5`, backtest shows WR dropping from 90% to 54% and Return from +32% to -6.3%. This strategy needs re-optimization with slippage before continued live use.

H1 strategies are less affected (larger pip movements relative to slippage) but should still be re-validated.

---

## Prerequisites

1. **OANDA Account** - Practice (demo) or live account at [oanda.com](https://www.oanda.com)
2. **API Key** - Generate at OANDA Hub > Manage API Access
3. **Python 3.10+** with pip
4. **Pipeline run** - A completed GREEN pipeline run with optimized parameters

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure OANDA credentials
cp .env.example .env
# Edit .env:
#   OANDA_API_KEY=your-api-key-here
#   OANDA_ACCOUNT_TYPE=practice
#   OANDA_ACCOUNT_ID=your-account-id

# 3. Test connection (single iteration, no trades)
python scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_M15_20260210_063223 --dry-run --once

# 4. Run paper trading (continuous loop)
python scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_M15_20260210_063223

# 5. Or with explicit pair/timeframe
python scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_M15_20260210_063223 --pair GBP_USD --timeframe M15
```

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `rsi_v3` | Strategy key: `rsi_v3`, `rsi_v4`, `rsi_v5`, `ema_cross`, `fair_price_ma`, `donchian_breakout`, `bollinger_squeeze`, `london_breakout`, `stochastic_adx` |
| `--pair` | `GBP_USD` | Currency pair (OANDA format) |
| `--timeframe` | `H1` | Candle timeframe: M1, M5, M15, M30, H1, H2, H4, D |
| `--from-run` | - | Load params from pipeline run ID (e.g., `GBP_USD_M15_20260210_063223`) |
| `--params-file` | - | Load params from JSON file |
| `--dry-run` | off | Log signals without placing trades |
| `--once` | off | Run single iteration and exit (for testing) |
| `--risk` | `1.0` | Risk per trade as % of equity |
| `--candles` | `500` | Number of candles to fetch for signal generation |
| `--instance-id` | - | Unique ID for multi-instance (e.g., `rsi_v3_GBP_USD_M15`) |
| `--instance-dir` | - | Base directory for instance state/logs |
| `--verbose` / `-v` | off | Enable DEBUG logging |
| `--status` | off | Print status and exit |
| `--yes` / `-y` | off | Skip live trading confirmation prompt |

---

## VPS Setup (Windows Server)

### 1. Initial Setup

```powershell
# Install Python 3.10+
# Download from python.org, check "Add to PATH"

# Clone repository
git clone <repo-url> C:\Trading\oandasystem
cd C:\Trading\oandasystem

# Install dependencies (no numba needed for live trading)
pip install -r requirements.txt
```

### 2. Configure Credentials

`.env` is in `.gitignore` and won't be pulled from git. Create it manually:

```powershell
# Create .env file
notepad C:\Trading\oandasystem\.env
```

Contents:
```
OANDA_API_KEY=your-api-key-here
OANDA_ACCOUNT_TYPE=practice
OANDA_ACCOUNT_ID=xxx-xxx-xxxxxxx-xxx
```

### 3. Numba Note

The live trading system does **not** require numba. Optimization imports are lazy-loaded so the live trading modules work without numba installed. If you see numba import errors, ensure you're using the latest code (`git pull`). See KI-20 in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

### 4. Test Connection

```powershell
python scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_M15_20260210_063223 --dry-run --once
```

Expected output: connection to OANDA, fetch 500 candles, check for signals, exit.

### 5. Run as Background Process

```powershell
# Option A: Start in background with nohup-style
Start-Process python -ArgumentList "scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_M15_20260210_063223 --instance-id rsi_v3_GBP_USD_M15 --instance-dir instances/rsi_v3_GBP_USD_M15 -y" -WindowStyle Hidden

# Option B: Use Task Scheduler for auto-restart on reboot
# Create a scheduled task that runs at startup
```

---

## Running the Trader

### Mode: Dry Run (Signal Logging Only)
```bash
python scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_M15_20260210_063223 --dry-run
```
Logs all signals to console and file but places no trades. Use for initial validation.

### Mode: Paper Trading (Practice Account)
```bash
python scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_M15_20260210_063223
```
Places real orders on OANDA practice account. No real money at risk.

### Mode: Live Trading
```bash
# Change .env: OANDA_ACCOUNT_TYPE=live (with live API key)
python scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_M15_20260210_063223
# Type "LIVE" at confirmation prompt
```

---

## Monitoring

### Health Heartbeat

When `--instance-id` and `--instance-dir` are provided, the trader writes a `health.json` file every candle:

```json
{
  "instance_id": "rsi_v3_GBP_USD_M15",
  "timestamp": "2026-02-10T12:15:00+00:00",
  "status": "running",
  "positions": 1,
  "last_candle": "2026-02-10 12:15:00",
  "errors": 0
}
```

### Monitor Script

```bash
# Check all instances once
python scripts/run_monitor.py --instances-dir instances/ --once

# Continuous monitoring (every 60s)
python scripts/run_monitor.py --instances-dir instances/

# With Telegram alerts
python scripts/run_monitor.py --instances-dir instances/ --telegram

# Print summary
python scripts/run_monitor.py --instances-dir instances/ --summary
```

Alert thresholds:
- **Stale heartbeat:** >300s (5 min) without update = instance likely down
- **Error count:** >3 consecutive errors = alert
- **Status "error":** immediate alert

### Log Files

Logs are written to `logs/` (or `instances/<id>/logs/` with `--instance-dir`):
- `live_trading_YYYY-MM-DD.log` - Daily rotation, 30-day retention
- Contains all signals, trades, risk checks, and errors

---

## Risk Controls

The `RiskManager` performs 7 checks before every trade:

| Check | Default | .env Key | Description |
|-------|---------|----------|-------------|
| Circuit breaker | - | - | If previously tripped, blocks all trading |
| Daily trade limit | 5 | `MAX_DAILY_TRADES` | Max trades per day |
| Daily loss limit | 3% | `MAX_DAILY_LOSS_PCT` | Max daily loss as % of equity |
| Drawdown circuit breaker | 25% | `MAX_DRAWDOWN_PCT` | Hard stop - halts all trading |
| Drawdown warning | 15% | `PAUSE_DRAWDOWN_PCT` | Logs warning, still allows trading |
| Position limit | 3 | `MAX_OPEN_POSITIONS` | Max concurrent open positions |
| Spread check | 3.0 pip | `MAX_SPREAD_PIPS` | Rejects trade if spread too wide |

**Position sizing:** Based on risk percentage. Default 1% of equity per trade (`MAX_RISK_PER_TRADE`). Calculated as: `units = (balance * risk_pct) / (sl_pips * pip_value)`.

---

## Trade Management

The `PipelineStrategyAdapter` manages open positions on each candle:

### Breakeven
- **Trigger:** Profit reaches `be_atr_mult` x ATR in pips
- **Action:** Move SL to entry price + `be_offset_pips`
- **M15 params:** 0.3x ATR trigger, +5 pip offset

### Trailing Stop
- **Trigger:** Profit reaches `trail_start_pips`
- **Action:** Move SL to `current_price - trail_step_pips` (for longs)
- **M15 params:** 50 pip start, 8 pip step
- **Note:** Independent of breakeven (not chained)

---

## Known Limitations

- **No news filter** - Trades during high-impact news events
- **No session time filter** - Trades 24/5 (no London/NY session filtering)
- **Telegram not wired in** - `TelegramAlerts` class exists but isn't called by the trader
- **Single pair per instance** - Run multiple instances for multiple pairs
- **No auto-reconnection** - If OANDA API drops, error count increments but no explicit reconnect logic
- **UTC only** - All times are UTC internally
- **No partial close in live** - Pipeline supports partial close but adapter doesn't implement it

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `401 Unauthorized` | Invalid API key or wrong account type | Check `.env`: `OANDA_API_KEY`, `OANDA_ACCOUNT_TYPE` |
| `ModuleNotFoundError: numba` | Numba import chain triggered | Pull latest code - lazy imports fix this (KI-20) |
| Spread too wide | Spread > `MAX_SPREAD_PIPS` | Normal during low liquidity (Asian session, weekends). Will retry next candle |
| No signals for hours | Normal for RSI divergence | M15 generates ~2 signals/day average, H1 even fewer. Check logs for "No signal on this candle" |
| `INSUFFICIENT_MARGIN` | Position size exceeds margin | Reduce `--risk` percentage or fund account |
| Health.json not updating | Instance crashed | Check logs, restart process |
| `FileNotFoundError: state.json` | Wrong `--from-run` ID | Verify run directory exists in `results/pipelines/` |

---

## Moving to Live Trading

Checklist before switching from practice to live:

- [ ] Paper traded for at least 2 weeks with consistent results
- [ ] Reviewed all trades manually - signals make sense
- [ ] Risk controls tested (daily limit, drawdown, spread)
- [ ] Monitoring in place (health.json, log review)
- [ ] Emergency shutdown procedure practiced (Ctrl+C or kill process)
- [ ] Starting with minimum position size (reduce `--risk` to 0.5% or less)
- [ ] `.env` updated: `OANDA_ACCOUNT_TYPE=live` with live API key
- [ ] Confirmation prompt shows "LIVE TRADING MODE" warning

---

## Multi-Instance Setup

Run multiple strategies or pairs simultaneously:

```bash
# Instance 1: V3 RSI on GBP_USD M15
python scripts/run_live.py \
  --strategy rsi_v3 \
  --from-run GBP_USD_M15_20260210_063223 \
  --instance-id rsi_v3_GBP_USD_M15 \
  --instance-dir instances/rsi_v3_GBP_USD_M15 \
  -y

# Instance 2: V3 RSI on GBP_USD H1
python scripts/run_live.py \
  --strategy rsi_v3 \
  --from-run GBP_USD_H1_20260206_151217 \
  --instance-id rsi_v3_GBP_USD_H1 \
  --instance-dir instances/rsi_v3_GBP_USD_H1 \
  -y
```

Each instance gets:
- Isolated state directory (`instances/<id>/state/`)
- Isolated logs (`instances/<id>/logs/`)
- Isolated health file (`instances/<id>/health.json`)

Monitor all instances:
```bash
python scripts/run_monitor.py --instances-dir instances/
```

---

## Trade Comparison Workflow (Backtest vs Live)

The primary purpose of the M15 validation fleet is to verify the backtester produces results that match real OANDA execution. Here's the workflow:

### Step 1: Collect Live Trades

After strategies have been running for a few days/weeks, export trade history from OANDA:

1. Log into [OANDA Trade Practice](https://trade.oanda.com/practice)
2. Go to **Portfolio** → **History** → **Trades**
3. Filter by date range covering the deployment period
4. Export as CSV (or copy/paste into a CSV file)

The CSV should contain at minimum: trade ID, instrument, direction (buy/sell), open time, close time, open price, close price, units, P&L.

### Step 2: Run the Check-Trades Skill

Share the CSV with Claude and use the `/check-trades` skill:

```
/check-trades
```

This will:
1. **Download fresh candles** from OANDA for the exact period the trades cover
2. **Run a full backtest simulation** using the same strategy + params that the live instance uses
3. **Match live trades to backtest trades** by timestamp and direction
4. **Compare key metrics** for each matched pair:
   - Entry price (live vs backtest)
   - Exit price and reason (SL/TP/trailing/time)
   - P&L per trade
   - Signal timing (same candle? off by 1?)
5. **Flag discrepancies** - any trade that doesn't appear in both live and backtest, or where prices diverge beyond spread+slippage tolerance

### Step 3: Interpret Results

What we're looking for:

| Check | Pass | Fail |
|-------|------|------|
| Signal match | Every live entry has a corresponding backtest signal on the same candle | Trades appear in live but not backtest (or vice versa) |
| Entry price | Within spread (1.5 pips) of backtest entry | Consistently worse than backtest by > 2 pips |
| SL/TP hit | Same exit reason (SL vs TP) | Backtest says TP but live hit SL (or vice versa) |
| P&L direction | Same sign (both profit or both loss) | Opposite signs on matched trades |
| Timing | Entries within same candle | Off by multiple candles |

**Common issues and what they mean:**
- **Live has extra trades backtest doesn't**: Signal generation bug, or live is processing candles at different boundaries
- **Backtest has trades live doesn't**: Risk manager blocked them (spread too wide, position limit, daily limit)
- **Same signal, different exit**: Trailing stop or breakeven logic differs between backtest and live adapter
- **Consistent slippage bias**: Cost model needs adjustment (spread_pips or slippage_pips too low)

### Step 4: Iterate

Fix any discrepancies found, redeploy, and repeat. The goal is that the backtester and live system produce statistically indistinguishable results, giving us confidence that pipeline scores predict real-world performance.

### Quick Reference

```
# What to share with Claude:
1. The OANDA trade CSV
2. Which instance(s) the trades are from (e.g., "rv3_EUR_USD_M15")
3. The date range

# Claude will handle the rest using /check-trades
```

---

## NSSM Service Management (VPS)

The VPS uses NSSM (Non-Sucking Service Manager) to run strategies as Windows services.

### Deploy/Update All Services

```powershell
# On VPS: pull latest code and reinstall services
cd C:\Trading\oandasystem
git pull
python deploy/install_services.py
```

This reads `deploy/strategies.json` and creates/updates NSSM services for each enabled strategy. Services auto-start on boot.

### strategies.json Format

```json
{
  "strategies": [
    {
      "instance_id": "rv3_EUR_USD_M15",
      "strategy": "rsi_v3",
      "pair": "EUR_USD",
      "timeframe": "M15",
      "from_run": "EUR_USD_M15_20260216_183241",
      "risk": 1.0,
      "enabled": true
    }
  ]
}
```

Key fields:
- `instance_id`: Unique ID, used as NSSM service name and instance directory name
- `from_run`: Pipeline run ID to load optimized params from (omit for default params)
- `enabled`: Set `false` to skip without deleting the entry
- Services without `from_run` load params from `instances/{instance_id}/config.json`

### Service Commands

```powershell
# List all OANDA services
nssm list | findstr oanda

# Check status of one service
nssm status oanda_rv3_EUR_USD_M15

# Start/stop/restart
nssm start oanda_rv3_EUR_USD_M15
nssm stop oanda_rv3_EUR_USD_M15
nssm restart oanda_rv3_EUR_USD_M15

# View service config
nssm edit oanda_rv3_EUR_USD_M15

# Remove a service
nssm remove oanda_rv3_EUR_USD_M15 confirm
```
