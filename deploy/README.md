# Deployment System

Manages OANDA trading strategies as Windows services via NSSM on a VPS.

## How It Works

1. `strategies.json` defines which strategies run, their pairs, timeframes, and risk
2. `install_services.py` reads strategies.json and creates NSSM services for each
3. Each service runs `scripts/run_live.py` with the strategy's config
4. The dashboard service runs `scripts/run_dashboard.py` on port 8080

## File Layout

```
deploy/
  strategies.json      # Master config - what strategies to run
  install_services.py  # Creates/updates Windows services
  nssm/nssm.exe        # NSSM service manager binary
  update.bat           # One-click update: git pull + reinstall services
  setup.bat            # First-time setup (downloads NSSM, installs everything)

instances/
  <strategy_id>/
    config.json        # Strategy parameters + backtest expectations
    health.json        # Written by trader every heartbeat cycle
    state/
      position_state.json   # Current positions + daily P&L
      trade_history.json    # All closed trades
    logs/
      stdout.log       # NSSM-captured stdout
      stderr.log       # NSSM-captured stderr

config/
  live_<PAIR>_<TF>.json    # Exported params (copied to instance on first install)
```

## Add a New Strategy

1. Run the pipeline: `python scripts/run_pipeline.py --strategy rsi_v3 --pair EUR_USD --timeframe H1`
2. Export params: `python scripts/export_params.py --run <RUN_DIR> --output config/live_EUR_USD_H1.json`
3. Add entry to `deploy/strategies.json`:
   ```json
   {
     "id": "rsi_v3_EUR_USD_H1",
     "strategy": "rsi_v3",
     "pair": "EUR_USD",
     "timeframe": "H1",
     "risk_pct": 1.0,
     "enabled": true,
     "from_run": "<RUN_DIR>",
     "description": "V3 RSI H1 - XX.X GREEN"
   }
   ```
4. Commit and push, then on VPS: double-click `update.bat`

## What update.bat Does

1. `git pull` to get latest code + strategies.json
2. Runs `install_services.py` which for each strategy:
   - Stops existing service
   - Removes it (with wait for Windows to release the handle)
   - Reinstalls with current config
   - Starts the new service
3. Reinstalls the dashboard service

## Disable a Strategy

Set `"enabled": false` in strategies.json. On next update, the service will be stopped and removed.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Marked for deletion" on update | Fixed in install_services.py with `wait_for_removal()`. If still happens, wait 30s and retry. |
| Dashboard shows 0/N running | Check timeframe — H1 heartbeat threshold is 90min, not 5min. |
| Service won't start | Check `instances/<id>/logs/stderr.log` |
| Dashboard 500 error | Check `instances/_dashboard/logs/stderr.log` |
| NSSM not found | Run `setup.bat` as admin, or manually download nssm.exe to `deploy/nssm/` |

## Service Names

Services are named `OandaTrader_<strategy_id>` (e.g., `OandaTrader_rsi_v3_GBP_USD_M15`).
Dashboard is `OandaTrader_Dashboard`.

Use `nssm status <service_name>` to check, or view all in Windows Services (services.msc).
