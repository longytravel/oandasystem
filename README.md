# OANDA Trading System

Python-based forex trading system with backtesting, optimization (Optuna), and live trading via OANDA API.

## Features

- **Full Python Stack** - Same code for backtest and live trading
- **Optuna Optimization** - Bayesian parameter search
- **Walk-Forward Testing** - Proper out-of-sample validation
- **OANDA Integration** - Free API, no minimum deposit
- **Low Cost** - No cloud optimization fees

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and edit config
cp .env.example .env
# Add your OANDA API key and account ID

# Test the system
python scripts/test_system.py

# Run a backtest
python scripts/run_backtest.py --strategy rsi_divergence --pair GBP_USD
```

## Project Structure

```
oandasystem/
├── config/          # Settings and configuration
├── data/            # Historical data storage
├── strategies/      # Trading strategy implementations
├── backtesting/     # Backtest engine
├── optimization/    # Optuna integration (TODO)
├── live/            # OANDA API client and live trading
├── scripts/         # Entry point scripts
└── tests/           # Unit tests
```

## Documentation

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for detailed architecture and implementation plan.

## Status

- [x] Project structure
- [x] OANDA API client
- [x] Data download
- [x] RSI Divergence strategy
- [x] Backtest engine
- [ ] Optuna optimization
- [ ] Walk-forward testing
- [ ] Live trading
- [ ] Telegram alerts

## License

Private - not for distribution.
