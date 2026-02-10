#!/usr/bin/env python
"""
Run a single backtest with specified parameters.

Usage:
    python scripts/run_backtest.py --pair GBP_USD --timeframe H1
    python scripts/run_backtest.py --pair GBP_USD --months 6 --params-file results/best_params.json
"""
import sys
from pathlib import Path
import argparse
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from loguru import logger

from strategies.rsi_divergence import RSIDivergenceStrategy
from backtesting.engine import BacktestEngine
from live.oanda_client import OandaClient
from data.download import load_data
from config.settings import settings


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--pair", default="GBP_USD", help="Currency pair")
    parser.add_argument("--timeframe", default="H1", help="Timeframe")
    parser.add_argument("--months", type=int, default=12, help="Months of data")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade (%)")
    parser.add_argument("--spread", type=float, default=1.5, help="Spread in pips")
    parser.add_argument("--params-file", help="JSON file with strategy parameters")
    parser.add_argument("--use-break-even", action="store_true", help="Enable break-even")
    parser.add_argument("--use-trailing", action="store_true", help="Enable trailing stop")
    parser.add_argument("--download", action="store_true", help="Download fresh data")

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    print(f"\n{'#' * 60}")
    print(f"# BACKTEST: {args.pair} {args.timeframe}")
    print(f"{'#' * 60}")

    # Get data
    if args.download:
        logger.info("Downloading fresh data from OANDA...")
        client = OandaClient()
        accounts = client.get_accounts()
        client.account_id = accounts[0]['id']

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30 * args.months)

        df = client.get_candles_range(
            instrument=args.pair,
            granularity=args.timeframe,
            from_time=start_time,
            to_time=end_time
        )
    else:
        # Try to load from file
        try:
            df = load_data(args.pair, args.timeframe)
        except FileNotFoundError:
            logger.info("No local data found, downloading from OANDA...")
            client = OandaClient()
            accounts = client.get_accounts()
            client.account_id = accounts[0]['id']

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30 * args.months)

            df = client.get_candles_range(
                instrument=args.pair,
                granularity=args.timeframe,
                from_time=start_time,
                to_time=end_time
            )

    logger.info(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Load params from file if specified
    params = None
    if args.params_file:
        with open(args.params_file) as f:
            data = json.load(f)
            params = data.get("best_params", data)
        logger.info(f"Loaded params from {args.params_file}")

    # Create strategy
    strategy = RSIDivergenceStrategy(params)
    logger.info(f"Strategy: {strategy.name}")
    logger.info(f"Params: {strategy.params}")

    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=args.capital,
        spread_pips=args.spread,
        slippage_pips=0.5,
        use_break_even=args.use_break_even,
        break_even_pips=30.0,
        break_even_offset=2.0,
        use_trailing_stop=args.use_trailing,
        trailing_start_pips=40.0,
        trailing_step_pips=10.0,
    )

    # Run backtest
    logger.info("Running backtest...")
    result = engine.run(strategy, df, risk_per_trade=args.risk)

    # Print results
    print(result.summary())

    # Show sample trades
    if result.trades:
        print("\nSample Trades (first 10):")
        print("-" * 100)
        for trade in result.trades[:10]:
            direction = "BUY " if trade.direction.value == 1 else "SELL"
            print(f"{trade.entry_time.strftime('%Y-%m-%d %H:%M')} | {direction} | "
                  f"Entry: {trade.entry_price:.5f} | Exit: {trade.exit_price:.5f} | "
                  f"PnL: {trade.pnl:+.2f} | {trade.exit_reason}")

        # Exit reason breakdown
        print("\nExit Reasons:")
        reasons = {}
        for t in result.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} ({count/len(result.trades)*100:.1f}%)")

    # Save results
    results_dir = settings.RESULTS_DIR / "backtests"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{args.pair}_{args.timeframe}_{timestamp}.json"

    output = {
        "pair": args.pair,
        "timeframe": args.timeframe,
        "date_range": f"{df.index[0]} to {df.index[-1]}",
        "params": strategy.params,
        "metrics": {
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "sharpe_ratio": result.sharpe_ratio,
            "total_return": result.total_return,
            "total_return_pct": result.total_return_pct,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_pct": result.max_drawdown_pct,
        }
    }

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
