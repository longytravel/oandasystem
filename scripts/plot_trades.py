#!/usr/bin/env python
"""
Generate interactive trade visualization charts from backtest results.

Usage:
    python scripts/plot_trades.py --pair GBP_USD --timeframe H1
    python scripts/plot_trades.py --pair GBP_USD --months 3 --params-file results/best_params.json
    python scripts/plot_trades.py --pair EUR_USD --save-png
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
from data.download import load_data
from visualization.charts import TradeChart
from config.settings import settings


def main():
    parser = argparse.ArgumentParser(description="Generate trade visualization chart")
    parser.add_argument("--pair", default="GBP_USD", help="Currency pair")
    parser.add_argument("--timeframe", default="H1", help="Timeframe")
    parser.add_argument("--months", type=int, default=3, help="Months of data to display")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade (%%)")
    parser.add_argument("--spread", type=float, default=1.5, help="Spread in pips")
    parser.add_argument("--params-file", help="JSON file with strategy parameters")
    parser.add_argument("--output", help="Output file path (defaults to results/chart_PAIR_TF.html)")
    parser.add_argument("--save-png", action="store_true", help="Also save as PNG")
    parser.add_argument("--no-show", action="store_true", help="Don't open in browser")
    parser.add_argument("--no-equity", action="store_true", help="Hide equity curve")
    parser.add_argument("--no-sl-tp", action="store_true", help="Hide SL/TP lines")
    parser.add_argument("--show-weekends", action="store_true", help="Show weekend gaps (hidden by default)")

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    print(f"\n{'=' * 60}")
    print(f"TRADE CHART: {args.pair} {args.timeframe}")
    print(f"{'=' * 60}")

    # Load data
    try:
        df = load_data(args.pair, args.timeframe, auto_download=True)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Filter to requested months
    if args.months:
        cutoff = df.index[-1] - timedelta(days=30 * args.months)
        df = df[df.index >= cutoff]

    logger.info(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Load strategy parameters
    params = None
    if args.params_file:
        with open(args.params_file) as f:
            data = json.load(f)
            params = data.get("best_params", data)
        logger.info(f"Loaded params from {args.params_file}")

    # Create strategy and run backtest
    strategy = RSIDivergenceStrategy(params)
    logger.info(f"Strategy: {strategy.name}")

    engine = BacktestEngine(
        initial_capital=args.capital,
        spread_pips=args.spread,
        slippage_pips=0.5,
    )

    logger.info("Running backtest...")
    result = engine.run(strategy, df, risk_per_trade=args.risk)

    if not result.trades:
        logger.warning("No trades generated - chart will show candlesticks only")

    # Print quick summary
    print(f"\nBacktest Summary:")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.1%}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Total Return: {result.total_return:+.2f} ({result.total_return_pct:+.1%})")
    print(f"  Max Drawdown: {result.max_drawdown:.2f} ({result.max_drawdown_pct:.1%})")

    # Create chart
    title = f"{args.pair} {args.timeframe} - {result.total_trades} trades | " \
            f"WR: {result.win_rate:.1%} | PF: {result.profit_factor:.2f}"

    chart = TradeChart(
        title=title,
        height=900,
        show_equity=not args.no_equity,
        hide_weekends=not args.show_weekends
    )

    logger.info("Creating chart...")
    fig = chart.create_chart(
        df=df,
        result=result,
        show_sl_tp_lines=not args.no_sl_tp
    )

    # Determine output path
    output_dir = settings.RESULTS_DIR / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        html_path = Path(args.output)
    else:
        html_path = output_dir / f"chart_{args.pair}_{args.timeframe}.html"

    # Save HTML
    chart.save_html(fig, str(html_path), auto_open=not args.no_show)
    logger.info(f"Saved HTML: {html_path}")

    # Save PNG if requested
    if args.save_png:
        png_path = html_path.with_suffix('.png')
        try:
            chart.save_png(fig, str(png_path))
            logger.info(f"Saved PNG: {png_path}")
        except Exception as e:
            logger.warning(f"Could not save PNG (install kaleido): {e}")

    print(f"\nChart saved to: {html_path}")
    if not args.no_show:
        print("Opening in browser...")


if __name__ == "__main__":
    main()
