#!/usr/bin/env python
"""
Run backtest with optimized parameters and plot equity curve.
"""
import sys
from pathlib import Path
import argparse
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger

from strategies.rsi_fast import get_strategy
from data.download import load_data
from optimization.numba_backtest import full_backtest_numba


def run_backtest(strategy, df, params, initial_capital=10000, risk_pct=1.0, spread_pips=1.5):
    """Run backtest using the same method as the optimizer."""

    # Pre-compute signals (must use precompute_for_dataset to store internally)
    logger.info("Pre-computing signals...")
    n_signals_raw = strategy.precompute_for_dataset(df)
    logger.info(f"Raw signals: {n_signals_raw}")

    # Get all arrays using strategy method (same as optimizer)
    signal_arrays, mgmt_arrays = strategy.get_all_arrays(
        params,
        df['high'].values.astype(np.float64),
        df['low'].values.astype(np.float64),
        df['close'].values.astype(np.float64),
        df.index.dayofweek.values.astype(np.int64),
    )

    n_signals = len(signal_arrays['entry_bars'])
    logger.info(f"Filtered signals: {n_signals}")

    if n_signals == 0:
        return None

    # Apply spread (same as optimizer does)
    pip_size = strategy._pip_size
    entry_prices = np.where(
        signal_arrays['directions'] == 1,
        signal_arrays['entry_prices'] + spread_pips * pip_size,  # Buy at worse price
        signal_arrays['entry_prices'] - spread_pips * pip_size   # Sell at worse price
    )
    signal_arrays['entry_prices'] = entry_prices

    # Build management arrays - use directly from strategy (already correct type/shape)
    n = n_signals
    use_trailing = mgmt_arrays.get('use_trailing', np.zeros(n, dtype=np.bool_))
    trail_start = mgmt_arrays.get('trail_start_pips', np.zeros(n, dtype=np.float64))
    trail_step = mgmt_arrays.get('trail_step_pips', np.zeros(n, dtype=np.float64))
    use_be = mgmt_arrays.get('use_breakeven', np.zeros(n, dtype=np.bool_))
    be_trigger = mgmt_arrays.get('be_trigger_pips', np.zeros(n, dtype=np.float64))
    be_offset = mgmt_arrays.get('be_offset_pips', np.zeros(n, dtype=np.float64))
    use_partial = mgmt_arrays.get('use_partial', np.zeros(n, dtype=np.bool_))
    partial_pct = mgmt_arrays.get('partial_pct', np.full(n, 0.5, dtype=np.float64))
    partial_target = mgmt_arrays.get('partial_target_pips', np.zeros(n, dtype=np.float64))
    max_bars = mgmt_arrays.get('max_bars', np.full(n, 1000, dtype=np.int64))

    # Ensure correct types
    use_trailing = np.asarray(use_trailing, dtype=np.bool_)
    trail_start = np.asarray(trail_start, dtype=np.float64)
    trail_step = np.asarray(trail_step, dtype=np.float64)
    use_be = np.asarray(use_be, dtype=np.bool_)
    be_trigger = np.asarray(be_trigger, dtype=np.float64)
    be_offset = np.asarray(be_offset, dtype=np.float64)
    use_partial = np.asarray(use_partial, dtype=np.bool_)
    partial_pct = np.asarray(partial_pct, dtype=np.float64)
    partial_target = np.asarray(partial_target, dtype=np.float64)
    max_bars = np.asarray(max_bars, dtype=np.int64)
    trail_mode = mgmt_arrays.get('trail_mode', np.zeros(n, dtype=np.int64))
    trail_mode = np.asarray(trail_mode, dtype=np.int64)
    chandelier_atr_mult = mgmt_arrays.get('chandelier_atr_mult', np.full(n, 3.0, dtype=np.float64))
    chandelier_atr_mult = np.asarray(chandelier_atr_mult, dtype=np.float64)
    atr_pips_arr = mgmt_arrays.get('atr_pips', np.full(n, 35.0, dtype=np.float64))
    atr_pips_arr = np.asarray(atr_pips_arr, dtype=np.float64)
    stale_exit_bars = mgmt_arrays.get('stale_exit_bars', np.zeros(n, dtype=np.int64))
    stale_exit_bars = np.asarray(stale_exit_bars, dtype=np.int64)

    # V6: ML exit arrays
    highs_arr = df['high'].values.astype(np.float64)
    lows_arr = df['low'].values.astype(np.float64)
    closes_arr = df['close'].values.astype(np.float64)
    n_bars = len(highs_arr)

    use_ml = np.asarray(mgmt_arrays.get('use_ml_exit', np.zeros(n, dtype=np.bool_)), dtype=np.bool_)
    ml_min_hold_arr = np.asarray(mgmt_arrays.get('ml_min_hold', np.zeros(n, dtype=np.int64)), dtype=np.int64)
    ml_threshold_arr = np.asarray(mgmt_arrays.get('ml_threshold', np.ones(n, dtype=np.float64)), dtype=np.float64)

    if hasattr(strategy, 'get_ml_score_arrays') and np.any(use_ml):
        ml_long, ml_short = strategy.get_ml_score_arrays(params, highs_arr, lows_arr, closes_arr)
    else:
        ml_long = np.zeros(n_bars, dtype=np.float64)
        ml_short = np.zeros(n_bars, dtype=np.float64)

    # Run full backtest
    # Signature: signal arrays, management arrays, market data, account params
    logger.info("Running Numba backtest...")
    pip_size = strategy._pip_size

    result = full_backtest_numba(
        # Signal arrays (5)
        signal_arrays['entry_bars'],
        signal_arrays['entry_prices'],
        signal_arrays['directions'],
        signal_arrays['sl_prices'],
        signal_arrays['tp_prices'],
        # Management arrays (10 + 4 V5 + 5 V6)
        use_trailing,
        trail_start,
        trail_step,
        use_be,
        be_trigger,
        be_offset,
        use_partial,
        partial_pct,
        partial_target,
        max_bars,
        trail_mode, chandelier_atr_mult, atr_pips_arr, stale_exit_bars,
        ml_long, ml_short, use_ml, ml_min_hold_arr, ml_threshold_arr,
        # Market data (4)
        highs_arr,
        lows_arr,
        closes_arr,
        df.index.dayofweek.values.astype(np.int64),
        # Account params (5)
        initial_capital,
        risk_pct,
        pip_size,
        0,    # max_daily_trades (0 = unlimited)
        5.0,  # max_daily_loss_pct
        spread_pips=spread_pips,
        slippage_pips=0.5,
    )

    n_trades, win_rate, pf, sharpe, max_dd, total_ret, r_squared, ontester, sortino, ulcer = result

    return {
        'trades': int(n_trades),
        'win_rate': win_rate * 100,  # Convert to percentage
        'profit_factor': pf,
        'sharpe': sharpe,
        'max_drawdown_pct': max_dd,
        'total_return_pct': total_ret,
        'r_squared': r_squared,
        'ontester': ontester,
        'final_equity': initial_capital * (1 + total_ret / 100)
    }


def plot_equity_estimate(metrics, df, title, save_path):
    """
    Create an estimated equity curve visualization.
    Since Numba doesn't return trade-by-trade data, we estimate the curve shape.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    initial_capital = 10000
    final_equity = metrics['final_equity']
    n_trades = metrics['trades']
    win_rate = metrics['win_rate'] / 100
    max_dd = metrics['max_drawdown_pct'] / 100
    r_squared = metrics['r_squared']

    # Create estimated equity curve based on metrics
    # Use R² to determine smoothness, win_rate for zigzag pattern
    n_points = min(n_trades, 500)  # Limit points for performance

    if n_points > 0:
        # Generate random but reproducible equity path
        np.random.seed(42)

        # Start with linear growth
        t = np.linspace(0, 1, n_points + 1)
        linear_growth = initial_capital + (final_equity - initial_capital) * t

        # Add noise inversely proportional to R²
        noise_scale = (1 - r_squared) * (final_equity - initial_capital) * 0.3
        noise = np.cumsum(np.random.randn(n_points + 1) * noise_scale / np.sqrt(n_points))
        noise = noise - noise * t  # Taper noise to end at 0

        equity_curve = linear_growth + noise

        # Ensure max drawdown matches
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = (peaks - equity_curve) / peaks
        if drawdowns.max() > 0:
            scale = max_dd / drawdowns.max()
            equity_curve = peaks - (peaks - equity_curve) * scale

        # Ensure final value matches
        equity_curve[-1] = final_equity

        # Time axis
        dates = pd.date_range(df.index[0], df.index[-1], periods=len(equity_curve))
    else:
        equity_curve = np.array([initial_capital, final_equity])
        dates = [df.index[0], df.index[-1]]

    # Main equity curve
    ax1 = axes[0]
    ax1.plot(dates, equity_curve, 'b-', linewidth=1.5, label='Estimated Equity')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)

    # Trend line
    x = np.arange(len(equity_curve))
    slope, intercept = np.polyfit(x, equity_curve, 1)
    trend = slope * x + intercept
    ax1.plot(dates, trend, 'r--', alpha=0.7, label=f'Trend (R²={r_squared:.3f})')

    ax1.fill_between(dates, initial_capital, equity_curve,
                     where=equity_curve >= initial_capital, alpha=0.3, color='green')
    ax1.fill_between(dates, initial_capital, equity_curve,
                     where=equity_curve < initial_capital, alpha=0.3, color='red')

    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Metrics box
    metrics_text = (
        f"Trades: {metrics['trades']}\n"
        f"Win Rate: {metrics['win_rate']:.1f}%\n"
        f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        f"Total Return: {metrics['total_return_pct']:.1f}%\n"
        f"Max Drawdown: {metrics['max_drawdown_pct']:.1f}%\n"
        f"Sharpe: {metrics['sharpe']:.2f}\n"
        f"R²: {metrics['r_squared']:.3f}\n"
        f"OnTester: {metrics['ontester']:.0f}"
    )
    ax1.text(0.98, 0.02, metrics_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')

    # Drawdown chart
    ax2 = axes[1]
    peaks = np.maximum.accumulate(equity_curve)
    drawdown = (peaks - equity_curve) / peaks * 100

    ax2.fill_between(dates, 0, drawdown, color='red', alpha=0.5)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylim(max(drawdown) * 1.2 if max(drawdown) > 0 else 10, 0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot equity curve from optimization results")
    parser.add_argument("--results", required=True, help="Path to optimization results JSON")
    parser.add_argument("--pair", default="EUR_USD", help="Currency pair")
    parser.add_argument("--timeframe", default="M1", help="Timeframe")
    parser.add_argument("--strategy", default="rsi_v3", help="Strategy name")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade (%)")
    parser.add_argument("--save", default="equity_curve.png", help="Save plot to file")
    parser.add_argument("--use-best", action="store_true",
                        help="Use best combined result instead of locked params")

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {message}")

    # Load results
    with open(args.results) as f:
        results = json.load(f)

    # Get parameters
    if args.use_best and 'results' in results and results['results']:
        params = results['results'][0]['params']
        logger.info("Using best combined result parameters")
    else:
        params = results.get('locked_params', results.get('best_params', {}))
        logger.info("Using locked parameters from staged optimization")

    logger.info(f"Parameters: {len(params)} total")

    # Load strategy
    strategy = get_strategy(args.strategy)
    strategy.set_pip_size(args.pair)

    # Load data
    logger.info(f"Loading {args.pair} {args.timeframe}...")
    df = load_data(args.pair, args.timeframe)
    logger.info(f"Loaded {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Run backtest
    metrics = run_backtest(strategy, df, params,
                          initial_capital=args.capital, risk_pct=args.risk)

    if metrics is None:
        print("No trades generated!")
        return

    # Print summary
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS - {args.pair} {args.timeframe}")
    print(f"{'='*60}")
    print(f"Period:          {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total Trades:    {metrics['trades']}")
    print(f"Win Rate:        {metrics['win_rate']:.1f}%")
    print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio:    {metrics['sharpe']:.2f}")
    print(f"Total Return:    {metrics['total_return_pct']:.1f}%")
    print(f"Max Drawdown:    {metrics['max_drawdown_pct']:.1f}%")
    print(f"R² (smoothness): {metrics['r_squared']:.3f}")
    print(f"OnTester Score:  {metrics['ontester']:.0f}")
    print(f"Final Equity:    ${metrics['final_equity']:,.2f}")
    print(f"{'='*60}")

    # Plot
    title = f"RSI Divergence - {args.pair} {args.timeframe}\n{df.index[0].date()} to {df.index[-1].date()}"
    plot_equity_estimate(metrics, df, title, args.save)


if __name__ == "__main__":
    main()
