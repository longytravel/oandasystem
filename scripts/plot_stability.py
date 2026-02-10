"""
Plot Parameter Stability - Visual comparison of equity curves

Shows the best parameters vs neighboring parameter variations to
visually demonstrate robustness. Similar curves = robust, divergent = overfit.

Usage:
    python scripts/plot_stability.py --pair GBP_USD --timeframe H1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from loguru import logger

from strategies.rsi_full import RSIDivergenceFullFast
from optimization.unified_optimizer import UnifiedOptimizer
from optimization.numba_backtest import full_backtest_numba


def load_data(pair: str, timeframe: str = 'H1') -> pd.DataFrame:
    """Load OANDA data for a pair."""
    data_dir = Path(__file__).parent.parent / 'data' / 'oanda'

    parquet_path = data_dir / f'{pair}_{timeframe}.parquet'
    csv_path = data_dir / f'{pair}_{timeframe}.csv'

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(f"Data file not found: {parquet_path} or {csv_path}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


def run_backtest_get_equity(strategy, params, df, pip_size, spread_pips=1.5):
    """Run backtest and return equity curve."""
    strategy.precompute_for_dataset(df)

    arrays = {
        'highs': df['high'].values.astype(np.float64),
        'lows': df['low'].values.astype(np.float64),
        'closes': df['close'].values.astype(np.float64),
        'days': df.index.dayofweek.values.astype(np.int64),
    }

    signal_arrays, mgmt_arrays = strategy.get_all_arrays(
        params,
        arrays['highs'],
        arrays['lows'],
        arrays['closes'],
        arrays['days'],
    )

    if len(signal_arrays['entry_bars']) < 3:
        return None, None, 0

    # Apply spread
    entry_prices = np.where(
        signal_arrays['directions'] == 1,
        signal_arrays['entry_prices'] + spread_pips * pip_size,
        signal_arrays['entry_prices'] - spread_pips * pip_size
    )

    n = len(signal_arrays['entry_bars'])

    use_trailing = mgmt_arrays.get('use_trailing', np.zeros(n, dtype=np.bool_))
    trail_start = mgmt_arrays.get('trail_start_pips', np.zeros(n, dtype=np.float64))
    trail_step = mgmt_arrays.get('trail_step_pips', np.zeros(n, dtype=np.float64))
    use_be = mgmt_arrays.get('use_breakeven', np.zeros(n, dtype=np.bool_))
    be_trigger = mgmt_arrays.get('be_trigger_pips', np.zeros(n, dtype=np.float64))
    be_offset = mgmt_arrays.get('be_offset_pips', np.zeros(n, dtype=np.float64))
    use_partial = mgmt_arrays.get('use_partial', np.zeros(n, dtype=np.bool_))
    partial_pct = mgmt_arrays.get('partial_pct', np.zeros(n, dtype=np.float64))
    partial_target = mgmt_arrays.get('partial_target_pips', np.zeros(n, dtype=np.float64))
    max_bars = mgmt_arrays.get('max_bars', np.zeros(n, dtype=np.int64))
    trail_mode = mgmt_arrays.get('trail_mode', np.zeros(n, dtype=np.int64))
    chandelier_atr_mult = mgmt_arrays.get('chandelier_atr_mult', np.full(n, 3.0, dtype=np.float64))
    atr_pips_arr = mgmt_arrays.get('atr_pips', np.full(n, 35.0, dtype=np.float64))
    stale_exit_bars = mgmt_arrays.get('stale_exit_bars', np.zeros(n, dtype=np.int64))
    quality_mult = mgmt_arrays.get('quality_mult', np.empty(0, dtype=np.float64))

    # V6: ML exit arrays
    n_bars = len(arrays['highs'])
    use_ml = mgmt_arrays.get('use_ml_exit', np.zeros(n, dtype=np.bool_))
    ml_min_hold_arr = mgmt_arrays.get('ml_min_hold', np.zeros(n, dtype=np.int64))
    ml_threshold_arr = mgmt_arrays.get('ml_threshold', np.ones(n, dtype=np.float64))

    if hasattr(strategy, 'get_ml_score_arrays') and np.any(use_ml):
        ml_long, ml_short = strategy.get_ml_score_arrays(
            params, arrays['highs'], arrays['lows'], arrays['closes']
        )
    else:
        ml_long = np.zeros(n_bars, dtype=np.float64)
        ml_short = np.zeros(n_bars, dtype=np.float64)

    result = full_backtest_numba(
        signal_arrays['entry_bars'],
        entry_prices,
        signal_arrays['directions'],
        signal_arrays['sl_prices'],
        signal_arrays['tp_prices'],
        use_trailing, trail_start, trail_step,
        use_be, be_trigger, be_offset,
        use_partial, partial_pct, partial_target,
        max_bars,
        trail_mode, chandelier_atr_mult, atr_pips_arr, stale_exit_bars,
        ml_long, ml_short, use_ml, ml_min_hold_arr, ml_threshold_arr,
        arrays['highs'],
        arrays['lows'],
        arrays['closes'],
        arrays['days'],
        10000.0,  # initial capital
        1.0,      # risk per trade
        pip_size,
        params.get('max_daily_trades', 0),
        params.get('max_daily_loss_pct', 0.0),
        quality_mult,
    )

    # Reconstruct equity curve from trades
    # Result: trades, win_rate, profit_factor, sharpe, max_dd, total_return, r_squared, ontester
    trades = result[0]
    total_return = result[5]

    if trades == 0:
        return None, None, 0

    # Simple equity reconstruction - we need to re-run to get individual trades
    # For now, just return the summary
    return total_return, result[6], trades  # return, r2, trades


def get_neighbor_params(base_params, param_name, param_space, direction):
    """Get params with one parameter shifted by direction (-1 or +1)."""
    if param_name not in param_space or param_name not in base_params:
        return None

    values = param_space[param_name]
    current_val = base_params[param_name]

    if current_val not in values:
        return None

    current_idx = values.index(current_val)
    new_idx = current_idx + direction

    if new_idx < 0 or new_idx >= len(values):
        return None

    new_params = dict(base_params)
    new_params[param_name] = values[new_idx]
    return new_params


def simulate_equity_curve(strategy, params, df, pip_size, initial_capital=10000):
    """Simulate and return full equity curve with timestamps."""
    strategy.precompute_for_dataset(df)

    arrays = {
        'highs': df['high'].values.astype(np.float64),
        'lows': df['low'].values.astype(np.float64),
        'closes': df['close'].values.astype(np.float64),
        'days': df.index.dayofweek.values.astype(np.int64),
    }

    signal_arrays, mgmt_arrays = strategy.get_all_arrays(
        params,
        arrays['highs'],
        arrays['lows'],
        arrays['closes'],
        arrays['days'],
    )

    if len(signal_arrays['entry_bars']) < 3:
        return None, None

    # Apply spread
    spread_pips = 1.5
    entry_prices = np.where(
        signal_arrays['directions'] == 1,
        signal_arrays['entry_prices'] + spread_pips * pip_size,
        signal_arrays['entry_prices'] - spread_pips * pip_size
    )

    # Simulate trades manually to get equity curve
    equity = [initial_capital]
    equity_times = [df.index[0]]

    entry_bars = signal_arrays['entry_bars']
    directions = signal_arrays['directions']
    sl_prices = signal_arrays['sl_prices']
    tp_prices = signal_arrays['tp_prices']

    capital = initial_capital
    risk_per_trade = 1.0  # 1%

    for i in range(len(entry_bars)):
        entry_bar = entry_bars[i]
        entry_price = entry_prices[i]
        direction = directions[i]
        sl = sl_prices[i]
        tp = tp_prices[i]

        if entry_bar >= len(df):
            continue

        # Calculate position size
        sl_pips = abs(entry_price - sl) / pip_size
        if sl_pips == 0:
            continue

        risk_amount = capital * (risk_per_trade / 100)

        # Simulate trade exit
        exit_price = None
        for bar in range(entry_bar + 1, min(entry_bar + 200, len(df))):
            high = arrays['highs'][bar]
            low = arrays['lows'][bar]

            if direction == 1:  # Long
                if low <= sl:
                    exit_price = sl
                    break
                elif high >= tp:
                    exit_price = tp
                    break
            else:  # Short
                if high >= sl:
                    exit_price = sl
                    break
                elif low <= tp:
                    exit_price = tp
                    break

        if exit_price is None:
            exit_price = arrays['closes'][min(entry_bar + 199, len(df) - 1)]

        # Calculate P&L
        if direction == 1:
            pnl_pips = (exit_price - entry_price) / pip_size
        else:
            pnl_pips = (entry_price - exit_price) / pip_size

        pnl = (pnl_pips / sl_pips) * risk_amount
        capital += pnl

        equity.append(capital)
        equity_times.append(df.index[min(entry_bar + 1, len(df) - 1)])

    return equity_times, equity


def plot_stability(pair: str, timeframe: str = 'H1', config_path: str = None):
    """Plot equity curves for best params and neighbors."""

    # Load config or run optimization
    if config_path:
        with open(config_path) as f:
            config = json.load(f)
        best_params = {}
        for group in config['parameters'].values():
            best_params.update(group)
    else:
        # Load from latest robust results
        results_dir = Path(__file__).parent.parent / 'results' / 'robust'
        result_files = sorted(results_dir.glob('*.json'), reverse=True)
        if not result_files:
            print("No robust results found. Run robust optimization first.")
            return

        with open(result_files[0]) as f:
            data = json.load(f)

        if not data['results']:
            print("No results in file")
            return

        best_params = data['results'][0]['params']

    print(f"\nLoading {pair} {timeframe} data...")
    df = load_data(pair, timeframe)

    # Split into back/forward
    split_idx = int(len(df) * 0.7)
    df_back = df.iloc[:split_idx]
    df_forward = df.iloc[split_idx:]

    print(f"Back: {df_back.index[0]} to {df_back.index[-1]} ({len(df_back)} bars)")
    print(f"Forward: {df_forward.index[0]} to {df_forward.index[-1]} ({len(df_forward)} bars)")

    # Initialize strategy
    strategy = RSIDivergenceFullFast()
    strategy.set_pip_size(pair)
    pip_size = strategy._pip_size

    # Get parameter space for neighbors
    groups = strategy.get_parameter_groups()
    param_space = {}
    for group in groups.values():
        param_space.update(group.get_param_space())

    # Key parameters to test stability
    key_params = ['rsi_period', 'swing_strength', 'sl_mode', 'tp_mode',
                  'min_rsi_diff', 'tp_rr_ratio', 'use_trend_filter']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{pair} {timeframe} - Parameter Stability Analysis', fontsize=14, fontweight='bold')

    # Plot 1: Best params equity curve (full period)
    ax1 = axes[0, 0]
    print("\nSimulating best params on full data...")
    times, equity = simulate_equity_curve(strategy, best_params, df, pip_size)
    if times:
        ax1.plot(times, equity, 'b-', linewidth=2, label='Best Params')
        ax1.axvline(x=df_back.index[-1], color='r', linestyle='--', alpha=0.7, label='Back/Forward Split')
        ax1.set_title('Best Parameters - Full Period')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Backtest period with neighbors
    ax2 = axes[0, 1]
    print("Simulating neighbors on backtest period...")

    # Best params
    times, equity = simulate_equity_curve(strategy, best_params, df_back, pip_size)
    if times:
        ax2.plot(times, equity, 'b-', linewidth=2.5, label='Best', zorder=10)

    # Neighbors (lighter colors)
    colors = plt.cm.Reds(np.linspace(0.3, 0.7, 4))
    neighbor_count = 0

    for param_name in key_params[:4]:  # Test 4 key params
        for direction in [-1, 1]:
            neighbor_params = get_neighbor_params(best_params, param_name, param_space, direction)
            if neighbor_params:
                times, equity = simulate_equity_curve(strategy, neighbor_params, df_back, pip_size)
                if times and len(equity) > 1:
                    label = f"{param_name}={neighbor_params[param_name]}"
                    ax2.plot(times, equity, '-', color=colors[neighbor_count % 4],
                            linewidth=1, alpha=0.7, label=label if neighbor_count < 4 else None)
                    neighbor_count += 1

    ax2.set_title('Backtest Period - Best vs Neighbors')
    ax2.set_ylabel('Equity ($)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Forward period with neighbors
    ax3 = axes[1, 0]
    print("Simulating neighbors on forward period...")

    # Best params
    times, equity = simulate_equity_curve(strategy, best_params, df_forward, pip_size)
    if times:
        ax3.plot(times, equity, 'b-', linewidth=2.5, label='Best', zorder=10)

    # Neighbors
    colors = plt.cm.Greens(np.linspace(0.3, 0.7, 4))
    neighbor_count = 0

    for param_name in key_params[:4]:
        for direction in [-1, 1]:
            neighbor_params = get_neighbor_params(best_params, param_name, param_space, direction)
            if neighbor_params:
                times, equity = simulate_equity_curve(strategy, neighbor_params, df_forward, pip_size)
                if times and len(equity) > 1:
                    label = f"{param_name}={neighbor_params[param_name]}"
                    ax3.plot(times, equity, '-', color=colors[neighbor_count % 4],
                            linewidth=1, alpha=0.7, label=label if neighbor_count < 4 else None)
                    neighbor_count += 1

    ax3.set_title('Forward Period - Best vs Neighbors')
    ax3.set_ylabel('Equity ($)')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Parameter stability bar chart
    ax4 = axes[1, 1]
    print("Calculating stability scores...")

    stability_data = []

    for param_name in key_params:
        if param_name not in best_params or param_name not in param_space:
            continue

        # Get base performance
        _, base_equity = simulate_equity_curve(strategy, best_params, df_back, pip_size)
        if not base_equity:
            continue
        base_return = (base_equity[-1] - base_equity[0]) / base_equity[0] * 100

        # Get neighbor performances
        neighbor_returns = []
        for direction in [-1, 1]:
            neighbor_params = get_neighbor_params(best_params, param_name, param_space, direction)
            if neighbor_params:
                _, neighbor_equity = simulate_equity_curve(strategy, neighbor_params, df_back, pip_size)
                if neighbor_equity and len(neighbor_equity) > 1:
                    ret = (neighbor_equity[-1] - neighbor_equity[0]) / neighbor_equity[0] * 100
                    neighbor_returns.append(ret)

        if neighbor_returns and base_return > 0:
            avg_neighbor = np.mean(neighbor_returns)
            stability = min(avg_neighbor / base_return * 100, 150)  # Cap at 150%
            stability_data.append((param_name, stability, best_params[param_name]))

    if stability_data:
        params_names = [f"{d[0]}\n({d[2]})" for d in stability_data]
        stabilities = [d[1] for d in stability_data]
        colors = ['green' if s >= 70 else 'orange' if s >= 40 else 'red' for s in stabilities]

        bars = ax4.barh(params_names, stabilities, color=colors, alpha=0.7)
        ax4.axvline(x=70, color='green', linestyle='--', alpha=0.5, label='Stable threshold')
        ax4.axvline(x=40, color='orange', linestyle='--', alpha=0.5, label='Fragile threshold')
        ax4.set_xlabel('Stability %')
        ax4.set_title('Parameter Stability (Neighbor Avg / Best)')
        ax4.set_xlim(0, 150)

        # Add value labels
        for bar, stab in zip(bars, stabilities):
            ax4.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    f'{stab:.0f}%', va='center', fontsize=9)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / 'results' / 'charts'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'stability_{pair}_{timeframe}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot parameter stability visualization')
    parser.add_argument('--pair', default='GBP_USD', help='Currency pair')
    parser.add_argument('--timeframe', default='H1', help='Timeframe')
    parser.add_argument('--config', help='Path to config JSON (optional)')

    args = parser.parse_args()

    plot_stability(
        pair=args.pair,
        timeframe=args.timeframe,
        config_path=args.config,
    )


if __name__ == '__main__':
    main()
