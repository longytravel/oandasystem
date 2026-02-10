#!/usr/bin/env python
"""
Independent verification script - proves backtest results are real.
Shows every single trade with dates, prices, and P&L.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from data.download import load_data
from strategies.rsi_full import RSIDivergenceFullFast

# ============================================================
# EXACT PARAMETERS FROM BEST GBP_USD M30 RESULT
# ============================================================
PARAMS = {
    'rsi_period': 14,
    'rsi_overbought': 75,
    'rsi_oversold': 20,
    'min_rsi_diff': 3.0,
    'swing_strength': 7,
    'min_bars_between': 15,
    'max_bars_between': 100,
    'require_pullback': False,
    'use_slope_filter': True,
    'min_price_slope': 25.0,
    'max_price_slope': 65.0,
    'use_rsi_extreme_filter': False,
    'use_trend_filter': True,
    'trend_ma_period': 200,
    'max_spread_pips': 3.0,
    'sl_mode': 'atr',
    'sl_fixed_pips': 50,
    'sl_atr_mult': 1.5,
    'sl_swing_buffer': 15,
    'tp_mode': 'fixed',
    'tp_rr_ratio': 2.5,
    'tp_atr_mult': 4.0,
    'use_trailing': True,
    'trail_start_pips': 50,
    'trail_step_pips': 15,
    'use_break_even': False,
    'be_trigger_pips': 40,
    'be_offset_pips': 0,
    'use_partial_close': True,
    'partial_close_pct': 0.7,
    'use_time_filter': True,
    'trade_start_hour': 8,
    'trade_end_hour': 23,
    'trade_monday': True,
    'trade_friday': True,
    'friday_close_hour': 18,
}


def run_verification(pair: str, timeframe: str):
    """Run verification showing every trade."""
    print(f"\n{'='*80}")
    print(f"VERIFICATION: {pair} {timeframe}")
    print(f"{'='*80}")

    # Load data
    df = load_data(pair, timeframe, auto_download=False)
    print(f"Data: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    # Split same as optimization (6 months forward)
    total = len(df)
    fwd_bars = int(total * 6 / 24)
    df_back = df.iloc[:total - fwd_bars]
    df_forward = df.iloc[total - fwd_bars:]

    split_date = df_forward.index[0].date()
    print(f"Split: Back ends {df_back.index[-1].date()}, Forward starts {split_date}")

    # Initialize strategy
    strategy = RSIDivergenceFullFast()
    strategy.set_pip_size(pair)
    pip_size = strategy._pip_size

    print(f"\nRunning with parameters:")
    print(f"  rsi_period={PARAMS['rsi_period']}, swing_strength={PARAMS['swing_strength']}")
    print(f"  sl_mode={PARAMS['sl_mode']}, tp_rr_ratio={PARAMS['tp_rr_ratio']}")
    print(f"  use_trailing={PARAMS['use_trailing']}, use_trend_filter={PARAMS['use_trend_filter']}")

    # Process each period
    for period_name, period_df in [("BACK", df_back), ("FORWARD", df_forward)]:
        print(f"\n{'-'*80}")
        print(f"{period_name} PERIOD: {period_df.index[0].date()} to {period_df.index[-1].date()}")
        print(f"{'-'*80}")

        # Precompute signals
        strategy.precompute_for_dataset(period_df)
        signals = strategy.filter_signals(strategy._precomputed_signals, PARAMS)

        print(f"Signals generated: {len(signals)}")

        if not signals:
            print("  No signals!")
            continue

        # Simulate trades
        highs = period_df['high'].values
        lows = period_df['low'].values
        closes = period_df['close'].values
        times = period_df.index

        trades = []
        equity = 10000.0

        for sig in signals:
            bar = sig.bar
            if bar >= len(closes):
                continue

            direction = sig.direction
            entry_price = sig.price

            # Get SL/TP
            sl, tp = strategy.compute_sl_tp(sig, PARAMS, pip_size)

            # Apply spread
            spread_cost = 1.5 * pip_size
            if direction == 1:
                entry_price += spread_cost
            else:
                entry_price -= spread_cost

            # Simulate exit
            exit_price = None
            exit_bar = None
            exit_reason = None

            # Track trailing stop
            current_sl = sl
            highest = entry_price if direction == 1 else 999999
            lowest = entry_price if direction == -1 else 0

            for i in range(bar + 1, len(closes)):
                if direction == 1:  # Long
                    # Update trailing
                    if PARAMS['use_trailing']:
                        if highs[i] > highest:
                            highest = highs[i]
                            profit_pips = (highest - entry_price) / pip_size
                            if profit_pips >= PARAMS['trail_start_pips']:
                                new_sl = highest - PARAMS['trail_step_pips'] * pip_size
                                if new_sl > current_sl:
                                    current_sl = new_sl

                    # Check SL
                    if lows[i] <= current_sl:
                        exit_price = current_sl
                        exit_bar = i
                        exit_reason = "SL" if current_sl == sl else "TRAIL"
                        break
                    # Check TP
                    if highs[i] >= tp:
                        exit_price = tp
                        exit_bar = i
                        exit_reason = "TP"
                        break
                else:  # Short
                    # Update trailing
                    if PARAMS['use_trailing']:
                        if lows[i] < lowest:
                            lowest = lows[i]
                            profit_pips = (entry_price - lowest) / pip_size
                            if profit_pips >= PARAMS['trail_start_pips']:
                                new_sl = lowest + PARAMS['trail_step_pips'] * pip_size
                                if new_sl < current_sl:
                                    current_sl = new_sl

                    # Check SL
                    if highs[i] >= current_sl:
                        exit_price = current_sl
                        exit_bar = i
                        exit_reason = "SL" if current_sl == sl else "TRAIL"
                        break
                    # Check TP
                    if lows[i] <= tp:
                        exit_price = tp
                        exit_bar = i
                        exit_reason = "TP"
                        break

            if exit_price is None:
                exit_price = closes[-1]
                exit_bar = len(closes) - 1
                exit_reason = "END"

            # Calculate P&L
            if direction == 1:
                pnl_pips = (exit_price - entry_price) / pip_size
            else:
                pnl_pips = (entry_price - exit_price) / pip_size

            # Risk-based sizing (1% risk)
            sl_pips = abs(entry_price - sl) / pip_size
            risk_amt = equity * 0.01
            position_size = risk_amt / sl_pips if sl_pips > 0 else 0.01
            pnl_dollars = pnl_pips * position_size

            equity += pnl_dollars

            trades.append({
                'entry_time': times[bar],
                'exit_time': times[exit_bar],
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'entry': entry_price,
                'exit': exit_price,
                'sl': sl,
                'tp': tp,
                'pnl_pips': pnl_pips,
                'pnl_dollars': pnl_dollars,
                'exit_reason': exit_reason,
                'equity': equity,
            })

        # Print all trades
        print(f"\nALL TRADES ({len(trades)} total):")
        print(f"{'#':<3} {'Entry Date':<12} {'Exit Date':<12} {'Dir':<6} {'Entry':<10} {'Exit':<10} {'P&L Pips':<10} {'Reason':<6} {'Equity':<10}")
        print("-" * 90)

        wins = 0
        total_pnl = 0
        for i, t in enumerate(trades):
            if t['pnl_pips'] > 0:
                wins += 1
            total_pnl += t['pnl_pips']

            print(f"{i+1:<3} {str(t['entry_time'].date()):<12} {str(t['exit_time'].date()):<12} "
                  f"{t['direction']:<6} {t['entry']:<10.5f} {t['exit']:<10.5f} "
                  f"{t['pnl_pips']:<+10.1f} {t['exit_reason']:<6} {t['equity']:<10.2f}")

        # Summary
        if trades:
            win_rate = wins / len(trades) * 100
            avg_win = np.mean([t['pnl_pips'] for t in trades if t['pnl_pips'] > 0]) if wins > 0 else 0
            avg_loss = np.mean([t['pnl_pips'] for t in trades if t['pnl_pips'] <= 0]) if wins < len(trades) else 0

            print(f"\n{period_name} SUMMARY:")
            print(f"  Trades: {len(trades)}")
            print(f"  Wins: {wins} ({win_rate:.1f}%)")
            print(f"  Total P&L: {total_pnl:+.1f} pips")
            print(f"  Avg Win: {avg_win:+.1f} pips")
            print(f"  Avg Loss: {avg_loss:+.1f} pips")
            print(f"  Final Equity: ${equity:,.2f} (started $10,000)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', default='GBP_USD')
    parser.add_argument('--timeframe', default='M30')
    args = parser.parse_args()
    run_verification(args.pair, args.timeframe)
