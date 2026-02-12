#!/usr/bin/env python
"""
Verify live trades against backtest simulation.

Downloads fresh candles, runs the strategy + full backtest engine on recent data,
and shows every simulated trade with entry/exit/SL/TP/outcome. Compares against
OANDA live trade history to confirm the backtest promise is playing out in reality.

Usage:
    # Run backtest on last 3 months of M15 data with live params
    python scripts/verify_live.py --instance rsi_v3_GBP_USD_M15

    # Specify custom date range
    python scripts/verify_live.py --instance rsi_v3_GBP_USD_M15 --start 2026-02-01 --end 2026-02-12

    # Also show signal replay (match live trades to backtest signals)
    python scripts/verify_live.py --instance rsi_v3_GBP_USD_M15 --replay

    # Fresh download from OANDA before running
    python scripts/verify_live.py --instance rsi_v3_GBP_USD_M15 --download

    # Check all active instances
    python scripts/verify_live.py --all --download
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

INSTANCES_DIR = ROOT / "instances"

EXIT_REASON_NAMES = {0: 'SL', 1: 'TP', 2: 'trailing', 3: 'time',
                     4: 'stale', 5: 'ml', 6: 'force_close'}

STRATEGY_MAP = {
    "rsi_v3": ("strategies.rsi_full_v3", "RSIDivergenceFullFastV3"),
    "RSI_Divergence_v3": ("strategies.rsi_full_v3", "RSIDivergenceFullFastV3"),
    "rsi_v1": ("strategies.rsi_full", "RSIDivergenceFullFast"),
    "RSI_Divergence": ("strategies.rsi_full", "RSIDivergenceFullFast"),
    "RSI_Divergence_Full": ("strategies.rsi_full", "RSIDivergenceFullFast"),
    "ema_cross": ("strategies.ema_cross_ml", "EMACrossMLFast"),
    "EMA_Cross_ML": ("strategies.ema_cross_ml", "EMACrossMLFast"),
}


def load_instance(instance_id: str) -> dict:
    """Load instance config."""
    config_path = INSTANCES_DIR / instance_id / "config.json"
    if not config_path.exists():
        print(f"  [ERROR] Config not found: {config_path}")
        return {}
    with open(config_path) as f:
        return json.load(f)


def load_trade_history(instance_id: str) -> list:
    """Load trade history if available."""
    history_path = INSTANCES_DIR / instance_id / "state" / "trade_history.json"
    if not history_path.exists():
        return []
    with open(history_path) as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("trades", data.get("trade_history", []))


def download_fresh_data(pair: str, timeframe: str):
    """Download fresh candle data from OANDA."""
    from data.download import download_data
    print(f"\n  Downloading fresh {pair} M1 data from OANDA...")
    download_data(instrument=pair, granularity="M1", years=1, force_full=False)
    if timeframe != "M1":
        print(f"  Building {timeframe} candles from M1 cache...")


def load_candles(pair: str, timeframe: str, start: datetime = None, end: datetime = None) -> pd.DataFrame:
    """Load candle data, optionally filtered to date range."""
    from data.download import load_data
    # Load with enough lookback for indicators (RSI needs ~200 bars minimum)
    df = load_data(instrument=pair, timeframe=timeframe, auto_download=False, years=5)
    if df.empty:
        return df

    # Apply date filter if specified, keeping lookback for indicators
    if start:
        lookback = timedelta(days=60)  # ~60 days for RSI/MA warmup
        mask_start = pd.Timestamp(start - lookback)
        if df.index.tz is not None:
            mask_start = mask_start.tz_localize(df.index.tz)
        df = df[df.index >= mask_start]
    if end:
        mask_end = pd.Timestamp(end + timedelta(days=1))
        if df.index.tz is not None:
            mask_end = mask_end.tz_localize(df.index.tz)
        df = df[df.index <= mask_end]

    return df


def load_strategy(strategy_name: str):
    """Load strategy class."""
    if strategy_name not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {strategy_name}. Known: {list(STRATEGY_MAP.keys())}")
    import importlib
    module_path, class_name = STRATEGY_MAP[strategy_name]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


def pip_distance(price1: float, price2: float, pair: str = "") -> float:
    """Calculate pip distance."""
    mult = 100 if "JPY" in pair else 10000
    return abs(price1 - price2) * mult


def run_backtest_verification(config: dict, df: pd.DataFrame, start_date: datetime = None):
    """
    Run full backtest with telemetry and return per-trade details.

    Returns list of trade dicts with: entry_time, exit_time, direction, entry_price,
    sl_price, tp_price, exit_price, exit_reason, pnl, bars_held, mfe_r, mae_r
    """
    from optimization.numba_backtest import full_backtest_with_telemetry, get_quote_conversion_rate

    pair = config["pair"]
    timeframe = config["timeframe"]
    params = config["params"]
    strategy_name = config["strategy"]

    pip_size = 0.01 if 'JPY' in pair else 0.0001

    # Set bars_per_year based on timeframe
    tf_bars_per_year = {
        'M1': 252 * 1440, 'M5': 252 * 288, 'M15': 252 * 96,
        'M30': 252 * 48, 'H1': 252 * 22, 'H4': 252 * 6, 'D': 252
    }
    bars_per_year = tf_bars_per_year.get(timeframe, 5544.0)

    # Prepare market data
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    days = df.index.dayofweek.values.astype(np.int64)

    # Generate signals
    strategy = load_strategy(strategy_name)
    strategy._pip_size = pip_size
    n_raw = strategy.precompute_for_dataset(df)
    signal_arrays, mgmt_arrays = strategy.get_all_arrays(params, highs, lows, closes, days)

    n = len(signal_arrays['entry_bars'])
    n_bars = len(highs)

    print(f"  Raw signals: {n_raw}  |  After filtering: {n}")

    if n == 0:
        print("  [WARN] No signals generated!")
        return [], {}

    # Build management arrays with defaults
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
    quality_mult = np.empty(0, dtype=np.float64)

    # No ML
    use_ml = np.zeros(n, dtype=np.bool_)
    ml_min_hold = np.zeros(n, dtype=np.int64)
    ml_threshold = np.ones(n, dtype=np.float64)
    ml_long = np.zeros(n_bars, dtype=np.float64)
    ml_short = np.zeros(n_bars, dtype=np.float64)

    quote_rate = get_quote_conversion_rate(pair, 'USD')

    # Run backtest
    result = full_backtest_with_telemetry(
        signal_arrays['entry_bars'],
        signal_arrays['entry_prices'],
        signal_arrays['directions'],
        signal_arrays['sl_prices'],
        signal_arrays['tp_prices'],
        use_trailing, trail_start, trail_step,
        use_be, be_trigger, be_offset,
        use_partial, partial_pct, partial_target,
        max_bars,
        trail_mode, chandelier_atr_mult, atr_pips_arr, stale_exit_bars,
        ml_long, ml_short, use_ml, ml_min_hold, ml_threshold,
        highs, lows, closes, days,
        10000.0,  # initial capital
        2.0,      # risk pct
        pip_size,
        0,        # max daily trades
        0.0,      # max daily loss
        quality_mult,
        quote_rate,
        bars_per_year,
    )

    (pnls, equity_curve, exit_reasons, bars_held, entry_bar_indices,
     exit_bar_indices, mfe_r, mae_r, signal_indices,
     n_trades, win_rate, pf, sharpe, max_dd, total_ret, r_sq, ontester) = result

    # Build trade list
    bar_times = df.index.tolist()
    trades = []

    for i in range(n_trades):
        entry_bar = int(entry_bar_indices[i])
        exit_bar = int(exit_bar_indices[i])
        sig_idx = int(signal_indices[i])

        entry_time = bar_times[entry_bar] if 0 <= entry_bar < len(bar_times) else None
        exit_time = bar_times[exit_bar] if 0 <= exit_bar < len(bar_times) else None

        direction = int(signal_arrays['directions'][sig_idx])
        entry_price = float(signal_arrays['entry_prices'][sig_idx])
        sl_price = float(signal_arrays['sl_prices'][sig_idx])
        tp_price = float(signal_arrays['tp_prices'][sig_idx])

        # Derive exit price from P&L (accounts for trailing/BE SL moves)
        exit_reason = EXIT_REASON_NAMES.get(int(exit_reasons[i]), '?')
        # Use close at exit bar as best approximation (actual exit may differ for SL/TP
        # exits within a bar, and trailing/BE may have moved the SL)
        if 0 <= exit_bar < len(closes):
            exit_price = float(closes[exit_bar])
        else:
            exit_price = 0.0

        trade = {
            'trade_num': i + 1,
            'entry_time': str(entry_time) if entry_time else '?',
            'exit_time': str(exit_time) if exit_time else '?',
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': float(pnls[i]),
            'bars_held': int(bars_held[i]),
            'mfe_r': float(mfe_r[i]),
            'mae_r': float(mae_r[i]),
            'sl_pips': pip_distance(entry_price, sl_price, pair),
            'tp_pips': pip_distance(entry_price, tp_price, pair),
        }
        trades.append(trade)

    # Filter to start_date if specified
    if start_date:
        ts = pd.Timestamp(start_date, tz='UTC')
        trades = [t for t in trades if pd.Timestamp(t['entry_time']) >= ts]

    # Summary stats
    summary = {
        'total_trades': len(trades),
        'wins': sum(1 for t in trades if t['pnl'] > 0),
        'losses': sum(1 for t in trades if t['pnl'] <= 0),
        'win_rate': sum(1 for t in trades if t['pnl'] > 0) / len(trades) if trades else 0,
        'sharpe': float(sharpe),
        'profit_factor': float(pf),
        'max_dd': float(max_dd),
        'total_return': float(total_ret),
        'r_squared': float(r_sq),
    }

    return trades, summary


def print_trades(trades: list, pair: str, last_n: int = 0):
    """Print trade details table."""
    subset = trades[-last_n:] if last_n > 0 else trades
    if not subset:
        print("  No trades in this period.")
        return

    print(f"\n  {'#':>3} | {'Entry Time':>19} | {'Dir':>4} | {'Entry':>9} | {'SL':>9} | {'TP':>9} | "
          f"{'Exit':>9} | {'Reason':>8} | {'SL pip':>6} | {'TP pip':>6} | {'Bars':>4} | {'MFE_R':>5} | {'P&L':>8}")
    print(f"  {'-'*3}-+-{'-'*19}-+-{'-'*4}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-"
          f"{'-'*9}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*4}-+-{'-'*5}-+-{'-'*8}")

    for t in subset:
        # Shorten entry time for display
        etime = t['entry_time'][:19] if len(t['entry_time']) > 19 else t['entry_time']
        marker = 'X' if t['pnl'] <= 0 else '+'
        print(f"  {marker}{t['trade_num']:>2} | {etime:>19} | {t['direction']:>4} | "
              f"{t['entry_price']:>9.5f} | {t['sl_price']:>9.5f} | {t['tp_price']:>9.5f} | "
              f"{t['exit_price']:>9.5f} | {t['exit_reason']:>8} | {t['sl_pips']:>6.1f} | "
              f"{t['tp_pips']:>6.1f} | {t['bars_held']:>4} | {t['mfe_r']:>5.2f} | "
              f"{t['pnl']:>+8.2f}")


def match_live_trades(backtest_trades: list, live_trades: list, pair: str, timeframe: str):
    """Match live OANDA trades against backtest trades."""
    if not live_trades:
        return

    tf_minutes = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D': 1440}
    max_delta = timedelta(minutes=tf_minutes.get(timeframe, 60) * 3)

    print(f"\n--- Live Trade Matching ---")
    matched = 0
    for lt in live_trades:
        entry_str = lt.get("entry_time", "")
        if not entry_str:
            continue
        try:
            live_time = pd.Timestamp(datetime.fromisoformat(entry_str)).tz_localize(None)
        except (ValueError, TypeError):
            continue

        live_dir = lt.get("direction", "").upper()
        live_entry = lt.get("entry_price", 0)
        live_sl = lt.get("stop_loss", 0)
        live_tp = lt.get("take_profit", 0)
        live_pnl = lt.get("realized_pnl", 0)

        # Find best matching backtest trade
        best = None
        best_delta = timedelta(days=999)
        for bt in backtest_trades:
            try:
                bt_time = pd.Timestamp(bt['entry_time']).tz_localize(None)
            except Exception:
                continue
            delta = abs(live_time - bt_time)
            if delta < best_delta and delta <= max_delta and bt['direction'] == live_dir:
                best_delta = delta
                best = bt

        if best:
            matched += 1
            entry_diff = pip_distance(live_entry, best['entry_price'], pair)
            sl_diff = pip_distance(live_sl, best['sl_price'], pair) if live_sl else 0
            tp_diff = pip_distance(live_tp, best['tp_price'], pair) if live_tp else 0

            status = "MATCH" if entry_diff < 5 and sl_diff < 5 else "CLOSE" if entry_diff < 15 else "MISMATCH"
            symbol = "OK" if status == "MATCH" else ("~" if status == "CLOSE" else "!!")

            print(f"\n  [{symbol}] Live: {live_dir} @ {live_entry:.5f} ({entry_str})")
            print(f"       BT:   {best['direction']} @ {best['entry_price']:.5f} ({best['entry_time']})")
            print(f"       Entry diff: {entry_diff:.1f} pip | SL diff: {sl_diff:.1f} pip | TP diff: {tp_diff:.1f} pip")
            print(f"       Live outcome: {'WIN' if live_pnl > 0 else 'LOSS'} ({live_pnl:+.2f})")
            print(f"       BT outcome:   {best['exit_reason']} ({best['pnl']:>+.2f})")
            if best['exit_reason'] != ('SL' if live_pnl < 0 else 'TP'):
                exit_match = 'SL' if live_pnl < 0 else 'TP/trailing'
                print(f"       [!] Exit mismatch: live={exit_match}, BT={best['exit_reason']}")
        else:
            print(f"\n  [!!] Live: {live_dir} @ {live_entry:.5f} ({entry_str})")
            print(f"       NO MATCHING BACKTEST TRADE within {max_delta}")

    print(f"\n  Match summary: {matched}/{len(live_trades)} live trades matched to backtest trades")


def verify_instance(instance_id: str, start: str = None, end: str = None,
                    last_n: int = 0, do_download: bool = False, do_replay: bool = False):
    """Full verification for one instance."""
    config = load_instance(instance_id)
    if not config:
        return

    pair = config["pair"]
    timeframe = config["timeframe"]
    params = config["params"]
    expectations = config.get("expectations", {})

    print(f"\n{'='*80}")
    print(f"  BACKTEST VERIFICATION: {instance_id}")
    print(f"{'='*80}")
    print(f"  Strategy:    {config.get('strategy', '?')}")
    print(f"  Pair/TF:     {pair} {timeframe}")
    print(f"  Rating:      {config.get('score', '?')}/100 {config.get('rating', '?')}")
    print(f"  Expected WR: {expectations.get('win_rate', 0):.1%}")
    print(f"  Expected PF: {expectations.get('profit_factor', 0):.2f}")

    sl_mode = params.get("sl_mode", "?")
    tp_mode = params.get("tp_mode", "?")
    if sl_mode == "fixed":
        sl_detail = f"{params.get('sl_fixed_pips', '?')} pips"
    elif params.get('sl_atr_pct'):
        sl_detail = f"{params['sl_atr_pct']}% ATR"
    elif params.get('sl_atr_mult'):
        sl_detail = f"{params['sl_atr_mult']}x ATR"
    else:
        sl_detail = "ATR"
    tp_detail = (f"{params.get('tp_rr_ratio', '?')}:1 R:R" if tp_mode == "rr"
                 else params.get("tp_fixed_pips") if tp_mode == "fixed"
                 else f"{params.get('tp_atr_mult', '?')}x ATR")
    print(f"  SL: {sl_mode} ({sl_detail})  |  TP: {tp_mode} ({tp_detail})")
    print(f"  Trailing: {'ON' if params.get('use_trailing') else 'OFF'}"
          f"  |  Break-even: {'ON' if params.get('use_break_even') else 'OFF'}")

    # Download fresh data if requested
    if do_download:
        download_fresh_data(pair, timeframe)

    # Parse date range
    start_date = datetime.fromisoformat(start) if start else datetime.now() - timedelta(days=90)
    end_date = datetime.fromisoformat(end) if end else datetime.now()

    print(f"\n--- Loading {pair} {timeframe} candles ---")
    df = load_candles(pair, timeframe, start_date, end_date)
    if df.empty:
        print("  [ERROR] No candle data available. Try --download first.")
        return

    print(f"  Loaded {len(df):,} candles ({df.index[0]} to {df.index[-1]})")

    # Run full backtest
    print(f"\n--- Running Backtest Simulation ---")
    trades, summary = run_backtest_verification(config, df, start_date)

    if not trades:
        print("  [WARN] No trades generated in this period!")
        return

    # Print summary
    print(f"\n--- Backtest Results ({start_date.date()} to {end_date.date()}) ---")
    print(f"  Trades:        {summary['total_trades']} ({summary['wins']}W / {summary['losses']}L)")
    print(f"  Win rate:      {summary['win_rate']:.1%}  (expected: {expectations.get('win_rate', 0):.1%})")
    print(f"  Sharpe:        {summary['sharpe']:.2f}  (expected: {expectations.get('sharpe', 0):.2f})")
    print(f"  Profit factor: {summary['profit_factor']:.2f}  (expected: {expectations.get('profit_factor', 0):.2f})")
    print(f"  Max drawdown:  {summary['max_dd']:.1f}%")
    print(f"  Total return:  {summary['total_return']:.1f}%")

    # Win rate comparison
    expected_wr = expectations.get('win_rate', 0)
    if expected_wr > 0 and summary['total_trades'] >= 10:
        import math
        se = math.sqrt(expected_wr * (1 - expected_wr) / summary['total_trades'])
        z = (summary['win_rate'] - expected_wr) / se if se > 0 else 0
        if abs(z) < 1.96:
            print(f"  WR status:     WITHIN NORMAL RANGE (z={z:.2f})")
        elif z < -1.96:
            print(f"  WR status:     [ALERT] SIGNIFICANTLY BELOW EXPECTED (z={z:.2f})")
        else:
            print(f"  WR status:     ABOVE EXPECTED (z={z:.2f})")

    # Print individual trades
    print(f"\n--- Individual Trades ---")
    print_trades(trades, pair, last_n=last_n)

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t['exit_reason']
        reasons[r] = reasons.get(r, 0) + 1
    print(f"\n  Exit reasons: {dict(sorted(reasons.items(), key=lambda x: -x[1]))}")

    # Win rate by exit reason
    for reason in sorted(reasons.keys()):
        r_trades = [t for t in trades if t['exit_reason'] == reason]
        r_wins = sum(1 for t in r_trades if t['pnl'] > 0)
        print(f"    {reason}: {len(r_trades)} trades, {r_wins}/{len(r_trades)} wins ({r_wins/len(r_trades):.0%})")

    # Match against live trades if available
    live_trades = load_trade_history(instance_id)
    if live_trades:
        match_live_trades(trades, live_trades, pair, timeframe)
    elif do_replay:
        print(f"\n  [INFO] No trade_history.json found for live trade matching.")
        print(f"  To match against OANDA, sync trade history from VPS first.")

    # BE dependency check: all SL exits being wins = BE is saving trades
    sl_exits = [t for t in trades if t['exit_reason'] == 'SL']
    sl_wins = [t for t in sl_exits if t['pnl'] > 0]
    if sl_exits and len(sl_wins) / len(sl_exits) > 0.8:
        print(f"\n--- BE/Trailing Dependency Warning ---")
        print(f"  {len(sl_wins)}/{len(sl_exits)} SL exits are WINS (BE moved SL into profit)")
        print(f"  The backtest triggers BE on intra-bar high/low wicks.")
        print(f"  If the live system uses candle close only, BE triggers less often")
        print(f"  and those SL exits become actual losses.")
        print(f"  FIX: Ensure live trader passes bar_high/bar_low to manage_positions()")

    # Verdict
    print(f"\n--- Verdict ---")
    wr_ok = abs(summary['win_rate'] - expected_wr) < 0.10 if expected_wr > 0 else True
    pf_ok = summary['profit_factor'] >= 1.0
    if wr_ok and pf_ok:
        print(f"  [OK] Backtest on recent data CONFIRMS pipeline expectations")
    elif pf_ok:
        print(f"  [WARN] Win rate deviates from expected but strategy is still profitable")
    else:
        print(f"  [ALERT] Strategy underperforming on recent data!")

    print()
    return {'trades': trades, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description="Verify live trades against backtest simulation")
    parser.add_argument("--instance", "-i", help="Instance ID to verify")
    parser.add_argument("--all", "-a", action="store_true", help="Verify all active instances")
    parser.add_argument("--start", "-s", help="Start date (YYYY-MM-DD), default: 90 days ago")
    parser.add_argument("--end", "-e", help="End date (YYYY-MM-DD), default: today")
    parser.add_argument("--last", "-n", type=int, default=0, help="Only show last N trades")
    parser.add_argument("--download", "-d", action="store_true", help="Download fresh candles from OANDA first")
    parser.add_argument("--replay", "-r", action="store_true", help="Match live trades against backtest")
    args = parser.parse_args()

    if args.instance:
        instances = [args.instance]
    elif args.all:
        deploy_file = ROOT / "deploy" / "strategies.json"
        if deploy_file.exists():
            with open(deploy_file) as f:
                data = json.load(f)
            instances = [s["id"] for s in data.get("strategies", []) if s.get("enabled", True)]
        else:
            instances = [d.name for d in INSTANCES_DIR.iterdir() if d.is_dir()]
    else:
        parser.print_help()
        sys.exit(1)

    for iid in instances:
        verify_instance(
            iid,
            start=args.start,
            end=args.end,
            last_n=args.last,
            do_download=args.download,
            do_replay=args.replay,
        )


if __name__ == "__main__":
    main()
