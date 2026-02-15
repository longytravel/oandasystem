#!/usr/bin/env python
"""
Check live trades against backtest expectations.

Reads trade history, compares SL/TP levels, win rates, streak probabilities,
and REPLAYS the strategy on historical data to verify signals match.

Usage:
    python scripts/check_trades.py --instance rsi_v3_GBP_USD_M15
    python scripts/check_trades.py --instance rsi_v3_GBP_USD_M15 --replay
    python scripts/check_trades.py --all --replay
    python scripts/check_trades.py --instance rsi_v3_GBP_USD_H1 --last 10
"""
import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

INSTANCES_DIR = ROOT / "instances"
STRATEGIES_FILE = ROOT / "deploy" / "strategies.json"

# Strategy name -> (module, class)
STRATEGY_MAP = {
    "rsi_v3": ("strategies.rsi_full_v3", "RSIDivergenceFullFastV3"),
    "RSI_Divergence_v3": ("strategies.rsi_full_v3", "RSIDivergenceFullFastV3"),
    "rsi_v1": ("strategies.rsi_full", "RSIDivergenceFullFast"),
    "RSI_Divergence": ("strategies.rsi_full", "RSIDivergenceFullFast"),
    "ema_cross": ("strategies.ema_cross_ml", "EMACrossMLFast"),
    "EMA_Cross_ML": ("strategies.ema_cross_ml", "EMACrossMLFast"),
    "fair_price_ma": ("strategies.fair_price_ma", "FairPriceMAStrategy"),
}


def load_instance(instance_id: str) -> dict:
    """Load config and trade history for an instance."""
    idir = INSTANCES_DIR / instance_id
    result = {"id": instance_id, "config": None, "trades": [], "health": None}

    config_path = idir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            result["config"] = json.load(f)

    history_path = idir / "state" / "trade_history.json"
    if history_path.exists():
        with open(history_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                result["trades"] = data
            elif isinstance(data, dict):
                result["trades"] = data.get("trades", data.get("trade_history", []))

    health_path = idir / "health.json"
    if health_path.exists():
        with open(health_path) as f:
            result["health"] = json.load(f)

    return result


def pip_distance(price1: float, price2: float, pair: str = "") -> float:
    """Calculate pip distance between two prices."""
    if "JPY" in pair:
        return abs(price1 - price2) * 100
    return abs(price1 - price2) * 10000


def check_trade_params(trade: dict, config: dict, pair: str) -> dict:
    """Check a single trade's SL/TP against config expectations."""
    params = config.get("params", {})
    issues = []
    info = {}

    entry = trade.get("entry_price", 0)
    sl = trade.get("stop_loss", 0)
    tp = trade.get("take_profit", 0)

    if entry and sl:
        sl_pips = pip_distance(entry, sl, pair)
        info["sl_pips"] = round(sl_pips, 1)

        sl_mode = params.get("sl_mode", "")
        if sl_mode == "fixed":
            expected_sl = params.get("sl_fixed_pips", 0)
            if expected_sl > 0 and abs(sl_pips - expected_sl) > expected_sl * 0.15:
                issues.append(f"SL {sl_pips:.1f} pips vs expected ~{expected_sl} pips (fixed)")

    if entry and tp:
        tp_pips = pip_distance(entry, tp, pair)
        info["tp_pips"] = round(tp_pips, 1)

        tp_mode = params.get("tp_mode", "")
        if tp_mode == "fixed":
            expected_tp = params.get("tp_fixed_pips", 0)
            if expected_tp > 0 and abs(tp_pips - expected_tp) > expected_tp * 0.15:
                issues.append(f"TP {tp_pips:.1f} pips vs expected ~{expected_tp} pips (fixed)")
        elif tp_mode == "rr":
            sl_pips_val = info.get("sl_pips", 0)
            if sl_pips_val > 0:
                expected_rr = params.get("tp_rr_ratio", 0)
                actual_rr = tp_pips / sl_pips_val
                info["rr_ratio"] = round(actual_rr, 2)
                if expected_rr > 0 and abs(actual_rr - expected_rr) > expected_rr * 0.2:
                    issues.append(f"R:R {actual_rr:.1f}:1 vs expected {expected_rr:.1f}:1")

    exit_reason = trade.get("exit_reason", "")
    pnl = trade.get("realized_pnl", 0)
    info["exit_reason"] = exit_reason
    info["pnl"] = pnl

    if exit_reason == "SL" and pnl > 0:
        issues.append(f"SL exit but positive P&L ({pnl:+.2f}) - unusual")
    if exit_reason == "TP" and pnl < 0:
        issues.append(f"TP exit but negative P&L ({pnl:+.2f}) - unusual")

    return {"info": info, "issues": issues}


def analyze_streaks(trades: list, expected_wr: float) -> dict:
    """Analyze win/loss streaks and their probability."""
    if not trades:
        return {}

    outcomes = ["W" if t.get("realized_pnl", 0) > 0 else "L" for t in trades]

    current_streak = 1
    for i in range(len(outcomes) - 2, -1, -1):
        if outcomes[i] == outcomes[-1]:
            current_streak += 1
        else:
            break
    streak_type = "win" if outcomes[-1] == "W" else "loss"

    max_loss_streak = 0
    current_loss = 0
    for o in outcomes:
        if o == "L":
            current_loss += 1
            max_loss_streak = max(max_loss_streak, current_loss)
        else:
            current_loss = 0

    loss_rate = 1 - expected_wr
    streak_prob = (loss_rate ** current_streak) if streak_type == "loss" else (expected_wr ** current_streak)

    n = len(trades)
    expected_max_loss = (math.log(n) / abs(math.log(loss_rate))) if (loss_rate > 0 and loss_rate < 1 and n > 0) else 0

    return {
        "total_trades": len(trades),
        "wins": outcomes.count("W"),
        "losses": outcomes.count("L"),
        "live_win_rate": outcomes.count("W") / len(outcomes),
        "current_streak": current_streak,
        "current_streak_type": streak_type,
        "current_streak_probability": streak_prob,
        "max_loss_streak": max_loss_streak,
        "expected_max_loss_streak": round(expected_max_loss, 1),
        "sequence": "".join(outcomes[-20:]),
    }


# ── Signal Replay ─────────────────────────────────────────

def load_strategy(strategy_name: str):
    """Dynamically load a FastStrategy class."""
    key = strategy_name
    if key not in STRATEGY_MAP:
        # Try common aliases
        for k, v in STRATEGY_MAP.items():
            if k.lower() == strategy_name.lower():
                key = k
                break
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}. Known: {list(STRATEGY_MAP.keys())}")

    module_path, class_name = STRATEGY_MAP[key]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def load_candle_data(pair: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load candle data covering the given time range from cache."""
    from data.download import load_data
    df = load_data(instrument=pair, timeframe=timeframe, auto_download=False, years=5)
    # Filter to range with some lookback for indicators
    lookback = timedelta(days=90)
    mask = (df.index >= pd.Timestamp(start - lookback)) & (df.index <= pd.Timestamp(end + timedelta(days=1)))
    return df[mask]


def replay_signals(config: dict, trades: list) -> list:
    """
    Re-run the strategy on historical data and check if signals match live trades.

    For each live trade, finds the nearest strategy signal and compares:
    - Entry bar/time match
    - Direction match
    - SL/TP price match

    Returns list of replay results per trade.
    """
    strategy_name = config.get("strategy", "")
    pair = config.get("pair", "")
    timeframe = config.get("timeframe", "")
    params = config.get("params", {})

    if not trades:
        return []

    # Parse trade times to find data range
    trade_times = []
    for t in trades:
        entry_str = t.get("entry_time", "")
        if entry_str:
            try:
                dt = datetime.fromisoformat(entry_str)
                trade_times.append(dt)
            except (ValueError, TypeError):
                pass

    if not trade_times:
        print("  [WARN] No parseable entry times in trade history")
        return []

    earliest = min(trade_times)
    latest = max(trade_times)

    print(f"\n--- Signal Replay ---")
    print(f"  Loading {pair} {timeframe} data from {earliest.date()} to {latest.date()}...")

    try:
        df = load_candle_data(pair, timeframe, earliest, latest)
    except (FileNotFoundError, Exception) as e:
        print(f"  [ERROR] Cannot load candle data: {e}")
        print(f"  Ensure M1 or {timeframe} cache exists in data/oanda/")
        return []

    if df.empty:
        print(f"  [ERROR] No candle data for this period")
        return []

    print(f"  Loaded {len(df):,} candles ({df.index[0]} to {df.index[-1]})")

    # Load and run strategy
    print(f"  Running {strategy_name} signal generation...")
    try:
        fast_strategy = load_strategy(strategy_name)
        pip_size = 0.01 if 'JPY' in pair else 0.0001
        fast_strategy._pip_size = pip_size
        n_signals = fast_strategy.precompute_for_dataset(df)
        print(f"  Found {n_signals} raw signals in dataset")

        if n_signals == 0:
            print(f"  [WARN] Strategy produced 0 signals on this data range!")
            return []

        # Get filtered signal arrays with the live params
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        closes = df['close'].values.astype(np.float64)
        days = df.index.dayofweek.values.astype(np.int64)

        signal_arrays, mgmt_arrays = fast_strategy.get_all_arrays(
            params, highs, lows, closes, days
        )

        entry_bars = signal_arrays['entry_bars']
        directions = signal_arrays['directions']
        entry_prices = signal_arrays['entry_prices']
        sl_prices = signal_arrays['sl_prices']
        tp_prices = signal_arrays['tp_prices']

        print(f"  {len(entry_bars)} signals after filtering with live params")

    except Exception as e:
        print(f"  [ERROR] Strategy replay failed: {e}")
        import traceback
        traceback.print_exc()
        return []

    # Build signal lookup: map bar index -> signal details
    bar_times = df.index.tolist()
    signal_list = []
    for i in range(len(entry_bars)):
        bar_idx = int(entry_bars[i])
        if 0 <= bar_idx < len(bar_times):
            signal_list.append({
                'bar_idx': bar_idx,
                'time': bar_times[bar_idx],
                'direction': 'BUY' if int(directions[i]) == 1 else 'SELL',
                'entry_price': float(entry_prices[i]),
                'sl_price': float(sl_prices[i]),
                'tp_price': float(tp_prices[i]),
            })

    # Match each live trade to the nearest backtest signal
    results = []
    tf_minutes = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D': 1440}
    max_delta = timedelta(minutes=tf_minutes.get(timeframe, 60) * 2)  # Allow 2-bar tolerance

    for trade in trades:
        entry_str = trade.get("entry_time", "")
        if not entry_str:
            continue

        try:
            trade_time = datetime.fromisoformat(entry_str)
        except (ValueError, TypeError):
            continue

        trade_time_ts = pd.Timestamp(trade_time)
        if trade_time_ts.tzinfo is not None:
            trade_time_ts = trade_time_ts.tz_localize(None)

        trade_dir = trade.get("direction", "").upper()
        trade_entry = trade.get("entry_price", 0)
        trade_sl = trade.get("stop_loss", 0)
        trade_tp = trade.get("take_profit", 0)

        # Find closest matching signal
        best_match = None
        best_delta = timedelta(days=999)

        for sig in signal_list:
            sig_time = sig['time']
            if hasattr(sig_time, 'to_pydatetime'):
                sig_time = sig_time.to_pydatetime()
            if hasattr(sig_time, 'tzinfo') and sig_time.tzinfo is not None:
                sig_time = sig_time.replace(tzinfo=None)

            delta = abs(trade_time_ts - pd.Timestamp(sig_time))
            if delta < best_delta and delta <= max_delta:
                # Also check direction matches
                if sig['direction'] == trade_dir:
                    best_delta = delta
                    best_match = sig

        result = {
            'trade_time': entry_str,
            'trade_direction': trade_dir,
            'trade_entry': trade_entry,
            'trade_sl': trade_sl,
            'trade_tp': trade_tp,
        }

        if best_match:
            # Compare prices
            entry_diff = pip_distance(trade_entry, best_match['entry_price'], pair)
            sl_diff = pip_distance(trade_sl, best_match['sl_price'], pair) if trade_sl else 0
            tp_diff = pip_distance(trade_tp, best_match['tp_price'], pair) if trade_tp else 0

            time_diff = best_delta.total_seconds()

            result['matched'] = True
            result['signal_time'] = str(best_match['time'])
            result['signal_direction'] = best_match['direction']
            result['signal_entry'] = best_match['entry_price']
            result['signal_sl'] = best_match['sl_price']
            result['signal_tp'] = best_match['tp_price']
            result['entry_diff_pips'] = round(entry_diff, 1)
            result['sl_diff_pips'] = round(sl_diff, 1)
            result['tp_diff_pips'] = round(tp_diff, 1)
            result['time_diff_seconds'] = time_diff

            # Print result
            status = "MATCH" if entry_diff < 5 and sl_diff < 5 and tp_diff < 5 else "CLOSE" if entry_diff < 15 else "MISMATCH"
            marker = "OK" if status == "MATCH" else ("~" if status == "CLOSE" else "!!")

            print(f"\n  [{marker}] {trade_dir} @ {trade_entry:.5f}  ({entry_str})")
            print(f"       Backtest signal: {best_match['direction']} @ {best_match['entry_price']:.5f}  ({best_match['time']})")
            print(f"       Entry diff: {entry_diff:.1f} pip  |  SL diff: {sl_diff:.1f} pip  |  TP diff: {tp_diff:.1f} pip")
            if time_diff > 0:
                print(f"       Time diff: {time_diff:.0f}s")

            if entry_diff >= 5:
                result['issue'] = f"Entry price differs by {entry_diff:.1f} pips (spread/slippage?)"
                print(f"       [!] {result['issue']}")
            if sl_diff >= 5:
                result['issue'] = f"SL price differs by {sl_diff:.1f} pips"
                print(f"       [!] {result['issue']}")
            if tp_diff >= 5:
                result['issue'] = f"TP price differs by {tp_diff:.1f} pips"
                print(f"       [!] {result['issue']}")

        else:
            result['matched'] = False
            print(f"\n  [!!] {trade_dir} @ {trade_entry:.5f}  ({entry_str})")
            print(f"       NO MATCHING BACKTEST SIGNAL within {max_delta}")
            print(f"       This trade should NOT have been taken!")

            # Show nearest signals for debugging
            nearby = []
            for sig in signal_list:
                sig_time = sig['time']
                if hasattr(sig_time, 'to_pydatetime'):
                    sig_time = sig_time.to_pydatetime()
                if hasattr(sig_time, 'tzinfo') and sig_time.tzinfo is not None:
                    sig_time = sig_time.replace(tzinfo=None)
                delta = abs(trade_time_ts - pd.Timestamp(sig_time))
                if delta < timedelta(days=3):
                    nearby.append((delta, sig))

            if nearby:
                nearby.sort(key=lambda x: x[0])
                print(f"       Nearest signals (within 3 days):")
                for delta, sig in nearby[:5]:
                    hours = delta.total_seconds() / 3600
                    print(f"         {sig['direction']} @ {sig['entry_price']:.5f} at {sig['time']} ({hours:.1f}h away)")

        results.append(result)

    # Summary
    matched = sum(1 for r in results if r.get('matched'))
    total = len(results)
    print(f"\n  Replay Summary: {matched}/{total} trades matched backtest signals")
    if matched < total:
        unmatched = total - matched
        print(f"  [ALERT] {unmatched} trade(s) have NO matching backtest signal!")
    else:
        print(f"  [OK] All live trades have corresponding backtest signals")

    return results


# ── Main Check ────────────────────────────────────────────

def check_instance(instance_id: str, last_n: int = 0, do_replay: bool = False) -> dict:
    """Full check for one instance."""
    data = load_instance(instance_id)
    config = data["config"]
    trades = data["trades"]

    print(f"\n{'='*60}")
    print(f"  TRADE CHECK: {instance_id}")
    print(f"{'='*60}")

    if not config:
        print("  [ERROR] No config.json found")
        return {"status": "error", "message": "no config"}

    if not trades:
        print("  [INFO] No trade history yet")
        return {"status": "ok", "message": "no trades"}

    pair = config.get("pair", "")
    expectations = config.get("expectations", {})
    params = config.get("params", {})

    # Summary
    print(f"\n  Strategy: {config.get('strategy', '?')}")
    print(f"  Pair/TF:  {pair} {config.get('timeframe', '?')}")
    print(f"  Rating:   {config.get('score', '?')}/100 {config.get('rating', '?')}")
    print(f"  Trades:   {len(trades)} total")

    sl_mode = params.get("sl_mode", "?")
    tp_mode = params.get("tp_mode", "?")
    sl_detail = params.get("sl_fixed_pips") if sl_mode == "fixed" else f"{params.get('sl_atr_pct', '?')}% ATR"
    tp_detail = params.get("tp_fixed_pips") if tp_mode == "fixed" else (
        f"{params.get('tp_rr_ratio', '?')}:1 R:R" if tp_mode == "rr" else f"{params.get('tp_atr_mult', '?')}x ATR"
    )
    print(f"  SL:       {sl_mode} ({sl_detail})")
    print(f"  TP:       {tp_mode} ({tp_detail})")

    # Win rate comparison
    expected_wr = expectations.get("win_rate", 0)
    print(f"\n--- Win Rate ---")

    subset = trades[-last_n:] if last_n > 0 else trades
    wins = sum(1 for t in subset if t.get("realized_pnl", 0) > 0)
    losses = len(subset) - wins
    live_wr = wins / len(subset) if subset else 0

    print(f"  Expected:   {expected_wr:.1%}")
    print(f"  Live:       {live_wr:.1%}  ({wins}W / {losses}L over {len(subset)} trades)")

    if len(subset) >= 5:
        se = math.sqrt(expected_wr * (1 - expected_wr) / len(subset))
        z = (live_wr - expected_wr) / se if se > 0 else 0
        if abs(z) < 1.96:
            print(f"  Status:     WITHIN NORMAL RANGE (z={z:.2f})")
        elif z < -1.96:
            print(f"  Status:     [ALERT] SIGNIFICANTLY BELOW EXPECTED (z={z:.2f})")
        else:
            print(f"  Status:     ABOVE EXPECTED (z={z:.2f})")
    else:
        print(f"  Status:     TOO FEW TRADES for statistical test (need 5+)")

    # Streak analysis
    print(f"\n--- Streaks ---")
    streaks = analyze_streaks(subset, expected_wr)
    if streaks:
        print(f"  Sequence (last 20): {streaks['sequence']}")
        print(f"  Current streak:     {streaks['current_streak']} {streaks['current_streak_type']}s "
              f"(probability: {streaks['current_streak_probability']:.1%})")
        print(f"  Max loss streak:    {streaks['max_loss_streak']} "
              f"(expected max over {len(subset)} trades: ~{streaks['expected_max_loss_streak']})")

    # Individual trade checks
    print(f"\n--- Trade Details (last {min(last_n or 10, len(trades))}) ---")
    check_trades_list = trades[-(last_n or 10):]
    all_issues = []

    for i, trade in enumerate(check_trades_list):
        result = check_trade_params(trade, config, pair)
        info = result["info"]
        issues = result["issues"]
        all_issues.extend(issues)

        entry_time = trade.get("entry_time", "?")
        exit_time = trade.get("exit_time", "?")
        direction = trade.get("direction", "?")
        entry_price = trade.get("entry_price", 0)
        pnl = trade.get("realized_pnl", 0)
        exit_reason = info.get("exit_reason", "?")

        marker = "X" if pnl < 0 else "+"
        print(f"\n  [{marker}] Trade {i+1}: {direction} @ {entry_price:.5f}")
        print(f"      Entry:  {entry_time}")
        print(f"      Exit:   {exit_time} ({exit_reason})")
        print(f"      SL:     {info.get('sl_pips', '?')} pips  |  TP: {info.get('tp_pips', '?')} pips"
              f"  |  R:R: {info.get('rr_ratio', '?')}:1")
        print(f"      P&L:    {pnl:+.2f}")

        if issues:
            for issue in issues:
                print(f"      [!] {issue}")

    # P&L summary
    print(f"\n--- P&L Summary ---")
    total_pnl = sum(t.get("realized_pnl", 0) for t in subset)
    win_trades = [t for t in subset if t.get("realized_pnl", 0) > 0]
    loss_trades = [t for t in subset if t.get("realized_pnl", 0) <= 0]
    avg_win = sum(t["realized_pnl"] for t in win_trades) / len(win_trades) if win_trades else 0
    avg_loss = sum(t["realized_pnl"] for t in loss_trades) / len(loss_trades) if loss_trades else 0

    print(f"  Total P&L:  {total_pnl:+.2f}")
    print(f"  Avg win:    {avg_win:+.2f}  |  Avg loss: {avg_loss:+.2f}")
    if avg_loss != 0:
        print(f"  Payoff ratio: {abs(avg_win/avg_loss):.2f}:1")

    expected_pf = expectations.get("profit_factor", 0)
    if expected_pf:
        gross_profit = sum(t["realized_pnl"] for t in subset if t.get("realized_pnl", 0) > 0)
        gross_loss = abs(sum(t["realized_pnl"] for t in subset if t.get("realized_pnl", 0) < 0))
        live_pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        print(f"  Profit factor: {live_pf:.2f} (expected: {expected_pf:.2f})")

    # Signal replay
    replay_results = []
    if do_replay:
        replay_results = replay_signals(config, subset)

    # Verdict
    print(f"\n--- Verdict ---")
    if all_issues:
        print(f"  [!] {len(all_issues)} parameter issue(s) found:")
        for issue in all_issues:
            print(f"      - {issue}")
    else:
        print(f"  [OK] All trade parameters match config expectations")

    if replay_results:
        unmatched = sum(1 for r in replay_results if not r.get('matched'))
        if unmatched:
            print(f"  [ALERT] {unmatched} trade(s) had NO matching backtest signal!")
        else:
            print(f"  [OK] All trades confirmed by backtest signal replay")

    if len(subset) >= 10 and live_wr < expected_wr - 0.15:
        print(f"  [ALERT] Win rate {live_wr:.0%} is >15% below expected {expected_wr:.0%}")
    elif len(subset) >= 5:
        print(f"  [OK] Performance within expected range for {len(subset)} trades")
    else:
        print(f"  [INFO] Only {len(subset)} trades - too early to judge")

    print()
    return {
        "status": "ok",
        "trades": len(subset),
        "live_wr": live_wr,
        "expected_wr": expected_wr,
        "issues": all_issues,
        "streaks": streaks,
        "replay": replay_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Check live trades against backtest expectations")
    parser.add_argument("--instance", "-i", help="Instance ID to check")
    parser.add_argument("--all", "-a", action="store_true", help="Check all instances")
    parser.add_argument("--last", "-n", type=int, default=0, help="Only check last N trades")
    parser.add_argument("--replay", "-r", action="store_true",
                       help="Replay strategy on historical data to verify signals match")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    instances_to_check = []

    if args.instance:
        instances_to_check = [args.instance]
    elif STRATEGIES_FILE.exists():
        with open(STRATEGIES_FILE) as f:
            data = json.load(f)
        instances_to_check = [s["id"] for s in data.get("strategies", []) if s.get("enabled", True)]
    else:
        parser.print_help()
        sys.exit(1)

    all_results = {}
    for iid in instances_to_check:
        all_results[iid] = check_instance(iid, last_n=args.last, do_replay=args.replay)

    if args.json:
        # Convert non-serializable types
        print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
