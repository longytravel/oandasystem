#!/usr/bin/env python
"""
Check live trades against backtest expectations.

Reads trade history, compares SL/TP levels, win rates, streak probabilities,
and REPLAYS the strategy on historical data to verify signals match.

Usage:
    # From instance trade history (on VPS):
    python scripts/check_trades.py --instance rsi_v3_GBP_USD_M15
    python scripts/check_trades.py --instance rsi_v3_GBP_USD_M15 --replay
    python scripts/check_trades.py --all --replay

    # From OANDA CSV export (local or VPS):
    python scripts/check_trades.py --csv path/to/transactions.csv
    python scripts/check_trades.py --csv path/to/transactions.csv --replay
"""
import argparse
import csv
import json
import math
import importlib
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

INSTANCES_DIR = ROOT / "instances"
STRATEGIES_FILE = ROOT / "deploy" / "strategies.json"
ENV_FILE = ROOT / ".env"


def ensure_oanda_credentials(api_key_override: str = None) -> str:
    """
    Ensure OANDA API credentials are available.
    Priority: --api-key arg > .env file > interactive prompt (saves to .env).
    Returns the API key.
    """
    if api_key_override:
        return api_key_override

    # Check .env file
    if ENV_FILE.exists():
        from dotenv import dotenv_values
        env = dotenv_values(ENV_FILE)
        key = env.get('OANDA_API_KEY', '')
        if key and key != 'your-api-key-here':
            return key

    # Check environment variable
    import os
    key = os.environ.get('OANDA_API_KEY', '')
    if key and key != 'your-api-key-here':
        return key

    # Prompt interactively and save to .env
    print("\n  OANDA API credentials not found (.env missing).")
    print("  Get your API key from: https://www.oanda.com/demo-account/tpa/personal_token")
    api_key = input("  Enter OANDA API Key: ").strip()
    if not api_key:
        raise ValueError("API key is required for signal replay")

    account_id = input("  Enter OANDA Account ID (e.g. 101-004-12345678-001): ").strip()
    account_type = input("  Account type [practice]: ").strip() or "practice"

    # Save to .env
    with open(ENV_FILE, 'w') as f:
        f.write(f"OANDA_API_KEY={api_key}\n")
        f.write(f"OANDA_ACCOUNT_ID={account_id}\n")
        f.write(f"OANDA_ACCOUNT_TYPE={account_type}\n")
    print(f"  Saved credentials to {ENV_FILE}")

    # Reload settings
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE, override=True)

    return api_key

# Strategy name -> (module, class)
STRATEGY_MAP = {
    "rsi_v1": ("strategies.archive.rsi_full", "RSIDivergenceFullFast"),  # archived, kept for VPS compat
    "rsi_v3": ("strategies.rsi_full_v3", "RSIDivergenceFullFastV3"),
    "RSI_Divergence_v3": ("strategies.rsi_full_v3", "RSIDivergenceFullFastV3"),
    "rsi_v4": ("strategies.rsi_full_v4", "RSIDivergenceFullFastV4"),
    "RSI_Divergence_v4": ("strategies.rsi_full_v4", "RSIDivergenceFullFastV4"),
    "rsi_v5": ("strategies.rsi_full_v5", "RSIDivergenceFullFastV5"),
    "RSI_Divergence_v5": ("strategies.rsi_full_v5", "RSIDivergenceFullFastV5"),
    "ema_cross": ("strategies.ema_cross_ml", "EMACrossMLStrategy"),
    "EMA_Cross_ML": ("strategies.ema_cross_ml", "EMACrossMLStrategy"),
    "fair_price_ma": ("strategies.fair_price_ma", "FairPriceMAStrategy"),
    "Fair_Price_MA": ("strategies.fair_price_ma", "FairPriceMAStrategy"),
    "donchian_breakout": ("strategies.donchian_breakout", "DonchianBreakoutStrategy"),
    "Donchian_Breakout": ("strategies.donchian_breakout", "DonchianBreakoutStrategy"),
    "bollinger_squeeze": ("strategies.bollinger_squeeze", "BollingerSqueezeStrategy"),
    "Bollinger_Squeeze": ("strategies.bollinger_squeeze", "BollingerSqueezeStrategy"),
    "london_breakout": ("strategies.london_breakout", "LondonBreakoutStrategy"),
    "London_Breakout": ("strategies.london_breakout", "LondonBreakoutStrategy"),
    "stochastic_adx": ("strategies.stochastic_adx", "StochasticADXStrategy"),
    "Stochastic_ADX": ("strategies.stochastic_adx", "StochasticADXStrategy"),
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


# ── CSV Import ────────────────────────────────────────────

def parse_oanda_csv(csv_path: str) -> list:
    """
    Parse OANDA transaction CSV into normalized trade records.

    Tracks MARKET_ORDER fills as entries, links SL/TP orders set ON_FILL,
    and matches exits (STOP_LOSS_ORDER / TAKE_PROFIT_ORDER fills).
    Returns list of trade dicts compatible with check_trade_params().
    """
    trades = []
    pending = {}  # instrument -> entry dict (most recent open)

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    last_entry = None  # Track most recent entry for SL/TP linking
    for row in rows:
        tx_type = row.get('TRANSACTION TYPE', '').strip()
        details = row.get('DETAILS', '').strip()
        instrument = row.get('INSTRUMENT', '').strip()
        price = row.get('PRICE', '').strip()
        units = row.get('UNITS', '').strip()
        direction = row.get('DIRECTION', '').strip()
        sl_val = row.get('STOP LOSS', '').strip()
        tp_val = row.get('TAKE PROFIT', '').strip()
        pl_val = row.get('PL', '').strip()
        ts_raw = row.get('TRANSACTION DATE', '').strip()

        if not ts_raw:
            continue

        # Parse timestamp: "2026-02-17 13:00:13 -12" -> UTC datetime
        try:
            parts = ts_raw.rsplit(' ', 1)
            dt_str = parts[0]
            tz_offset = int(parts[1]) if len(parts) > 1 else 0
            dt_local = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            dt_utc = dt_local - timedelta(hours=tz_offset)
        except Exception:
            continue

        pair = instrument.replace('/', '_')

        # ── Entry fills ──
        if tx_type == 'ORDER_FILL' and details == 'MARKET_ORDER' and instrument and price:
            pending[pair] = {
                'entry_time': dt_utc.isoformat(),
                'entry_price': float(price),
                'units': float(units) if units else 0,
                'direction': direction.upper(),
                'instrument': instrument,
                'pair': pair,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                '_entry_utc': dt_utc,
            }
            last_entry = pending[pair]

        # ── SL/TP set on fill (comes right after entry) ──
        # ON_FILL rows have no instrument, so link to the most recent entry only.
        # Previous bug: matched ALL entries within 5s, so when multiple trades
        # open within seconds, later SL/TP overwrote earlier entries.
        if tx_type == 'STOP_LOSS_ORDER' and details == 'ON_FILL' and price:
            if last_entry and abs((dt_utc - last_entry['_entry_utc']).total_seconds()) < 5:
                last_entry['stop_loss'] = float(price)
        if tx_type == 'TAKE_PROFIT_ORDER' and details == 'ON_FILL' and price:
            if last_entry and abs((dt_utc - last_entry['_entry_utc']).total_seconds()) < 5:
                last_entry['take_profit'] = float(price)

        # ── Exits ──
        if tx_type == 'ORDER_FILL' and details in ('STOP_LOSS_ORDER', 'TAKE_PROFIT_ORDER'):
            if not instrument:
                continue
            exit_type = 'TP' if details == 'TAKE_PROFIT_ORDER' else 'SL'
            pnl = float(pl_val) if pl_val else 0.0

            if pair in pending:
                entry = pending.pop(pair)
                entry['exit_time'] = dt_utc.isoformat()
                entry['exit_reason'] = exit_type
                entry['realized_pnl'] = pnl
                entry.pop('_entry_utc', None)
                trades.append(entry)
            else:
                # Exit of pre-existing position (entry before CSV window)
                trades.append({
                    'entry_time': '',
                    'entry_price': 0,
                    'direction': direction.upper(),
                    'instrument': instrument,
                    'pair': pair,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'exit_time': dt_utc.isoformat(),
                    'exit_reason': exit_type,
                    'realized_pnl': pnl,
                })

    # Remaining open positions
    for entry in pending.values():
        entry['exit_time'] = ''
        entry['exit_reason'] = 'OPEN'
        entry['realized_pnl'] = 0
        entry.pop('_entry_utc', None)
        trades.append(entry)

    return trades


def _fetch_instance_tags(client, trades):
    """
    Enrich trade dicts with 'instance_id' from OANDA tradeClientExtensions.tag.

    Matches CSV trades to OANDA ORDER_FILL transactions by time, instrument, direction.
    This eliminates guesswork — we get the exact strategy instance that placed each trade.
    """
    entry_trades = [t for t in trades if t.get('entry_time')]
    if not entry_trades:
        return

    times = []
    for t in entry_trades:
        try:
            times.append(datetime.fromisoformat(t['entry_time']))
        except (ValueError, TypeError):
            pass
    if not times:
        return

    from_time = min(times) - timedelta(minutes=5)
    to_time = max(times) + timedelta(minutes=5)

    try:
        transactions = client.get_transactions(from_time, to_time, tx_type="ORDER_FILL")
    except Exception as e:
        print(f"  [WARN] Could not fetch transactions: {e}")
        return

    # Filter to MARKET_ORDER fills only (trade entries, not SL/TP exits)
    entry_fills = [tx for tx in transactions if tx.get('reason') == 'MARKET_ORDER']
    print(f"  Fetched {len(entry_fills)} entry fills from OANDA transactions")

    for trade in entry_trades:
        try:
            trade_time = datetime.fromisoformat(trade['entry_time'])
        except (ValueError, TypeError):
            continue

        pair = trade.get('pair', '')
        direction = trade.get('direction', '').upper()

        for tx in entry_fills:
            if tx.get('instrument', '') != pair:
                continue

            tx_units = float(tx.get('units', 0))
            tx_dir = 'BUY' if tx_units > 0 else 'SELL'
            if tx_dir != direction:
                continue

            tx_time_str = tx.get('time', '')
            try:
                # OANDA format: "2026-02-17T13:00:13.123456789Z"
                clean = tx_time_str.replace('Z', '')
                if '.' in clean:
                    clean = clean.split('.')[0]
                tx_time = datetime.fromisoformat(clean)
                time_diff = abs((trade_time - tx_time).total_seconds())
                if time_diff < 10:
                    # Get tag from tradeOpened.clientExtensions
                    tag = ''
                    trade_opened = tx.get('tradeOpened', {})
                    if trade_opened:
                        ce = trade_opened.get('clientExtensions', {})
                        tag = ce.get('tag', '')
                    if not tag:
                        ce = tx.get('clientExtensions', {})
                        tag = ce.get('tag', '')

                    if tag:
                        trade['instance_id'] = tag
                    break
            except (ValueError, TypeError):
                continue


def check_csv(csv_path: str, do_replay: bool = False, api_key: str = None):
    """
    Parse OANDA CSV, display trade summary, and optionally replay signals
    from all matching strategy instances to verify backtester alignment.
    """
    print("=" * 70)
    print("  OANDA CSV TRADE CHECK")
    print("=" * 70)

    # ── 1. Parse CSV ──
    trades = parse_oanda_csv(csv_path)
    entries = [t for t in trades if t.get('entry_price')]
    exits_only = [t for t in trades if not t.get('entry_price')]

    print(f"\n  Parsed {len(entries)} trade entries, "
          f"{len(exits_only)} exit-only (pre-existing)")

    # Rejected orders
    rejected = defaultdict(int)
    margin_cancel = 0
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('TRANSACTION TYPE', '').strip() == 'MARKET_ORDER_REJECT':
                inst = row.get('INSTRUMENT', '').strip().replace('/', '_')
                rejected[inst] += 1
            if 'INSUFFICIENT_MARGIN' in row.get('DETAILS', ''):
                margin_cancel += 1
    if rejected:
        print(f"\n  Rejected orders:")
        for inst, cnt in sorted(rejected.items()):
            print(f"    {inst}: {cnt}")
    if margin_cancel:
        print(f"  Insufficient margin cancels: {margin_cancel}")

    # Group by pair
    pairs = sorted(set(t['pair'] for t in entries))
    print(f"\n  Pairs traded: {', '.join(pairs)}")

    # ── 2. Trade summary ──
    print(f"\n{'='*70}")
    print("  TRADE SUMMARY")
    print("=" * 70)

    total_pnl = 0
    wins = 0
    losses = 0

    for t in exits_only:
        pnl = t.get('realized_pnl', 0)
        total_pnl += pnl
        marker = "+" if pnl > 0 else "X"
        exit_time = t.get('exit_time', '?')
        print(f"  [{marker}] {t['pair']:<9} {t.get('direction', '?'):<5} "
              f"(pre-existing) {t['exit_reason']} P&L={pnl:+.2f}  @{exit_time}")

    for t in entries:
        pair = t['pair']
        pnl = t.get('realized_pnl', 0)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        elif t.get('exit_reason') != 'OPEN':
            losses += 1

        pip_sz = 0.01 if 'JPY' in pair else 0.0001
        ep = t['entry_price']
        sl = t.get('stop_loss', 0)
        tp = t.get('take_profit', 0)
        sl_p = abs(ep - sl) / pip_sz if sl else 0
        tp_p = abs(tp - ep) / pip_sz if tp else 0
        rr = tp_p / sl_p if sl_p > 0 else 0

        marker = "+" if pnl > 0 else ("." if t.get('exit_reason') == 'OPEN' else "X")
        entry_time_str = t.get('entry_time', '?')
        if 'T' in entry_time_str:
            entry_time_str = entry_time_str[11:16]  # Just HH:MM

        print(f"  [{marker}] {pair:<9} {t['direction']:<5} @ {ep:.5f}  "
              f"SL={sl_p:.0f}p TP={tp_p:.0f}p RR={rr:.1f}  "
              f"{t.get('exit_reason', '?'):<4} P&L={pnl:+.2f}  @{entry_time_str}")

    total_trades = wins + losses
    wr = wins / total_trades if total_trades else 0
    print(f"\n  Results: {wins}W / {losses}L  ({wr:.0%} win rate)")
    print(f"  Net P&L: {total_pnl:+.2f}")

    # ── 3. Signal replay ──
    if not do_replay:
        print(f"\n  Run without --no-replay to compare against backtester signals")
        return

    print(f"\n{'='*70}")
    print("  SIGNAL REPLAY vs LIVE TRADES")
    print("=" * 70)

    # Connect to OANDA API
    resolved_key = ensure_oanda_credentials(api_key)
    from live.oanda_client import OandaClient
    client = OandaClient(api_key=resolved_key)
    if not client.account_id:
        try:
            accts = client.get_accounts()
            if accts:
                client.account_id = accts[0]['id']
        except Exception as e:
            print(f"  [WARN] Could not fetch account list: {e}")
    print(f"  Connected to OANDA API (account: {client.account_id or 'auto'})")

    # Fetch instance tags from OANDA transactions
    _fetch_instance_tags(client, entries)
    tagged = sum(1 for t in entries if t.get('instance_id'))
    print(f"  Instance tags resolved: {tagged}/{len(entries)} trades")

    tf_minutes = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                  'H1': 60, 'H2': 120, 'H4': 240, 'D': 1440}

    total_ok = 0
    total_mismatch = 0
    total_no_tag = 0
    total_no_signal = 0

    for trade in entries:
        instance_id = trade.get('instance_id', '')
        pair = trade.get('pair', '')
        entry_str = trade.get('entry_time', '')
        entry_hhmm = entry_str[11:16] if 'T' in entry_str else entry_str
        direction = trade.get('direction', '').upper()
        trade_entry = trade.get('entry_price', 0)
        trade_sl = trade.get('stop_loss', 0)
        trade_tp = trade.get('take_profit', 0)
        pnl = trade.get('realized_pnl', 0)
        pip_sz = 0.01 if 'JPY' in pair else 0.0001

        if not instance_id:
            total_no_tag += 1
            print(f"\n    [??] {entry_hhmm} {pair} {direction} @ {trade_entry:.5f}  "
                  f"{trade.get('exit_reason', '?')} P&L={pnl:+.2f}")
            print(f"         No instance tag from OANDA - cannot identify strategy")
            continue

        # Load instance config
        config_path = INSTANCES_DIR / instance_id / "config.json"
        if not config_path.exists():
            total_no_tag += 1
            print(f"\n    [??] {entry_hhmm} {pair} {direction} @ {trade_entry:.5f}  "
                  f"({instance_id})")
            print(f"         Instance config not found: {config_path}")
            continue

        with open(config_path) as f:
            config = json.load(f)
        params = config.get('params', {})
        strategy_name = config.get('strategy', '')
        timeframe = config.get('timeframe', 'M15')
        tf_min = tf_minutes.get(timeframe, 15)

        # Load strategy
        try:
            fast = load_strategy(strategy_name)
            fast._pip_size = pip_sz
        except (ValueError, Exception) as e:
            print(f"\n    [!!] {entry_hhmm} {pair} {direction} ({instance_id})")
            print(f"         Failed to load strategy '{strategy_name}': {e}")
            total_mismatch += 1
            continue

        # Compute candle boundary: the open time of the current candle period.
        # The live trader fetches candles right after candle close, so the most
        # recent complete candle has open_time < candle_boundary.
        # Using to_time=candle_boundary gives candles with time < boundary (OANDA exclusive).
        try:
            trade_time = datetime.fromisoformat(entry_str)
        except (ValueError, TypeError):
            continue

        if tf_min >= 60:
            hours = tf_min // 60
            h = (trade_time.hour // hours) * hours
            candle_boundary = trade_time.replace(hour=h, minute=0, second=0, microsecond=0)
        else:
            m = (trade_time.minute // tf_min) * tf_min
            candle_boundary = trade_time.replace(minute=m, second=0, microsecond=0)

        # Fetch 200 candles ending just before candle_boundary.
        # This replicates the live trader's get_candles(count=200) call exactly.
        try:
            df = client.get_candles(pair, timeframe, count=200, to_time=candle_boundary)
            if df is None or df.empty:
                print(f"\n    [!!] {entry_hhmm} {pair} {direction} ({instance_id})")
                print(f"         No candle data from OANDA for to_time={candle_boundary}")
                total_mismatch += 1
                continue
        except Exception as e:
            print(f"\n    [!!] {entry_hhmm} {pair} {direction} ({instance_id})")
            print(f"         Candle fetch error: {e}")
            total_mismatch += 1
            continue

        # Run strategy on the exact same data the live trader had
        try:
            n_raw = fast.precompute_for_dataset(df)
            if n_raw == 0:
                print(f"\n    [!!] {entry_hhmm} {pair} {direction} ({instance_id})")
                print(f"         Strategy produced 0 raw signals on {len(df)} candles")
                total_no_signal += 1
                continue

            highs = df['high'].values.astype(np.float64)
            lows = df['low'].values.astype(np.float64)
            closes = df['close'].values.astype(np.float64)
            days = df.index.dayofweek.values.astype(np.int64)
            sa, _ = fast.get_all_arrays(params, highs, lows, closes, days)
            entry_bars = sa['entry_bars']
        except Exception as e:
            print(f"\n    [!!] {entry_hhmm} {pair} {direction} ({instance_id})")
            print(f"         Strategy execution error: {e}")
            import traceback
            traceback.print_exc()
            total_mismatch += 1
            continue

        # Check for signal on LAST bar only (matching pipeline_adapter.py:114)
        last_bar = len(df) - 1
        signal_found = False

        for i in range(len(entry_bars)):
            if int(entry_bars[i]) != last_bar:
                continue
            sig_dir = 'BUY' if int(sa['directions'][i]) == 1 else 'SELL'
            if sig_dir != direction:
                continue

            sig_entry = float(sa['entry_prices'][i])
            sig_sl = float(sa['sl_prices'][i])
            sig_tp = float(sa['tp_prices'][i])
            signal_found = True

            # Compare SL/TP prices
            e_diff = abs(trade_entry - sig_entry) / pip_sz
            sl_diff = abs(trade_sl - sig_sl) / pip_sz if trade_sl else 0
            tp_diff = abs(trade_tp - sig_tp) / pip_sz if trade_tp else 0

            ok = sl_diff < 0.5 and tp_diff < 0.5
            if ok:
                total_ok += 1
                marker = "OK"
            else:
                total_mismatch += 1
                marker = "!!"

            print(f"\n    [{marker}] {entry_hhmm} {pair} {direction} @ {trade_entry:.5f}  "
                  f"-> {trade.get('exit_reason', '?')} P&L={pnl:+.2f}")
            print(f"         Instance: {instance_id}")
            print(f"         Replay:   {sig_dir} @ {sig_entry:.5f}  "
                  f"SL={sig_sl:.5f}  TP={sig_tp:.5f}")
            print(f"         Live:     {direction} @ {trade_entry:.5f}  "
                  f"SL={trade_sl:.5f}  TP={trade_tp:.5f}")
            print(f"         Diff:     entry={e_diff:.1f}p  "
                  f"SL={sl_diff:.1f}p  TP={tp_diff:.1f}p")

            # Diagnostic dump on mismatch
            if not ok:
                sig_sl_pips = abs(sig_entry - sig_sl) / pip_sz
                sig_tp_pips = abs(sig_tp - sig_entry) / pip_sz
                live_sl_pips = abs(trade_entry - trade_sl) / pip_sz if trade_sl else 0
                live_tp_pips = abs(trade_tp - trade_entry) / pip_sz if trade_tp else 0
                print(f"         --- DIAGNOSTIC ---")
                print(f"         Replay SL={sig_sl_pips:.1f}p  TP={sig_tp_pips:.1f}p")
                print(f"         Live   SL={live_sl_pips:.1f}p  TP={live_tp_pips:.1f}p")
                print(f"         Candles: {len(df)} [{df.index[0]} .. {df.index[-1]}]")
                print(f"         to_time={candle_boundary}")
            break

        if not signal_found:
            total_no_signal += 1
            print(f"\n    [!!] {entry_hhmm} {pair} {direction} @ {trade_entry:.5f}  "
                  f"-> {trade.get('exit_reason', '?')} P&L={pnl:+.2f}")
            print(f"         Instance: {instance_id}")
            print(f"         NO SIGNAL on last bar (bar {last_bar}, "
                  f"time {df.index[last_bar]})")

            # Show what signals exist for debugging
            n_filtered = len(entry_bars)
            if n_filtered > 0:
                bar_diffs = [(abs(int(entry_bars[j]) - last_bar), j)
                             for j in range(n_filtered)]
                bar_diffs.sort()
                print(f"         {n_filtered} signals found, nearest bars:")
                for bd, idx in bar_diffs[:3]:
                    b = int(entry_bars[idx])
                    d = 'BUY' if int(sa['directions'][idx]) == 1 else 'SELL'
                    print(f"           bar {b} ({df.index[b]}) {d} "
                          f"@ {float(sa['entry_prices'][idx]):.5f}  "
                          f"({bd} bars away)")
            else:
                print(f"         0 signals after filtering (from {n_raw} raw)")

    # Summary
    total = len(entries)
    print(f"\n{'='*70}")
    print(f"  REPLAY SUMMARY")
    print(f"{'='*70}")
    print(f"  Trades with entries:    {total}")
    print(f"  Exact match (0p diff):  {total_ok}/{total}")
    print(f"  SL/TP mismatch:        {total_mismatch}/{total}")
    print(f"  No signal on last bar: {total_no_signal}/{total}")
    print(f"  No instance tag:       {total_no_tag}/{total}")

    if total_mismatch == 0 and total_no_signal == 0 and total_no_tag == 0 and total > 0:
        print(f"  [OK] All live trades confirmed by backtester with exact SL/TP match")
    if total_mismatch > 0:
        print(f"  [ALERT] {total_mismatch} trade(s) had SL/TP mismatch - backtester != live!")
    if total_no_signal > 0:
        print(f"  [ALERT] {total_no_signal} trade(s) had no signal on expected bar!")
    if total_no_tag > 0:
        print(f"  [WARN] {total_no_tag} trade(s) had no OANDA instance tag")


# ── Management Replay ──────────────────────────────────────

def replay_management_for_trade(
    entry_price: float,
    direction: str,
    initial_sl: float,
    params: dict,
    candles: pd.DataFrame,
    pair: str,
) -> list:
    """
    Simulate backtester management logic bar-by-bar for a single trade.

    Replicates the management order from numba_backtest.py:
    max_bars → stale → breakeven → trailing/chandelier → partial close.

    Args:
        entry_price: Trade entry price
        direction: 'BUY' or 'SELL'
        initial_sl: Initial stop loss price
        params: Strategy params dict (from instance config)
        candles: OHLC DataFrame starting from entry bar
        pair: Currency pair for pip size

    Returns:
        List of expected management actions with bar_number, action, new_sl
    """
    pip_size = 0.01 if 'JPY' in pair else 0.0001
    is_long = direction.upper() == 'BUY'

    # Extract params (matching pipeline_adapter.py and create_management_arrays)
    use_trailing = params.get('use_trailing', False)
    trail_mode = params.get('trail_mode', 0)
    trail_start_pips = params.get('trail_start_pips', 50)
    trail_step_pips = params.get('trail_step_pips', 10)
    trail_step_pips = min(trail_step_pips, trail_start_pips)
    chandelier_atr_mult = params.get('chandelier_atr_mult', 3.0)

    use_break_even = params.get('use_break_even', False)
    be_trigger_pips = params.get('be_trigger_pips', None)
    be_atr_mult = params.get('be_atr_mult', 0.5)
    be_offset_pips = params.get('be_offset_pips', 5)

    use_partial_close = params.get('use_partial_close', False)
    partial_target_pips = params.get('partial_target_pips', 20)

    stale_exit_bars = params.get('stale_exit_bars', 0)
    max_bars_in_trade = params.get('max_bars_in_trade', 0)
    param_atr_pips = params.get('atr_pips', 35.0)

    # For BE trigger: use fixed if available, else compute from atr
    if be_trigger_pips is not None:
        trigger_pips = be_trigger_pips
    else:
        trigger_pips = be_atr_mult * param_atr_pips

    # State
    current_sl = initial_sl
    trail_high = 0.0
    trail_active = False
    be_triggered = False
    partial_done = False

    expected_actions = []

    # Bar 0 is entry bar — skip (backtester does `continue`)
    for bar_idx in range(1, len(candles)):
        bar = candles.iloc[bar_idx]
        bar_high = float(bar['high'])
        bar_low = float(bar['low'])
        bar_close = float(bar['close'])
        bar_time = str(candles.index[bar_idx])
        bars = bar_idx

        # ── Max bars exit ──
        if max_bars_in_trade > 0 and bars >= max_bars_in_trade:
            expected_actions.append({
                'bar_number': bars, 'bar_time': bar_time,
                'action': 'max_bars_exit', 'expected_sl': current_sl,
            })
            break

        # ── Stale exit ──
        if stale_exit_bars > 0 and bars >= stale_exit_bars:
            half_r = param_atr_pips * pip_size * 0.5
            move = (bar_close - entry_price) if is_long else (entry_price - bar_close)
            if move < half_r:
                expected_actions.append({
                    'bar_number': bars, 'bar_time': bar_time,
                    'action': 'stale_exit', 'expected_sl': current_sl,
                })
                break

        new_sl = None

        # ── Breakeven ──
        if use_break_even and not be_triggered:
            be_trigger_dist = trigger_pips * pip_size
            effective_offset = min(be_offset_pips, trigger_pips)

            if is_long:
                if bar_high - entry_price >= be_trigger_dist:
                    be_triggered = True
                    be_sl = entry_price + effective_offset * pip_size
                    if be_sl > current_sl:
                        new_sl = be_sl
                        expected_actions.append({
                            'bar_number': bars, 'bar_time': bar_time,
                            'action': 'breakeven',
                            'old_sl': current_sl, 'new_sl': new_sl,
                        })
            else:
                if entry_price - bar_low >= be_trigger_dist:
                    be_triggered = True
                    be_sl = entry_price - effective_offset * pip_size
                    if be_sl < current_sl:
                        new_sl = be_sl
                        expected_actions.append({
                            'bar_number': bars, 'bar_time': bar_time,
                            'action': 'breakeven',
                            'old_sl': current_sl, 'new_sl': new_sl,
                        })

        # ── Trailing (mode 0: fixed pip) ──
        if use_trailing and trail_mode == 0:
            trail_start = trail_start_pips * pip_size
            trail_step = trail_step_pips * pip_size

            if is_long:
                profit = bar_high - entry_price
                if not trail_active and profit >= trail_start:
                    trail_active = True
                    trail_high = bar_high
                if trail_active:
                    if bar_high > trail_high:
                        trail_high = bar_high
                    trail_sl = trail_high - trail_step
                    if trail_sl > current_sl and (new_sl is None or trail_sl > new_sl):
                        new_sl = trail_sl
                        expected_actions.append({
                            'bar_number': bars, 'bar_time': bar_time,
                            'action': 'trailing',
                            'old_sl': current_sl, 'new_sl': new_sl,
                        })
            else:
                profit = entry_price - bar_low
                if not trail_active and profit >= trail_start:
                    trail_active = True
                    trail_high = bar_low
                if trail_active:
                    if trail_high == 0.0 or bar_low < trail_high:
                        trail_high = bar_low
                    trail_sl = trail_high + trail_step
                    if trail_sl < current_sl and (new_sl is None or trail_sl < new_sl):
                        new_sl = trail_sl
                        expected_actions.append({
                            'bar_number': bars, 'bar_time': bar_time,
                            'action': 'trailing',
                            'old_sl': current_sl, 'new_sl': new_sl,
                        })

        # ── Trailing (mode 1: chandelier) ──
        elif use_trailing and trail_mode == 1:
            ch_dist = chandelier_atr_mult * param_atr_pips * pip_size

            if is_long:
                if bar_high > trail_high or trail_high == 0.0:
                    trail_high = bar_high
                trail_sl = trail_high - ch_dist
                if trail_sl > current_sl and (new_sl is None or trail_sl > new_sl):
                    new_sl = trail_sl
                    trail_active = True
                    expected_actions.append({
                        'bar_number': bars, 'bar_time': bar_time,
                        'action': 'chandelier',
                        'old_sl': current_sl, 'new_sl': new_sl,
                    })
            else:
                if trail_high == 0.0 or bar_low < trail_high:
                    trail_high = bar_low
                trail_sl = trail_high + ch_dist
                if trail_sl < current_sl and (new_sl is None or trail_sl < new_sl):
                    new_sl = trail_sl
                    trail_active = True
                    expected_actions.append({
                        'bar_number': bars, 'bar_time': bar_time,
                        'action': 'chandelier',
                        'old_sl': current_sl, 'new_sl': new_sl,
                    })

        # Apply SL change
        if new_sl is not None:
            current_sl = new_sl

        # ── Partial close ──
        if use_partial_close and not partial_done:
            partial_target = partial_target_pips * pip_size
            if is_long:
                if bar_high - entry_price >= partial_target:
                    partial_done = True
                    expected_actions.append({
                        'bar_number': bars, 'bar_time': bar_time,
                        'action': 'partial_close',
                    })
            else:
                if entry_price - bar_low >= partial_target:
                    partial_done = True
                    expected_actions.append({
                        'bar_number': bars, 'bar_time': bar_time,
                        'action': 'partial_close',
                    })

    return expected_actions


def check_management(instance_id: str, api_key: str = None):
    """
    Validate live management actions against backtest replay.

    Loads mgmt_actions.jsonl from the instance state dir, replays
    management bar-by-bar for each trade, and compares per-bar SL.
    """
    idir = INSTANCES_DIR / instance_id

    config_path = idir / "config.json"
    if not config_path.exists():
        print(f"  [ERROR] No config.json for {instance_id}")
        return

    with open(config_path) as f:
        config = json.load(f)

    params = config.get('params', {})
    pair = config.get('pair', '')
    timeframe = config.get('timeframe', 'M15')
    pip_size = 0.01 if 'JPY' in pair else 0.0001

    # Check if any management features are enabled
    has_mgmt = any([
        params.get('use_trailing', False),
        params.get('use_break_even', False),
        params.get('use_partial_close', False),
        params.get('stale_exit_bars', 0) > 0,
        params.get('max_bars_in_trade', 0) > 0,
    ])
    if not has_mgmt:
        print(f"  [INFO] {instance_id}: No management features enabled in params")
        return

    # Load mgmt actions log
    log_path = idir / "state" / "mgmt_actions.jsonl"
    live_actions = []
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        live_actions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # Load trade history
    history_path = idir / "state" / "trade_history.json"
    trades = []
    if history_path.exists():
        with open(history_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                trades = data
            elif isinstance(data, dict):
                trades = data.get("trades", data.get("trade_history", []))

    if not trades:
        print(f"  [INFO] No trade history for {instance_id}")
        return

    # Group live actions by trade_id
    live_by_trade = defaultdict(list)
    for a in live_actions:
        live_by_trade[a.get('trade_id', '')].append(a)

    print(f"\n{'='*70}")
    print(f"  MANAGEMENT VALIDATION: {instance_id}")
    print(f"{'='*70}")
    print(f"  Strategy: {config.get('strategy', '?')}")
    print(f"  Pair/TF:  {pair} {timeframe}")
    print(f"  Params:   trail_mode={params.get('trail_mode', 0)}, "
          f"use_trailing={params.get('use_trailing', False)}, "
          f"use_break_even={params.get('use_break_even', False)}, "
          f"use_partial={params.get('use_partial_close', False)}")
    print(f"  Stale bars: {params.get('stale_exit_bars', 0)}, "
          f"Max bars: {params.get('max_bars_in_trade', 0)}")
    print(f"  Live mgmt actions logged: {len(live_actions)}")
    print(f"  Trades with mgmt data: {len(live_by_trade)}")

    # Connect to OANDA for candle data
    resolved_key = ensure_oanda_credentials(api_key)
    from live.oanda_client import OandaClient
    client = OandaClient(api_key=resolved_key)
    if not client.account_id:
        try:
            accts = client.get_accounts()
            if accts:
                client.account_id = accts[0]['id']
        except Exception as e:
            print(f"  [WARN] Could not fetch account list: {e}")

    tf_minutes = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
                  'H1': 60, 'H2': 120, 'H4': 240, 'D': 1440}

    total_sl_match = 0
    total_sl_mismatch = 0
    total_action_match = 0
    total_action_mismatch = 0
    trades_checked = 0

    for trade in trades:
        tid = trade.get('trade_id', '')
        entry_str = trade.get('entry_time', '')
        exit_str = trade.get('exit_time', '')
        entry_price = trade.get('entry_price', 0)
        direction = trade.get('direction', '')
        initial_sl = trade.get('stop_loss', 0)

        if not entry_str or not entry_price or not initial_sl:
            continue

        # Parse times
        try:
            entry_time = datetime.fromisoformat(entry_str)
            exit_time = datetime.fromisoformat(exit_str) if exit_str else datetime.utcnow()
        except (ValueError, TypeError):
            continue

        # Fetch candle data from entry to exit
        try:
            candles = client.get_candles_range(
                pair, timeframe,
                entry_time - timedelta(minutes=1),
                exit_time + timedelta(hours=1),
            )
        except Exception as e:
            print(f"    [!!] Trade {tid}: Failed to fetch candles: {e}")
            continue

        if candles is None or candles.empty:
            continue

        # Find entry bar in candle data
        entry_ts = pd.Timestamp(entry_time)
        if entry_ts.tzinfo is not None:
            entry_ts = entry_ts.tz_localize(None)

        candle_times = candles.index
        if candle_times.tzinfo is not None or (len(candle_times) > 0 and hasattr(candle_times[0], 'tzinfo') and candle_times[0].tzinfo is not None):
            candle_times = candle_times.tz_localize(None)

        diffs = abs(candle_times - entry_ts)
        entry_bar_idx = diffs.argmin()

        # Slice from entry onward
        trade_candles = candles.iloc[entry_bar_idx:]
        if len(trade_candles) < 2:
            continue

        # Replay management
        expected = replay_management_for_trade(
            entry_price, direction, initial_sl,
            params, trade_candles, pair,
        )

        live = live_by_trade.get(tid, [])
        trades_checked += 1

        # Compare: index both by bar_number
        exp_sl_by_bar = {a['bar_number']: a for a in expected if 'new_sl' in a}
        live_sl_by_bar = {a['bar_number']: a for a in live if 'new_sl' in a}
        exp_close_bars = {a['bar_number'] for a in expected if a['action'] in ('stale_exit', 'max_bars_exit', 'partial_close')}
        live_close_bars = {a['bar_number'] for a in live if a['action'] in ('stale_exit', 'max_bars_exit', 'partial_close')}

        all_sl_bars = sorted(set(exp_sl_by_bar.keys()) | set(live_sl_by_bar.keys()))
        all_close_bars = sorted(set(exp_close_bars) | set(live_close_bars))

        entry_hhmm = entry_str[11:16] if 'T' in entry_str else entry_str
        has_issues = False

        # SL modification comparison
        for bar in all_sl_bars:
            exp = exp_sl_by_bar.get(bar)
            liv = live_sl_by_bar.get(bar)

            if exp and liv:
                sl_diff = abs(exp['new_sl'] - liv['new_sl']) / pip_size
                if sl_diff < 0.5:
                    total_sl_match += 1
                else:
                    total_sl_mismatch += 1
                    has_issues = True
                    print(f"\n    [!!] {entry_hhmm} {tid} bar {bar}: SL MISMATCH")
                    print(f"         Replay: {exp['action']} SL={exp['new_sl']:.5f}")
                    print(f"         Live:   {liv['action']} SL={liv['new_sl']:.5f}")
                    print(f"         Diff:   {sl_diff:.1f} pips")
            elif exp and not liv:
                total_sl_mismatch += 1
                has_issues = True
                print(f"\n    [!!] {entry_hhmm} {tid} bar {bar}: MISSING in live log")
                print(f"         Replay expected: {exp['action']} SL={exp['new_sl']:.5f}")
            elif liv and not exp:
                total_sl_mismatch += 1
                has_issues = True
                print(f"\n    [!!] {entry_hhmm} {tid} bar {bar}: EXTRA in live log")
                print(f"         Live had: {liv['action']} SL={liv['new_sl']:.5f}")

        # Close action comparison
        for bar in all_close_bars:
            exp_actions = {a['action'] for a in expected if a['bar_number'] == bar and a['action'] in ('stale_exit', 'max_bars_exit', 'partial_close')}
            liv_actions = {a['action'] for a in live if a['bar_number'] == bar and a['action'] in ('stale_exit', 'max_bars_exit', 'partial_close')}

            if exp_actions == liv_actions:
                total_action_match += 1
            else:
                total_action_mismatch += 1
                has_issues = True
                print(f"\n    [!!] {entry_hhmm} {tid} bar {bar}: CLOSE ACTION MISMATCH")
                print(f"         Replay: {exp_actions or 'none'}")
                print(f"         Live:   {liv_actions or 'none'}")

        if not has_issues and (all_sl_bars or all_close_bars):
            n_actions = len(all_sl_bars) + len(all_close_bars)
            print(f"    [OK] {entry_hhmm} {tid}: {n_actions} management action(s) matched")

    # Summary
    print(f"\n{'='*70}")
    print(f"  MANAGEMENT VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Trades checked:          {trades_checked}")
    print(f"  SL modifications match:  {total_sl_match}")
    print(f"  SL modifications differ: {total_sl_mismatch}")
    print(f"  Close actions match:     {total_action_match}")
    print(f"  Close actions differ:    {total_action_mismatch}")

    if total_sl_mismatch == 0 and total_action_mismatch == 0 and trades_checked > 0:
        print(f"  [OK] All management actions match backtest replay!")
    elif total_sl_mismatch > 0:
        print(f"  [ALERT] {total_sl_mismatch} SL modification(s) differ from backtester!")
    if trades_checked == 0:
        print(f"  [INFO] No trades with management data to validate")


def main():
    parser = argparse.ArgumentParser(description="Check live trades against backtest expectations")
    parser.add_argument("--instance", "-i", help="Instance ID to check")
    parser.add_argument("--all", "-a", action="store_true", help="Check all instances")
    parser.add_argument("--csv", help="Path to OANDA transactions CSV export")
    parser.add_argument("--last", "-n", type=int, default=0, help="Only check last N trades")
    parser.add_argument("--replay", "-r", action="store_true",
                       help="Replay strategy on historical data to verify signals match")
    parser.add_argument("--no-replay", action="store_true",
                       help="Skip signal replay in CSV mode (just show summary)")
    parser.add_argument("--mgmt", "-m", action="store_true",
                       help="Validate management actions (trailing/BE/etc) against backtest replay")
    parser.add_argument("--api-key", help="OANDA API key (overrides .env)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # CSV mode: parse OANDA export and check trades
    # Replay is ON by default for CSV mode (the whole point is backtester comparison)
    if args.csv:
        do_replay = not args.no_replay
        check_csv(args.csv, do_replay=do_replay, api_key=args.api_key)
        return

    instances_to_check = []

    if args.instance:
        instances_to_check = [args.instance]
    elif args.all and STRATEGIES_FILE.exists():
        with open(STRATEGIES_FILE) as f:
            data = json.load(f)
        instances_to_check = [s["id"] for s in data.get("strategies", []) if s.get("enabled", True)]
    else:
        parser.print_help()
        sys.exit(1)

    all_results = {}
    for iid in instances_to_check:
        all_results[iid] = check_instance(iid, last_n=args.last, do_replay=args.replay)

        # Management validation (separate pass)
        if args.mgmt:
            check_management(iid, api_key=args.api_key)

    if args.json:
        # Convert non-serializable types
        print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
