#!/usr/bin/env python
"""
Verify live trades vs backtester signals.

Loads each active strategy instance, generates signals on today's M15 data,
and compares with actual OANDA transactions from CSV.

Usage:
    python scripts/verify_live_vs_backtest.py --csv path/to/transactions.csv
"""
import sys
import csv
import json
import importlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from live.oanda_client import OandaClient
from live.pipeline_adapter import PipelineStrategyAdapter


# ── Strategy registry (same as run_live.py) ──────────────────────
PIPELINE_STRATEGIES = {
    "rsi_v1": ("strategies.archive.rsi_full", "RSIDivergenceFullFast"),
    "rsi_v3": ("strategies.rsi_full_v3", "RSIDivergenceFullFastV3"),
    "rsi_v4": ("strategies.rsi_full_v4", "RSIDivergenceFullFastV4"),
    "rsi_v5": ("strategies.rsi_full_v5", "RSIDivergenceFullFastV5"),
    "ema_cross": ("strategies.ema_cross_ml", "EMACrossMLStrategy"),
    "fair_price_ma": ("strategies.fair_price_ma", "FairPriceMAStrategy"),
    "donchian_breakout": ("strategies.donchian_breakout", "DonchianBreakoutStrategy"),
    "bollinger_squeeze": ("strategies.bollinger_squeeze", "BollingerSqueezeStrategy"),
    "london_breakout": ("strategies.london_breakout", "LondonBreakoutStrategy"),
    "stochastic_adx": ("strategies.stochastic_adx", "StochasticADXStrategy"),
}


def parse_csv_trades(csv_path: str) -> list:
    """Parse OANDA transaction CSV into completed trade entries."""
    trades = []
    pending_entries = {}  # Track open entries by instrument

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        tx_type = row.get('TRANSACTION TYPE', '').strip()
        details = row.get('DETAILS', '').strip()
        instrument = row.get('INSTRUMENT', '').strip()
        price = row.get('PRICE', '').strip()
        units = row.get('UNITS', '').strip()
        direction = row.get('DIRECTION', '').strip()
        sl = row.get('STOP LOSS', '').strip()
        tp = row.get('TAKE PROFIT', '').strip()
        pl = row.get('PL', '').strip()
        ts_raw = row.get('TRANSACTION DATE', '').strip()
        ticket = row.get('TICKET', '').strip()

        if not ts_raw:
            continue

        # Parse timestamp: "2026-02-17 13:00:13 -12" -> UTC
        try:
            # Split off timezone offset
            parts = ts_raw.rsplit(' ', 1)
            dt_str = parts[0]
            tz_offset = int(parts[1]) if len(parts) > 1 else 0
            dt_local = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            dt_utc = dt_local - timedelta(hours=tz_offset)
        except Exception:
            continue

        # Track order fills for entries (MARKET_ORDER)
        if tx_type == 'ORDER_FILL' and details == 'MARKET_ORDER':
            if instrument and price:
                inst_key = instrument.replace('/', '_')
                pending_entries[inst_key] = {
                    'ticket': ticket,
                    'instrument': instrument,
                    'pair': inst_key,
                    'entry_time_utc': dt_utc,
                    'entry_price': float(price),
                    'units': float(units) if units else 0,
                    'direction': direction,
                }

        # Track SL/TP orders set ON_FILL (these come right after the entry)
        if tx_type in ('STOP_LOSS_ORDER', 'TAKE_PROFIT_ORDER') and details == 'ON_FILL':
            # Find the most recent entry for any pair
            for key, entry in list(pending_entries.items()):
                if abs((dt_utc - entry['entry_time_utc']).total_seconds()) < 5:
                    if tx_type == 'STOP_LOSS_ORDER' and price:
                        entry['sl_price'] = float(price)
                    elif tx_type == 'TAKE_PROFIT_ORDER' and price:
                        entry['tp_price'] = float(price)

        # Track exits (SL/TP fills)
        if tx_type == 'ORDER_FILL' and details in ('STOP_LOSS_ORDER', 'TAKE_PROFIT_ORDER'):
            if instrument and pl:
                inst_key = instrument.replace('/', '_')
                exit_type = 'SL' if details == 'STOP_LOSS_ORDER' else 'TP'
                exit_info = {
                    'exit_time_utc': dt_utc,
                    'exit_price': float(price) if price else 0,
                    'exit_type': exit_type,
                    'pl': float(pl) if pl else 0,
                }

                # Match with pending entry
                if inst_key in pending_entries:
                    entry = pending_entries.pop(inst_key)
                    entry.update(exit_info)
                    trades.append(entry)
                else:
                    # Exit of pre-existing position
                    trades.append({
                        'ticket': ticket,
                        'instrument': instrument,
                        'pair': inst_key,
                        'entry_time_utc': None,
                        'entry_price': None,
                        'direction': direction,
                        **exit_info,
                    })

    # Add any remaining pending entries (still open)
    for entry in pending_entries.values():
        entry['exit_type'] = 'OPEN'
        entry['pl'] = 0
        trades.append(entry)

    return trades


def load_strategy_instance(strat_key: str, pair: str) -> tuple:
    """Load a strategy instance with its config. Returns (adapter, config) or (None, None)."""
    # Try instance config first
    instance_id_patterns = []

    # Build possible instance IDs
    prefix_map = {
        'rsi_v3': 'rsi_v3', 'rsi_v1': 'rsi_v1',
        'ema_cross': 'ema_v6', 'fair_price_ma': 'fpma',
        'donchian_breakout': 'dch', 'bollinger_squeeze': 'bbsq',
        'london_breakout': 'ldn', 'stochastic_adx': 'stadx',
    }
    prefix = prefix_map.get(strat_key, strat_key)
    instance_id = f"{prefix}_{pair}_M15"

    # Check instance config
    config_path = project_root / "instances" / instance_id / "config.json"
    if not config_path.exists():
        # Try config/ directory
        config_path = project_root / "config" / f"live_{pair}_M15_{instance_id}.json"
    if not config_path.exists():
        # Try config/ with simpler name
        config_path = project_root / "config" / f"live_{pair}_M15.json"

    if not config_path.exists():
        return None, None

    with open(config_path) as f:
        config = json.load(f)

    params = config.get('params', {})
    if not params:
        return None, None

    # Load the FastStrategy class
    if strat_key not in PIPELINE_STRATEGIES:
        return None, None

    mod_path, cls_name = PIPELINE_STRATEGIES[strat_key]
    try:
        mod = importlib.import_module(mod_path)
        fast_cls = getattr(mod, cls_name)
        fast_strategy = fast_cls()
        adapter = PipelineStrategyAdapter(fast_strategy, params, pair=pair)
        return adapter, config
    except Exception as e:
        print(f"  [ERROR] Loading {strat_key} for {pair}: {e}")
        return None, None


def generate_signals_for_pair(pair: str, df: pd.DataFrame, strategies_json: dict) -> list:
    """Generate signals from all active strategy instances for a pair."""
    signals = []

    # Find all enabled strategies for this pair
    for entry in strategies_json.get('strategies', []):
        if entry.get('pair') != pair or not entry.get('enabled', True):
            continue

        strat_key = entry['strategy']
        instance_id = entry['id']
        timeframe = entry.get('timeframe', 'M15')

        if timeframe != 'M15':
            # Skip H1 strategies for M15 data (different candle alignment)
            continue

        adapter, config = load_strategy_instance(strat_key, pair)
        if adapter is None:
            print(f"  [SKIP] {instance_id}: no config/params found")
            continue

        try:
            raw_signals = adapter.generate_signals(df)
            for sig in raw_signals:
                # Get signal for ALL bars, not just last
                pass

            # Instead, run full signal generation to get all signals
            fast = adapter.fast_strategy
            fast._pip_size = adapter._pip_size
            n_signals = fast.precompute_for_dataset(df)

            if n_signals == 0:
                continue

            highs = df['high'].values.astype(np.float64)
            lows = df['low'].values.astype(np.float64)
            closes = df['close'].values.astype(np.float64)
            days = df.index.dayofweek.values.astype(np.int64)

            signal_arrays, mgmt_arrays = fast.get_all_arrays(
                adapter.optimized_params, highs, lows, closes, days
            )

            entry_bars = signal_arrays['entry_bars']
            if len(entry_bars) == 0:
                continue

            for i in range(len(entry_bars)):
                bar_idx = int(entry_bars[i])
                if bar_idx < 0 or bar_idx >= len(df):
                    continue

                candle_time = df.index[bar_idx]
                direction = int(signal_arrays['directions'][i])
                entry_price = float(signal_arrays['entry_prices'][i])
                sl_price = float(signal_arrays['sl_prices'][i])
                tp_price = float(signal_arrays['tp_prices'][i])

                pip_size = adapter._pip_size
                sl_pips = abs(entry_price - sl_price) / pip_size
                tp_pips = abs(tp_price - entry_price) / pip_size
                rr = tp_pips / sl_pips if sl_pips > 0 else 0

                signals.append({
                    'instance_id': instance_id,
                    'strategy': strat_key,
                    'candle_time': candle_time,
                    'bar_idx': bar_idx,
                    'direction': 'Buy' if direction == 1 else 'Sell',
                    'entry_price': entry_price,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'sl_pips': round(sl_pips, 1),
                    'tp_pips': round(tp_pips, 1),
                    'rr': round(rr, 1),
                })

        except Exception as e:
            print(f"  [ERROR] {instance_id}: {e}")
            import traceback
            traceback.print_exc()

    return signals


def match_signals_to_trades(signals: list, trades: list, pair: str) -> list:
    """Match backtester signals to actual live trades by time and direction."""
    matches = []
    pair_trades = [t for t in trades if t.get('pair') == pair and t.get('entry_time_utc')]

    for trade in pair_trades:
        entry_utc = trade['entry_time_utc']
        trade_dir = trade.get('direction', '')

        # Find the M15 candle that this trade was triggered on
        # Live system fires ~30-60s after candle close, so round down to M15 boundary
        candle_minute = (entry_utc.minute // 15) * 15
        candle_time = entry_utc.replace(minute=candle_minute, second=0, microsecond=0)

        # Match signals within +/- 1 candle
        best_match = None
        best_score = float('inf')

        for sig in signals:
            sig_time = sig['candle_time']
            if hasattr(sig_time, 'to_pydatetime'):
                sig_time = sig_time.to_pydatetime()
            if sig_time.tzinfo:
                sig_time = sig_time.replace(tzinfo=None)

            time_diff = abs((sig_time - candle_time).total_seconds())

            # Must be same direction
            dir_match = (
                (trade_dir == 'Buy' and sig['direction'] == 'Buy') or
                (trade_dir == 'Sell' and sig['direction'] == 'Sell')
            )

            if dir_match and time_diff <= 900:  # Within 1 candle (15 min)
                price_diff = abs(trade.get('entry_price', 0) - sig['entry_price'])
                score = time_diff + price_diff * 100000
                if score < best_score:
                    best_score = score
                    best_match = sig

        # Compute SL/TP comparison
        if best_match:
            pip_size = 0.01 if 'JPY' in pair else 0.0001
            sl_diff = abs(trade.get('sl_price', 0) - best_match['sl_price']) / pip_size
            tp_diff = abs(trade.get('tp_price', 0) - best_match['tp_price']) / pip_size
            entry_diff = abs(trade.get('entry_price', 0) - best_match['entry_price']) / pip_size

            matches.append({
                'trade': trade,
                'signal': best_match,
                'entry_diff_pips': round(entry_diff, 1),
                'sl_diff_pips': round(sl_diff, 1),
                'tp_diff_pips': round(tp_diff, 1),
                'time_diff_sec': int(abs((
                    (best_match['candle_time'].to_pydatetime().replace(tzinfo=None)
                     if hasattr(best_match['candle_time'], 'to_pydatetime')
                     else best_match['candle_time'].replace(tzinfo=None))
                    - candle_time
                ).total_seconds())),
            })
        else:
            matches.append({
                'trade': trade,
                'signal': None,
                'note': 'NO MATCHING SIGNAL FOUND',
            })

    return matches


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify live trades vs backtester signals")
    parser.add_argument('--csv', required=True, help='Path to OANDA transactions CSV')
    parser.add_argument('--candles', type=int, default=500, help='Candles to fetch (default: 500)')
    args = parser.parse_args()

    print("=" * 70)
    print("LIVE vs BACKTESTER SIGNAL VERIFICATION")
    print("=" * 70)

    # 1. Parse CSV trades
    print("\n[1] Parsing CSV trades...")
    trades = parse_csv_trades(args.csv)

    # Filter to entries only (with entry time)
    entry_trades = [t for t in trades if t.get('entry_time_utc')]
    exit_only = [t for t in trades if not t.get('entry_time_utc')]

    print(f"  Found {len(entry_trades)} trade entries, {len(exit_only)} exit-only (pre-existing)")

    # Summary
    pairs_traded = set(t['pair'] for t in entry_trades)
    print(f"  Pairs: {', '.join(sorted(pairs_traded))}")

    for t in entry_trades:
        pip_size = 0.01 if 'JPY' in t['pair'] else 0.0001
        sl_pips = abs(t.get('entry_price', 0) - t.get('sl_price', 0)) / pip_size if t.get('sl_price') else 0
        tp_pips = abs(t.get('tp_price', 0) - t.get('entry_price', 0)) / pip_size if t.get('tp_price') else 0
        rr = tp_pips / sl_pips if sl_pips > 0 else 0
        print(f"  {t['entry_time_utc'].strftime('%H:%M')} UTC | {t['pair']:<8} | "
              f"{t['direction']:<4} @ {t['entry_price']:.5f} | "
              f"SL={t.get('sl_price', 0):.5f} ({sl_pips:.0f}p) | "
              f"TP={t.get('tp_price', 0):.5f} ({tp_pips:.0f}p) | "
              f"RR={rr:.1f} | {t.get('exit_type', '?')} P&L={t.get('pl', 0):.2f}")

    # 2. Load strategies.json
    print("\n[2] Loading strategy configurations...")
    strats_file = project_root / 'deploy' / 'strategies.json'
    with open(strats_file) as f:
        strategies_json = json.load(f)

    enabled_count = sum(1 for s in strategies_json['strategies'] if s.get('enabled', True))
    print(f"  {enabled_count} enabled strategy instances")

    # 3. Fetch M15 data for each pair
    print("\n[3] Fetching M15 candle data from OANDA...")
    client = OandaClient()
    if not client.account_id:
        accounts = client.get_accounts()
        client.account_id = accounts[0]['id']

    pair_data = {}
    for pair in sorted(pairs_traded):
        oanda_pair = pair  # Already in OANDA format (underscore)
        try:
            df = client.get_candles(oanda_pair, "M15", count=args.candles)
            if df is not None and not df.empty:
                pair_data[pair] = df
                print(f"  {pair}: {len(df)} candles [{df.index[0]} .. {df.index[-1]}]")
            else:
                print(f"  {pair}: NO DATA")
        except Exception as e:
            print(f"  {pair}: ERROR - {e}")

    # 4. Generate signals from all strategies
    print("\n[4] Generating backtester signals...")
    all_signals = {}
    for pair in sorted(pairs_traded):
        if pair not in pair_data:
            continue
        print(f"\n  --- {pair} ---")
        signals = generate_signals_for_pair(pair, pair_data[pair], strategies_json)
        all_signals[pair] = signals

        # Filter to today's signals
        today_start = datetime(2026, 2, 18, 0, 0)
        today_end = datetime(2026, 2, 19, 0, 0)
        today_signals = []
        for s in signals:
            st = s['candle_time']
            if hasattr(st, 'to_pydatetime'):
                st = st.to_pydatetime()
            if st.tzinfo:
                st = st.replace(tzinfo=None)
            if today_start <= st <= today_end:
                today_signals.append(s)

        print(f"  Total signals in data: {len(signals)}")
        print(f"  Signals today (Feb 18): {len(today_signals)}")

        for s in today_signals:
            st = s['candle_time']
            if hasattr(st, 'to_pydatetime'):
                st = st.to_pydatetime()
            if st.tzinfo:
                st = st.replace(tzinfo=None)
            print(f"    {st.strftime('%H:%M')} | {s['instance_id']:<25} | "
                  f"{s['direction']:<4} @ {s['entry_price']:.5f} | "
                  f"SL={s['sl_price']:.5f} ({s['sl_pips']}p) | "
                  f"TP={s['tp_price']:.5f} ({s['tp_pips']}p) | "
                  f"RR={s['rr']}")

    # 5. Match signals to trades
    print("\n" + "=" * 70)
    print("[5] MATCHING: Live Trades vs Backtester Signals")
    print("=" * 70)

    total_matched = 0
    total_unmatched = 0
    total_sl_ok = 0
    total_tp_ok = 0

    for pair in sorted(pairs_traded):
        if pair not in all_signals:
            continue

        matches = match_signals_to_trades(all_signals[pair], entry_trades, pair)
        if not matches:
            continue

        print(f"\n  === {pair} ===")
        for m in matches:
            trade = m['trade']
            sig = m.get('signal')

            print(f"\n  Live: {trade['entry_time_utc'].strftime('%H:%M')} UTC "
                  f"{trade['direction']} @ {trade['entry_price']:.5f} "
                  f"SL={trade.get('sl_price', 0):.5f} TP={trade.get('tp_price', 0):.5f} "
                  f"-> {trade.get('exit_type', '?')} (P&L: ${trade.get('pl', 0):.2f})")

            if sig:
                total_matched += 1
                entry_ok = m['entry_diff_pips'] < 2.0
                sl_ok = m['sl_diff_pips'] < 2.0
                tp_ok = m['tp_diff_pips'] < 2.0

                if sl_ok:
                    total_sl_ok += 1
                if tp_ok:
                    total_tp_ok += 1

                status = "MATCH" if (sl_ok and tp_ok) else "MISMATCH"
                print(f"  Back: {sig['candle_time']} "
                      f"{sig['direction']} @ {sig['entry_price']:.5f} "
                      f"SL={sig['sl_price']:.5f} TP={sig['tp_price']:.5f} "
                      f"[{sig['instance_id']}]")
                print(f"  Diff: entry={m['entry_diff_pips']:.1f}p "
                      f"SL={m['sl_diff_pips']:.1f}p "
                      f"TP={m['tp_diff_pips']:.1f}p "
                      f"time={m['time_diff_sec']}s "
                      f"-> [{status}]")
            else:
                total_unmatched += 1
                print(f"  Back: *** NO MATCHING SIGNAL FOUND ***")

    # 6. Summary
    print("\n" + "=" * 70)
    print("[6] SUMMARY")
    print("=" * 70)
    total = total_matched + total_unmatched
    print(f"  Trades with entries:  {total}")
    print(f"  Matched to signals:   {total_matched}/{total} "
          f"({100*total_matched/total:.0f}%)" if total > 0 else "")
    print(f"  Unmatched:            {total_unmatched}/{total}")
    if total_matched > 0:
        print(f"  SL within 2 pips:     {total_sl_ok}/{total_matched}")
        print(f"  TP within 2 pips:     {total_tp_ok}/{total_matched}")

    # 7. Rejected orders analysis
    print("\n  --- Rejected Orders ---")
    rejected = defaultdict(int)
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('TRANSACTION TYPE', '').strip() == 'MARKET_ORDER_REJECT':
                inst = row.get('INSTRUMENT', '').strip()
                rejected[inst] += 1
    for inst, count in sorted(rejected.items()):
        print(f"  {inst}: {count} rejections")

    margin_rejects = 0
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'INSUFFICIENT_MARGIN' in row.get('DETAILS', ''):
                margin_rejects += 1
    if margin_rejects:
        print(f"  Insufficient margin cancels: {margin_rejects}")


if __name__ == '__main__':
    main()
