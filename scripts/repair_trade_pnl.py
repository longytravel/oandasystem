"""Repair trades with $0 realized P&L by fetching real data from OANDA.

Scans all instance trade_history.json files, finds trades with
realized_pnl=0.0 and exit_price==entry_price (indicating a failed
get_trade() call), and backfills the correct data from OANDA API.

Usage:
    python scripts/repair_trade_pnl.py              # Dry run (show what would change)
    python scripts/repair_trade_pnl.py --apply       # Apply fixes
    python scripts/repair_trade_pnl.py --apply --recalc-daily  # Also recalculate daily stats
"""
import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from live.oanda_client import OandaClient


def find_broken_trades(instances_dir: Path) -> dict:
    """Find all trades with suspicious $0 P&L across all instances.

    Returns dict: {instance_id: [(index, trade_record), ...]}
    """
    broken = {}

    for instance_dir in sorted(instances_dir.iterdir()):
        if not instance_dir.is_dir() or instance_dir.name.startswith('_'):
            continue

        history_file = instance_dir / "state" / "trade_history.json"
        if not history_file.exists():
            continue

        try:
            with open(history_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        trades = data if isinstance(data, list) else data.get("trades", [])

        for i, trade in enumerate(trades):
            pnl = trade.get("realized_pnl", 0)
            entry = trade.get("entry_price", 0)
            exit_p = trade.get("exit_price", 0)
            trade_id = trade.get("trade_id")

            # Suspicious: $0 P&L AND exit_price == entry_price AND has a trade ID
            if (pnl == 0.0 and trade_id and
                    abs(exit_p - entry) < 1e-8):
                broken.setdefault(instance_dir.name, []).append((i, trade))

    return broken


def repair_from_oanda(client: OandaClient, broken: dict, instances_dir: Path,
                       apply: bool = False) -> dict:
    """Fetch real P&L from OANDA and optionally apply fixes.

    Returns summary of repairs made.
    """
    summary = {"checked": 0, "repaired": 0, "failed": 0, "skipped": 0}

    for instance_id, trade_list in broken.items():
        print(f"\n--- {instance_id} ({len(trade_list)} broken trades) ---")

        history_file = instances_dir / instance_id / "state" / "trade_history.json"

        try:
            with open(history_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"  [ERROR] Cannot read {history_file}")
            continue

        trades = data if isinstance(data, list) else data.get("trades", [])
        modified = False

        for idx, broken_trade in trade_list:
            trade_id = broken_trade.get("trade_id")
            if not trade_id:
                summary["skipped"] += 1
                continue

            summary["checked"] += 1

            try:
                details = client.get_trade(str(trade_id))
                time.sleep(0.1)  # Rate limit courtesy
            except Exception as e:
                print(f"  Trade {trade_id}: API error - {e}")
                summary["failed"] += 1
                continue

            if not details:
                print(f"  Trade {trade_id}: empty response from OANDA")
                summary["failed"] += 1
                continue

            real_pnl = float(details.get("realizedPL", 0))
            real_exit = float(details.get("averageClosePrice", 0))
            state = details.get("state", "")

            # Determine exit reason
            exit_reason = "CLOSED"
            if state == "CLOSED":
                sl_order = details.get("stopLossOrder", {})
                tp_order = details.get("takeProfitOrder", {})
                if sl_order.get("state") == "FILLED":
                    exit_reason = "SL"
                elif tp_order.get("state") == "FILLED":
                    exit_reason = "TP"

            close_time = details.get("closeTime", "")

            old_entry = broken_trade.get("entry_price", 0)
            print(f"  Trade {trade_id}: PnL {real_pnl:+.4f}, "
                  f"exit {real_exit:.5f} (was {old_entry:.5f}), "
                  f"reason={exit_reason}")

            if real_pnl == 0.0 and abs(real_exit - old_entry) < 1e-8:
                print(f"    ^ Still looks broken from OANDA side, skipping")
                summary["skipped"] += 1
                continue

            if apply:
                trades[idx]["realized_pnl"] = real_pnl
                trades[idx]["exit_price"] = real_exit
                trades[idx]["exit_reason"] = exit_reason
                if close_time:
                    trades[idx]["exit_time"] = close_time
                modified = True

            summary["repaired"] += 1

        if apply and modified:
            if isinstance(data, list):
                write_data = trades
            else:
                data["trades"] = trades
                write_data = data

            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(write_data, f, indent=2, default=str)
            print(f"  [SAVED] {history_file}")

    return summary


def recalc_daily_stats(instances_dir: Path):
    """Recalculate daily_stats in position_state.json from today's trade history."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for instance_dir in sorted(instances_dir.iterdir()):
        if not instance_dir.is_dir() or instance_dir.name.startswith('_'):
            continue

        pos_file = instance_dir / "state" / "position_state.json"
        history_file = instance_dir / "state" / "trade_history.json"

        if not pos_file.exists() or not history_file.exists():
            continue

        try:
            with open(history_file) as f:
                hist_data = json.load(f)
            with open(pos_file) as f:
                pos_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        trades = hist_data if isinstance(hist_data, list) else hist_data.get("trades", [])

        # Find today's trades
        gross_profit = 0.0
        gross_loss = 0.0
        wins = 0
        losses = 0

        for t in trades:
            exit_time = t.get("exit_time", "")
            if not exit_time or today not in str(exit_time):
                continue

            pnl = t.get("realized_pnl", 0)
            if pnl > 0:
                gross_profit += pnl
                wins += 1
            elif pnl < 0:
                gross_loss += pnl
                losses += 1

        trades_closed = wins + losses
        if trades_closed == 0:
            continue

        daily = pos_data.get("daily_stats", {})
        old_pnl = daily.get("gross_profit", 0) + daily.get("gross_loss", 0)
        new_pnl = gross_profit + gross_loss

        if abs(old_pnl - new_pnl) < 0.001:
            continue

        daily["gross_profit"] = gross_profit
        daily["gross_loss"] = gross_loss
        daily["wins"] = wins
        daily["losses"] = losses
        daily["trades_closed"] = trades_closed
        pos_data["daily_stats"] = daily

        with open(pos_file, "w", encoding="utf-8") as f:
            json.dump(pos_data, f, indent=2, default=str)

        print(f"  {instance_dir.name}: daily P&L {old_pnl:+.2f} -> {new_pnl:+.2f} "
              f"({wins}W/{losses}L)")


def main():
    parser = argparse.ArgumentParser(description="Repair $0 P&L trades from OANDA")
    parser.add_argument("--apply", action="store_true", help="Apply fixes (default: dry run)")
    parser.add_argument("--recalc-daily", action="store_true",
                        help="Also recalculate daily stats after repair")
    parser.add_argument("--instances-dir", default=None,
                        help="Instances directory (default: instances/)")
    args = parser.parse_args()

    instances_dir = Path(args.instances_dir) if args.instances_dir else ROOT / "instances"

    if not instances_dir.exists():
        print(f"Instances directory not found: {instances_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  TRADE P&L REPAIR TOOL")
    print("=" * 60)
    print(f"  Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"  Instances: {instances_dir}")

    # Find broken trades
    print("\nScanning for broken trades...")
    broken = find_broken_trades(instances_dir)

    total_broken = sum(len(v) for v in broken.values())
    if total_broken == 0:
        print("No broken trades found!")
        return

    print(f"Found {total_broken} broken trades across {len(broken)} instances")

    # Connect to OANDA
    print("\nConnecting to OANDA...")
    client = OandaClient()

    # Repair
    summary = repair_from_oanda(client, broken, instances_dir, apply=args.apply)

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"  Checked:  {summary['checked']}")
    print(f"  Repaired: {summary['repaired']}")
    print(f"  Failed:   {summary['failed']}")
    print(f"  Skipped:  {summary['skipped']}")

    if not args.apply and summary['repaired'] > 0:
        print(f"\n  Run with --apply to save changes")

    # Recalculate daily stats
    if args.apply and args.recalc_daily:
        print(f"\nRecalculating daily stats...")
        recalc_daily_stats(instances_dir)

    print()


if __name__ == "__main__":
    main()
