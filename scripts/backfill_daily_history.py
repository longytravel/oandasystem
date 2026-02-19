"""
Backfill daily_history.json from existing trade_history.json files.

For each instance that has trade_history.json, aggregates closed trades by date
and writes daily_history.json. Safe to run multiple times — skips dates that
already exist in the history.

Usage:
    python scripts/backfill_daily_history.py
"""
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).parent.parent
INSTANCES_DIR = ROOT / "instances"
STRATEGIES_FILE = ROOT / "deploy" / "strategies.json"


def read_json_safe(path: Path):
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError, OSError):
        pass
    return None


def backfill_instance(instance_dir: Path) -> int:
    """Backfill daily_history.json for one instance. Returns number of days added."""
    state_dir = instance_dir / "state"
    trade_file = state_dir / "trade_history.json"
    history_file = state_dir / "daily_history.json"

    # Read trade history
    data = read_json_safe(trade_file)
    if not data:
        return 0

    trades = data if isinstance(data, list) else data.get("trades", data.get("trade_history", []))
    if not trades:
        return 0

    # Load existing daily history
    existing = read_json_safe(history_file)
    if not isinstance(existing, list):
        existing = []
    existing_dates = {e["date"] for e in existing}

    # Aggregate trades by exit date
    daily = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0})

    for trade in trades:
        exit_time = trade.get("exit_time")
        if not exit_time:
            continue
        try:
            dt = datetime.fromisoformat(str(exit_time).replace("Z", "+00:00"))
            day_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        pnl = float(trade.get("realized_pnl", 0))
        daily[day_str]["pnl"] += pnl
        daily[day_str]["trades"] += 1
        if pnl > 0:
            daily[day_str]["wins"] += 1
        elif pnl < 0:
            daily[day_str]["losses"] += 1

    # Merge new days into history
    added = 0
    for day_str, stats in sorted(daily.items()):
        if day_str in existing_dates:
            continue
        existing.append({
            "date": day_str,
            "pnl": round(stats["pnl"], 4),
            "trades": stats["trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "balance": 0.0,  # Unknown from trade history alone
            "max_dd": 0.0,
        })
        added += 1

    if added > 0:
        existing.sort(key=lambda x: x["date"])
        state_dir.mkdir(parents=True, exist_ok=True)
        tmp = history_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f, indent=2)
        os.replace(str(tmp), str(history_file))

    return added


def main():
    # Get instance IDs from strategies.json
    data = read_json_safe(STRATEGIES_FILE)
    if not data:
        print(f"Could not read {STRATEGIES_FILE}")
        return

    strategies = data.get("strategies", [])
    total_added = 0
    total_instances = 0

    for strat in strategies:
        sid = strat["id"]
        instance_dir = INSTANCES_DIR / sid

        if not instance_dir.exists():
            continue

        added = backfill_instance(instance_dir)
        if added > 0:
            print(f"  {sid}: +{added} days")
            total_added += added
            total_instances += 1

    print(f"\nDone: {total_added} days added across {total_instances} instances")


if __name__ == "__main__":
    main()
