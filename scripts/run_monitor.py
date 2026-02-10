#!/usr/bin/env python
"""
Monitor all trading instances via health.json heartbeat files.

Scans instances/*/health.json and alerts via Telegram if:
- Heartbeat is older than threshold (instance down)
- Consecutive errors exceed threshold
- Sends daily summary on request

Usage:
    python scripts/run_monitor.py --instances-dir instances/
    python scripts/run_monitor.py --instances-dir instances/ --once
    python scripts/run_monitor.py --instances-dir instances/ --summary
"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


def scan_instances(instances_dir: Path) -> list:
    """Scan all instance health files and return status list."""
    statuses = []
    for health_file in sorted(instances_dir.glob("*/health.json")):
        try:
            with open(health_file) as f:
                data = json.load(f)

            ts = datetime.fromisoformat(data["timestamp"])
            # Ensure timezone-aware
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age = datetime.now(timezone.utc) - ts
            data["_age_seconds"] = age.total_seconds()
            data["_health_file"] = str(health_file)
            statuses.append(data)
        except Exception as e:
            statuses.append({
                "instance_id": health_file.parent.name,
                "status": "unreadable",
                "errors": -1,
                "_age_seconds": -1,
                "_health_file": str(health_file),
                "_error": str(e),
            })
    return statuses


def check_alerts(statuses: list, stale_seconds: int = 300, max_errors: int = 3) -> list:
    """Check statuses and return list of alert messages."""
    alerts = []
    for s in statuses:
        instance = s.get("instance_id", "unknown")
        age = s.get("_age_seconds", -1)
        errors = s.get("errors", 0)
        status = s.get("status", "unknown")

        if status == "unreadable":
            alerts.append(f"CRITICAL: {instance} - health.json unreadable: {s.get('_error', '')}")
        elif age > stale_seconds:
            minutes = age / 60
            alerts.append(f"DOWN: {instance} - no heartbeat for {minutes:.1f} min (status: {status})")
        elif errors > max_errors:
            alerts.append(f"ERRORS: {instance} - {errors} consecutive errors")
        elif status == "error":
            alerts.append(f"ERROR: {instance} - status is 'error' ({errors} errors)")

    return alerts


def format_summary(statuses: list) -> str:
    """Format a summary of all instances."""
    if not statuses:
        return "No instances found."

    lines = ["TRADING INSTANCES SUMMARY", "=" * 40]
    for s in statuses:
        instance = s.get("instance_id", "unknown")
        status = s.get("status", "?")
        positions = s.get("positions", "?")
        age = s.get("_age_seconds", -1)
        errors = s.get("errors", 0)

        if age >= 0:
            age_str = f"{age / 60:.0f}m ago"
        else:
            age_str = "N/A"

        icon = "OK" if status == "running" and age < 300 else "!!"
        lines.append(f"  [{icon}] {instance}: {status} | {positions} pos | heartbeat {age_str} | errors: {errors}")

    lines.append("=" * 40)
    return "\n".join(lines)


def send_telegram_alert(message: str):
    """Send alert via Telegram (uses existing alerts module)."""
    try:
        from live.alerts import TelegramAlerts
        alerts = TelegramAlerts(instance_id="monitor")
        if alerts.enabled:
            alerts.custom_alert("MONITOR ALERT", message)
            return True
    except Exception as e:
        logger.warning(f"Telegram alert failed: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Monitor trading instances")
    parser.add_argument("--instances-dir", default="instances", help="Base instances directory")
    parser.add_argument("--stale-seconds", type=int, default=300, help="Heartbeat stale threshold (seconds)")
    parser.add_argument("--max-errors", type=int, default=3, help="Max consecutive errors before alert")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--summary", action="store_true", help="Print summary and exit")
    parser.add_argument("--telegram", action="store_true", help="Send alerts via Telegram")

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    instances_dir = Path(args.instances_dir)
    if not instances_dir.exists():
        logger.error(f"Instances directory not found: {instances_dir}")
        sys.exit(1)

    if args.summary:
        statuses = scan_instances(instances_dir)
        print(format_summary(statuses))
        return

    logger.info(f"Monitoring {instances_dir} (interval: {args.interval}s, stale: {args.stale_seconds}s)")

    while True:
        statuses = scan_instances(instances_dir)
        alerts = check_alerts(statuses, args.stale_seconds, args.max_errors)

        if alerts:
            for alert in alerts:
                logger.warning(alert)
                if args.telegram:
                    send_telegram_alert(alert)
        else:
            running = sum(1 for s in statuses if s.get("status") == "running")
            logger.info(f"All OK - {running}/{len(statuses)} instances running")

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
