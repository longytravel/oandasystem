"""
Health monitoring for live trading instances.

Writes a heartbeat JSON file that can be monitored by run_monitor.py.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

# Module-level instance dir, set by trader on startup
_instance_dir: Optional[Path] = None


def set_instance_dir(instance_dir: Path):
    """Set the instance directory for heartbeat files."""
    global _instance_dir
    _instance_dir = instance_dir
    _instance_dir.mkdir(parents=True, exist_ok=True)


def write_heartbeat(
    instance_id: str,
    status: str = "running",
    positions: int = 0,
    last_candle: str = "",
    errors: int = 0,
    last_error: str = "",
    instance_dir: Optional[Path] = None,
):
    """
    Write health.json heartbeat file.

    Args:
        instance_id: Unique instance identifier
        status: Current status (running, error, stopped)
        positions: Number of open positions
        last_candle: Timestamp of last processed candle
        errors: Consecutive error count
        last_error: Last error message (truncated to 500 chars)
        instance_dir: Override directory (uses module default if None)
    """
    target_dir = instance_dir or _instance_dir
    if target_dir is None:
        return

    health_file = target_dir / "health.json"
    data = {
        "instance_id": instance_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "positions": positions,
        "last_candle": last_candle,
        "errors": errors,
        "last_error": str(last_error)[:500] if last_error else "",
    }

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(health_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.debug(f"Failed to write health.json: {e}")
