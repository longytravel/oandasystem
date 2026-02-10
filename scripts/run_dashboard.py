#!/usr/bin/env python
"""
Start the OANDA Trading Dashboard web server.

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8080 --host 0.0.0.0
"""
import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    # Load default port from strategies.json
    strategies_file = project_root / "deploy" / "strategies.json"
    default_port = 8080
    if strategies_file.exists():
        try:
            with open(strategies_file) as f:
                config = json.load(f)
            default_port = config.get("dashboard", {}).get("port", 8080)
        except (json.JSONDecodeError, IOError):
            pass

    parser = argparse.ArgumentParser(description="OANDA Trading Dashboard")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port (default: {default_port})")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev mode)")

    args = parser.parse_args()

    import uvicorn
    from dashboard.app import app  # noqa: F401

    print(f"\n{'=' * 50}")
    print(f"  OANDA Trading Dashboard")
    print(f"  http://{args.host}:{args.port}")
    print(f"{'=' * 50}\n")

    uvicorn.run(
        "dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
