#!/usr/bin/env python
"""
Export best candidate params from a pipeline run to standalone JSON config.

This decouples the VPS deployment from local pipeline result directories.
Output format matches existing config/live_*.json (works with --params-file).

Usage:
    python scripts/export_params.py --run GBP_USD_M15_20260210_063223
    python scripts/export_params.py --run GBP_USD_M15_20260210_063223 --output instances/rsi_v3_GBP_USD_M15/config.json
"""
import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def export_params(run_id: str, output_path: Path = None):
    """Export best candidate from pipeline run to JSON config."""
    run_dir = project_root / "results" / "pipelines" / run_id
    state_file = run_dir / "state.json"

    if not state_file.exists():
        print(f"ERROR: Pipeline run not found: {state_file}")
        sys.exit(1)

    with open(state_file) as f:
        state = json.load(f)

    best = state.get("best_candidate", {})
    if not best or not best.get("params"):
        print(f"ERROR: No best candidate in {run_id}")
        sys.exit(1)

    # Build export config (matches live_GBP_USD_M15.json format)
    config = {
        "run_id": run_id,
        "strategy": state.get("strategy_name", "unknown"),
        "pair": state.get("pair", "GBP_USD"),
        "timeframe": state.get("timeframe", "H1"),
        "score": state.get("final_score", 0),
        "rating": state.get("final_rating", "RED"),
        "rank": best.get("rank", "?"),
        "params": best["params"],
    }

    # Default output path
    if output_path is None:
        output_path = project_root / "config" / f"live_{config['pair']}_{config['timeframe']}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Exported {run_id} -> {output_path}")
    print(f"  Strategy: {config['strategy']}")
    print(f"  Pair: {config['pair']} {config['timeframe']}")
    print(f"  Score: {config['score']}/100 {config['rating']}")
    print(f"  Rank: #{config['rank']}")
    print(f"  Params: {len(config['params'])} parameters")


def main():
    parser = argparse.ArgumentParser(description="Export pipeline params to JSON config")
    parser.add_argument("--run", required=True, help="Pipeline run ID (e.g., GBP_USD_M15_20260210_063223)")
    parser.add_argument("--output", "-o", help="Output JSON path (default: config/live_{pair}_{tf}.json)")

    args = parser.parse_args()
    output = Path(args.output) if args.output else None
    export_params(args.run, output)


if __name__ == "__main__":
    main()
