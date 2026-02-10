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

    # Extract test_months from config for trade frequency calculation
    pipeline_config = state.get("config", {})
    test_months = pipeline_config.get("walkforward", {}).get("test_months", 6)

    # Build expectations from best_candidate metrics
    forward_trades = best.get("forward_trades", 0)
    expectations = {
        "win_rate": round(best.get("forward_win_rate", 0), 4),
        "sharpe": round(best.get("forward_sharpe", 0), 3),
        "profit_factor": round(best.get("forward_profit_factor", 0), 3),
        "avg_trades_per_month": round(forward_trades / max(test_months, 1), 1),
        "max_drawdown_pct": round(best.get("forward_max_dd", 0), 2),
        "forward_return_pct": round(best.get("forward_return", 0), 2),
        "back_win_rate": round(best.get("back_win_rate", 0), 4),
        "back_sharpe": round(best.get("back_sharpe", 0), 3),
        "forward_trades": forward_trades,
        "forward_months": test_months,
    }

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
        "expectations": expectations,
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
    print(f"  Expectations:")
    print(f"    Win Rate:     {expectations['win_rate']:.1%}")
    print(f"    Sharpe:       {expectations['sharpe']:.3f}")
    print(f"    Profit Factor:{expectations['profit_factor']:.3f}")
    print(f"    Trades/Month: {expectations['avg_trades_per_month']:.1f}")
    print(f"    Max DD:       {expectations['max_drawdown_pct']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Export pipeline params to JSON config")
    parser.add_argument("--run", required=True, help="Pipeline run ID (e.g., GBP_USD_M15_20260210_063223)")
    parser.add_argument("--output", "-o", help="Output JSON path (default: config/live_{pair}_{tf}.json)")

    args = parser.parse_args()
    output = Path(args.output) if args.output else None
    export_params(args.run, output)


if __name__ == "__main__":
    main()
