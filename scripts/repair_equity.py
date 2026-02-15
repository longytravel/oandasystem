"""Repair runs with missing equity data in report_data.json.

Finds pipeline runs where back_equity/forward_equity are empty, re-runs
the backtest for the best candidate, and updates report_data.json.

Usage:
    python scripts/repair_equity.py          # dry-run (show affected runs)
    python scripts/repair_equity.py --fix    # actually repair
"""
import json
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_broken_runs(pipelines_dir: Path):
    """Find runs with empty equity data that have a valid best candidate."""
    broken = []
    for d in sorted(pipelines_dir.iterdir()):
        rd_path = d / 'report_data.json'
        state_path = d / 'state.json'
        if not rd_path.exists() or not state_path.exists():
            continue
        try:
            rd = json.load(open(rd_path))
            st = json.load(open(state_path))
        except (json.JSONDecodeError, OSError):
            continue

        be = rd.get('back_equity', [])
        fe = rd.get('forward_equity', [])
        score = st.get('final_score', 0)

        if (not be and not fe) and score > 0:
            bc = st.get('best_candidate', {})
            if bc and bc.get('params'):
                broken.append({
                    'run_id': d.name,
                    'run_dir': d,
                    'score': score,
                    'rating': st.get('final_rating', 'RED'),
                    'pair': st.get('pair', '?'),
                    'timeframe': st.get('timeframe', '?'),
                    'strategy': st.get('strategy_name', '?'),
                    'params': bc['params'],
                    'config': st.get('config', {}),
                })
    return broken


def repair_run(run_info: dict):
    """Regenerate equity data for a single run."""
    from data.download import load_data
    from pipeline.stages.s5_montecarlo import MonteCarloStage
    from pipeline.stages.s2_optimization import get_strategy
    from pipeline.config import PipelineConfig
    from pipeline.state import _json_serializer

    run_id = run_info['run_id']
    pair = run_info['pair']
    tf = run_info['timeframe']
    params = run_info['params']
    cfg_dict = run_info['config']

    print(f"  Loading data for {pair} {tf}...")
    config = PipelineConfig.from_dict(cfg_dict)
    config.pair = pair
    config.timeframe = tf

    # Load data and split into back/forward (same as pipeline data stage)
    df = load_data(pair, tf, years=config.data.years)
    if df is None or df.empty:
        print(f"  ERROR: Could not load data for {pair} {tf}")
        return False

    holdout = config.data.holdout_months
    if holdout > 0:
        from dateutil.relativedelta import relativedelta
        holdout_start = df.index[-1] - relativedelta(months=holdout)
        back_ratio = len(df[df.index < holdout_start]) / len(df)
    else:
        back_ratio = 0.8

    split_idx = int(len(df) * back_ratio)
    df_back = df.iloc[:split_idx].copy()
    df_forward = df.iloc[split_idx:].copy()
    print(f"  Data: {len(df)} candles, back={len(df_back)}, forward={len(df_forward)}")

    strategy = get_strategy(run_info['strategy'])
    pip_size = 0.01 if 'JPY' in pair else 0.0001

    print(f"  Running backtest with best params...")
    mc_stage = MonteCarloStage(config)
    trade_details = mc_stage._collect_trade_details(
        params, df_back, df_forward, strategy, pip_size,
    )

    back_eq = trade_details.get('back_equity', [])
    fwd_eq = trade_details.get('forward_equity', [])
    n_back = len(trade_details.get('back_trades', []))
    n_fwd = len(trade_details.get('forward_trades', []))

    print(f"  Got {n_back} back trades, {n_fwd} forward trades")
    print(f"  Equity points: {len(back_eq)} back, {len(fwd_eq)} forward")

    if not back_eq and not fwd_eq:
        print(f"  WARNING: Still no equity data after regeneration")
        return False

    # Update report_data.json
    rd_path = run_info['run_dir'] / 'report_data.json'
    rd = json.load(open(rd_path))
    rd['back_equity'] = back_eq
    rd['forward_equity'] = fwd_eq
    rd['trade_details'] = trade_details
    rd['trade_summary'] = trade_details.get('summary', {})

    # Recompute drawdown curve
    all_eq = back_eq + fwd_eq
    if all_eq:
        dd_curve = []
        peak = all_eq[0]['equity']
        for pt in all_eq:
            eq = pt['equity']
            if eq > peak:
                peak = eq
            dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0
            dd_curve.append({
                'trade_num': pt.get('trade_num', 0),
                'timestamp': pt.get('timestamp', ''),
                'drawdown': dd_pct,
            })
        rd['drawdown_curve'] = dd_curve

    with open(rd_path, 'w') as f:
        json.dump(rd, f, indent=2, default=_json_serializer)

    print(f"  Updated {rd_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Repair missing equity data')
    parser.add_argument('--fix', action='store_true', help='Actually repair (default: dry-run)')
    parser.add_argument('--pipelines-dir', default=None)
    args = parser.parse_args()

    pipelines_dir = Path(args.pipelines_dir) if args.pipelines_dir else project_root / 'results' / 'pipelines'

    broken = find_broken_runs(pipelines_dir)
    print(f"Found {len(broken)} runs with missing equity data:\n")
    for r in broken:
        print(f"  {r['run_id']}  ({r['score']:.0f} {r['rating']})  {r['pair']} {r['timeframe']}")

    if not args.fix:
        print(f"\nDry run. Use --fix to repair.")
        return

    print(f"\nRepairing {len(broken)} runs...\n")
    fixed = 0
    for r in broken:
        print(f"\n[{r['run_id']}]")
        if repair_run(r):
            fixed += 1

    print(f"\nRepaired {fixed}/{len(broken)} runs.")


if __name__ == '__main__':
    main()
