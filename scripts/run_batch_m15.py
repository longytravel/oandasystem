#!/usr/bin/env python
"""
Batch runner: 10 pairs x 3 strategies on M15 with --turbo mode.

Runs full 7-stage pipeline for each combo sequentially.
After each successful run, auto-exports params and adds to strategies.json.

Usage:
    python scripts/run_batch_m15.py
    python scripts/run_batch_m15.py --skip-existing   # skip if already in strategies.json
    python scripts/run_batch_m15.py --dry-run          # print combos without running
"""
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from pipeline.pipeline import Pipeline
from pipeline.config import PipelineConfig

# ── Configuration ──────────────────────────────────────────

PAIRS = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD',
    'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'NZD_USD', 'USD_CHF',
]

# Strategy key -> (pipeline strategy_name, strategies.json key, display name)
STRATEGIES = {
    'rsi_v3': ('RSI_Divergence_v3', 'rsi_v3', 'RSI Div V3'),
    'ema_cross_ml': ('EMA_Cross_ML', 'ema_cross_ml', 'EMA Cross V6'),
    'fair_price_ma': ('Fair_Price_MA', 'fair_price_ma', 'Fair Price MA'),
    'donchian_breakout': ('Donchian_Breakout', 'donchian_breakout', 'Donchian Brk'),
    'bollinger_squeeze': ('Bollinger_Squeeze', 'bollinger_squeeze', 'BB Squeeze'),
    'london_breakout': ('London_Breakout', 'london_breakout', 'London Brk'),
    'stochastic_adx': ('Stochastic_ADX', 'stochastic_adx', 'Stoch ADX'),
}

TIMEFRAME = 'M15'
STRATEGIES_FILE = project_root / 'deploy' / 'strategies.json'


def setup_logging():
    logger.remove()
    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>"
    logger.add(sys.stdout, format=fmt, level="INFO", colorize=True)
    logger.add(
        project_root / "results" / "batch_m15.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        level="INFO",
        rotation="10 MB",
    )


def build_instance_id(strat_key: str, pair: str) -> str:
    """Build instance ID for strategies.json."""
    prefix_map = {
        'rsi_v3': 'rsi_v3',
        'ema_cross_ml': 'ema_v6',
        'fair_price_ma': 'fpma',
        'donchian_breakout': 'dch',
        'bollinger_squeeze': 'bbsq',
        'london_breakout': 'ldn',
        'stochastic_adx': 'stadx',
    }
    prefix = prefix_map.get(strat_key, strat_key)
    return f"{prefix}_{pair}_{TIMEFRAME}"


def generate_description(strat_display: str, pair: str, score: float, rating: str,
                         best: dict, test_months: int) -> str:
    """Generate a rich description for the strategy instance."""
    parts = [f"{strat_display} {TIMEFRAME}"]
    parts.append(f"{score:.0f} {rating}")

    fwd_trades = best.get('forward_trades', 0)
    if fwd_trades > 0 and test_months > 0:
        trades_per_mo = fwd_trades / test_months
        parts.append(f"{trades_per_mo:.1f}/mo")

    wr = best.get('forward_win_rate', 0)
    if wr > 0:
        parts.append(f"WR {wr*100:.0f}%")

    pf = best.get('forward_profit_factor', 0)
    if 0 < pf < 999:
        parts.append(f"PF {pf:.2f}")

    return " | ".join(parts)


def load_strategies_json() -> dict:
    """Load existing strategies.json."""
    if STRATEGIES_FILE.exists():
        with open(STRATEGIES_FILE) as f:
            return json.load(f)
    return {"strategies": [], "dashboard": {"port": 8080, "refresh_seconds": 15}}


def save_strategies_json(data: dict):
    """Save strategies.json atomically."""
    tmp = STRATEGIES_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    tmp.replace(STRATEGIES_FILE)


def add_to_strategies_json(instance_id: str, strat_key: str, pair: str,
                           run_id: str, description: str):
    """Add or update an instance in strategies.json."""
    data = load_strategies_json()
    strategies = data.get('strategies', [])

    # Remove existing entry with same ID
    strategies = [s for s in strategies if s['id'] != instance_id]

    entry = {
        'id': instance_id,
        'strategy': strat_key,
        'pair': pair,
        'timeframe': TIMEFRAME,
        'risk_pct': 1.0,
        'enabled': True,
        'from_run': run_id,
        'description': description,
    }
    strategies.append(entry)
    data['strategies'] = strategies
    save_strategies_json(data)
    logger.info(f"  Added to strategies.json: {instance_id}")


def export_params(run_id: str, instance_id: str) -> bool:
    """Export params from pipeline run to instance config directory."""
    run_dir = project_root / "results" / "pipelines" / run_id
    state_file = run_dir / "state.json"

    if not state_file.exists():
        logger.error(f"  State file not found: {state_file}")
        return False

    with open(state_file) as f:
        state = json.load(f)

    best = state.get("best_candidate", {})
    if not best or not best.get("params"):
        logger.error(f"  No best candidate in {run_id}")
        return False

    pipeline_config = state.get("config", {})
    test_months = pipeline_config.get("walkforward", {}).get("test_months", 6)
    forward_trades = best.get("forward_trades", 0)

    expectations = {
        "win_rate": round(best.get("forward_win_rate", 0), 4),
        "sharpe": round(best.get("forward_sharpe", 0), 3),
        "profit_factor": round(best.get("forward_profit_factor", 0), 3),
        "avg_trades_per_month": round(forward_trades / max(test_months, 1), 1),
        "max_drawdown_pct": round(best.get("forward_max_dd", 0), 2),
        "forward_return_pct": round(best.get("forward_return", 0), 2),
        "forward_trades": forward_trades,
        "forward_months": test_months,
    }

    config = {
        "run_id": run_id,
        "strategy": state.get("strategy_name", "unknown"),
        "pair": state.get("pair", ""),
        "timeframe": state.get("timeframe", ""),
        "score": state.get("final_score", 0),
        "rating": state.get("final_rating", "RED"),
        "rank": best.get("rank", "?"),
        "params": best["params"],
        "expectations": expectations,
    }

    # Save to instance directory
    instance_dir = project_root / "instances" / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    config_path = instance_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Also save to config/ directory
    config_export_path = project_root / "config" / f"live_{config['pair']}_{config['timeframe']}_{instance_id}.json"
    with open(config_export_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"  Exported params -> {config_path}")
    return True


def run_single(pair: str, strat_key: str, strat_info: tuple, skip_existing: bool) -> dict:
    """Run a single pipeline combo. Returns result dict."""
    pipeline_name, json_key, display_name = strat_info
    instance_id = build_instance_id(strat_key, pair)

    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH: {display_name} | {pair} {TIMEFRAME}")
    logger.info(f"Instance: {instance_id}")
    logger.info(f"{'='*60}")

    if skip_existing:
        data = load_strategies_json()
        existing_ids = {s['id'] for s in data.get('strategies', [])}
        if instance_id in existing_ids:
            logger.info(f"  SKIPPED (already in strategies.json)")
            return {'status': 'skipped', 'instance_id': instance_id}

    # Build turbo config
    config = PipelineConfig(
        pair=pair,
        timeframe=TIMEFRAME,
        strategy_name=pipeline_name,
        initial_capital=1000.0,
    )
    # Turbo settings for trade-generation testing
    config.optimization.trials_per_stage = 500
    config.optimization.final_trials = 1500
    config.optimization.top_n_candidates = 5
    config.montecarlo.iterations = 200
    config.montecarlo.bootstrap_iterations = 200
    # Relaxed filters: we want trades to compare, not profits
    config.optimization.max_dd_hard_limit = 50.0   # Allow more DD (normal: 30%)
    config.optimization.min_r2_hard = 0.2           # Allow noisier equity curves (normal: 0.5)

    pipeline = Pipeline(config)
    start = time.time()

    try:
        result = pipeline.run()
        elapsed = time.time() - start
        state = result['state']
        score = state.final_score
        rating = state.final_rating
        run_id = state.run_id

        logger.info(f"  Result: {score:.1f}/100 {rating} ({elapsed/60:.1f} min)")

        # Export params and add to strategies.json
        best = state.best_candidate or {}
        test_months = config.walkforward.test_months
        description = generate_description(display_name, pair, score, rating, best, test_months)

        if best.get('params'):
            export_params(run_id, instance_id)
            add_to_strategies_json(instance_id, json_key, pair, run_id, description)
        else:
            logger.warning(f"  No valid candidate - skipping export")

        return {
            'status': 'ok',
            'instance_id': instance_id,
            'run_id': run_id,
            'score': score,
            'rating': rating,
            'elapsed_minutes': elapsed / 60,
        }

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"  FAILED: {e} ({elapsed/60:.1f} min)")
        return {
            'status': 'error',
            'instance_id': instance_id,
            'error': str(e),
            'elapsed_minutes': elapsed / 60,
        }


def main():
    parser = argparse.ArgumentParser(description="Batch M15 pipeline runner")
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip combos already in strategies.json')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print combos without running')
    parser.add_argument('--pairs', nargs='+', default=None,
                        help='Specific pairs to run (default: all 10)')
    parser.add_argument('--strategies', nargs='+', default=None,
                        help='Specific strategies to run (default: all 3)')
    args = parser.parse_args()

    setup_logging()

    pairs = args.pairs or PAIRS
    strat_keys = args.strategies or list(STRATEGIES.keys())

    combos = [(p, sk, STRATEGIES[sk]) for p in pairs for sk in strat_keys if sk in STRATEGIES]
    total = len(combos)

    logger.info(f"Batch M15 Pipeline Runner")
    logger.info(f"Combos: {total} ({len(pairs)} pairs x {len(strat_keys)} strategies)")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'TURBO (200/500/100)'}")

    if args.dry_run:
        for i, (pair, sk, info) in enumerate(combos, 1):
            iid = build_instance_id(sk, pair)
            logger.info(f"  {i:2d}. {info[2]:<16} {pair:<10} -> {iid}")
        return

    results = []
    batch_start = time.time()

    for i, (pair, sk, info) in enumerate(combos, 1):
        logger.info(f"\n[{i}/{total}] Starting...")
        result = run_single(pair, sk, info, args.skip_existing)
        results.append(result)

        # Progress summary
        ok = sum(1 for r in results if r['status'] == 'ok')
        err = sum(1 for r in results if r['status'] == 'error')
        skip = sum(1 for r in results if r['status'] == 'skipped')
        logger.info(f"Progress: {i}/{total} done | {ok} ok | {err} errors | {skip} skipped")

    # Final summary
    batch_elapsed = (time.time() - batch_start) / 60
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {batch_elapsed:.0f} minutes")

    for r in results:
        status_icon = {'ok': '[OK]', 'error': '[X]', 'skipped': '[--]'}.get(r['status'], '[?]')
        score_str = f"{r.get('score', 0):.0f} {r.get('rating', '')}" if r['status'] == 'ok' else r.get('error', '')[:40]
        logger.info(f"  {status_icon} {r['instance_id']:<30} {score_str}")

    # Save batch results
    results_path = project_root / "results" / "batch_m15_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_minutes': batch_elapsed,
            'results': results,
        }, f, indent=2)
    logger.info(f"Results saved: {results_path}")


if __name__ == '__main__':
    main()
