#!/usr/bin/env python
"""
Multi-Pair Scanner

Three-phase approach:
1. Signal pre-check (~5 min) - Count signals per pair/timeframe, skip low-signal combos
2. Fast pipeline on survivors (~15-20 min each, parallel) - Real Optuna optimization
3. (--overnight) Full pipeline on GREEN results sequentially - 5000/10000 trials

Usage:
    python scripts/fast_scan.py -s fair_price_ma --timeframes H1 H4           # Scan with any strategy
    python scripts/fast_scan.py --pairs EUR_USD USD_JPY                       # Specific pairs
    python scripts/fast_scan.py --workers 3                                   # More parallelism
    python scripts/fast_scan.py --min-signals 30                              # Higher signal threshold
    python scripts/fast_scan.py --skip-precheck                               # Skip signal check, run all
    python scripts/fast_scan.py --full                                        # Full pipeline (not --fast)
    python scripts/fast_scan.py -s fair_price_ma --timeframes H1 H4 --overnight  # Auto full pipelines on GREEN
"""
import sys
import re
import json
import argparse
import subprocess
import time
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

PROGRESS_FILE = project_root / 'results' / 'scan_progress.json'


# Realistic OANDA practice spreads in pips
PAIR_SPREADS = {
    # Majors
    'EUR_USD': 1.0, 'GBP_USD': 1.5, 'USD_JPY': 1.2, 'USD_CHF': 1.5,
    'AUD_USD': 1.5, 'NZD_USD': 2.0, 'USD_CAD': 2.0,
    # Crosses
    'EUR_GBP': 1.8, 'EUR_JPY': 2.0, 'GBP_JPY': 3.0, 'AUD_JPY': 2.5,
    'EUR_CHF': 2.5, 'GBP_CHF': 3.5, 'AUD_NZD': 2.5, 'EUR_AUD': 3.0,
    'EUR_CAD': 3.0, 'GBP_AUD': 4.0, 'GBP_CAD': 4.0, 'CAD_JPY': 2.5,
    'NZD_JPY': 3.0, 'CHF_JPY': 2.5,
}

ALL_PAIRS = list(PAIR_SPREADS.keys())
DEFAULT_TIMEFRAMES = ['M15', 'H1']


def signal_precheck(pair: str, timeframe: str, years: float = 3.0, strategy_name: str = 'RSI_Divergence_v3') -> int:
    """
    Count signals for a pair/timeframe combo using any strategy.
    Returns signal count, or -1 on error.
    """
    try:
        from data.download import load_data
        from pipeline.stages.s2_optimization import get_strategy

        df = load_data(instrument=pair, timeframe=timeframe, auto_download=True, years=years)
        if df is None or len(df) < 500:
            return -1

        strategy = get_strategy(strategy_name)
        strategy.set_pip_size(pair)
        signals = strategy.precompute(df)
        return len(signals)

    except Exception as e:
        logger.warning(f"Pre-check failed for {pair} {timeframe}: {e}")
        return -1


def run_pipeline_subprocess(
    pair: str,
    timeframe: str,
    spread: float,
    fast: bool = True,
    years: float = 3.0,
    strategy: str = 'RSI_Divergence_v3',
    description: str = '',
) -> dict:
    """
    Run pipeline for a single pair/timeframe via subprocess.
    Returns result dict with score, rating, timing, etc.
    """
    start = time.time()

    desc = description or f"scan:{'fast' if fast else 'full'} {pair} {timeframe} {strategy}"
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'run_pipeline.py'),
        '--pair', pair,
        '--timeframe', timeframe,
        '--strategy', strategy,
        '--years', str(years),
        '--spread', str(spread),
        '-d', desc,
    ]
    if fast:
        cmd.append('--fast')

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour max per combo
        )

        elapsed = time.time() - start
        output = result.stdout + result.stderr

        # Try to find run directory and read state.json for accurate results
        score = 0.0
        rating = 'UNKNOWN'
        wf_pass_rate = 0.0
        sharpe = 0.0
        stability = 0.0
        run_dir = None

        # Parse run directory from output (strip ANSI codes first)
        ansi_re = re.compile(r'\x1b\[[0-9;]*m')
        for line in output.split('\n'):
            if 'Run directory:' in line:
                clean = ansi_re.sub('', line)
                path_part = clean.split('Run directory:')[1].strip().rstrip('|').strip()
                run_dir = Path(path_part)
                break

        # Read state.json if available (most reliable)
        if run_dir and (run_dir / 'state.json').exists():
            try:
                with open(run_dir / 'state.json', 'r') as f:
                    state = json.load(f)
                score = state.get('final_score', 0.0)
                rating = state.get('final_rating', 'RED')

                # Extract WF and stability from best candidate
                best = state.get('best_candidate')
                if best:
                    wf = best.get('walkforward', {})
                    wf_stats = wf.get('stats', {})
                    wf_pass_rate = wf_stats.get('pass_rate', 0.0)
                    sharpe = wf_stats.get('mean_sharpe', 0.0)
                    stab = best.get('stability', {})
                    stab_overall = stab.get('overall', {})
                    stability = stab_overall.get('mean_stability', 0.0)
            except Exception as e:
                logger.warning(f"Could not read state.json for {pair} {timeframe}: {e}")

        # Fallback: parse from output if state.json not found
        if rating == 'UNKNOWN':
            for line in output.split('\n'):
                if 'CONFIDENCE SCORE:' in line:
                    try:
                        score = float(line.split('CONFIDENCE SCORE:')[1].split('/')[0].strip())
                    except (ValueError, IndexError):
                        pass
                if 'RATING:' in line:
                    if 'GREEN' in line:
                        rating = 'GREEN'
                    elif 'YELLOW' in line:
                        rating = 'YELLOW'
                    elif 'RED' in line:
                        rating = 'RED'

        return {
            'pair': pair,
            'timeframe': timeframe,
            'spread': spread,
            'score': score,
            'rating': rating,
            'wf_pass_rate': wf_pass_rate,
            'sharpe': sharpe,
            'stability': stability,
            'elapsed_minutes': elapsed / 60,
            'run_dir': str(run_dir) if run_dir else '',
            'success': result.returncode == 0,
            'error': None,
        }

    except subprocess.TimeoutExpired:
        return {
            'pair': pair, 'timeframe': timeframe, 'spread': spread,
            'score': 0.0, 'rating': 'TIMEOUT',
            'wf_pass_rate': 0.0, 'sharpe': 0.0, 'stability': 0.0,
            'elapsed_minutes': 120.0, 'run_dir': '', 'success': False,
            'error': 'Pipeline timed out after 2 hours',
        }
    except Exception as e:
        return {
            'pair': pair, 'timeframe': timeframe, 'spread': spread,
            'score': 0.0, 'rating': 'ERROR',
            'wf_pass_rate': 0.0, 'sharpe': 0.0, 'stability': 0.0,
            'elapsed_minutes': (time.time() - start) / 60, 'run_dir': '',
            'success': False, 'error': str(e),
        }


def run_phase1(pairs, timeframes, min_signals, years, strategy_name='RSI_Divergence_v3'):
    """Phase 1: Signal pre-check. Returns list of (pair, tf, signal_count) that passed."""
    total = len(pairs) * len(timeframes)
    logger.info(f"=== PHASE 1: Signal Pre-Check ({len(pairs)} pairs x {len(timeframes)} timeframes = {total} combos) ===")
    print()

    passed = []
    skipped = []

    for pair in pairs:
        line_parts = []
        for tf in timeframes:
            count = signal_precheck(pair, tf, years=years, strategy_name=strategy_name)
            if count >= min_signals:
                passed.append((pair, tf, count))
                line_parts.append(f"{tf}: {count:>5} signals  OK")
            elif count == -1:
                skipped.append((pair, tf, 'ERROR'))
                line_parts.append(f"{tf}:   ERROR       SKIP")
            else:
                skipped.append((pair, tf, count))
                line_parts.append(f"{tf}: {count:>5} signals  SKIP (<{min_signals})")

        # Print pair results on one line
        pair_str = f"  {pair:<10}"
        print(pair_str + "   ".join(line_parts))

    print()
    logger.info(f"Passed: {len(passed)}/{total} combos. Skipped: {len(skipped)}.")
    if skipped:
        skip_strs = [f"{p} {t} ({c})" for p, t, c in skipped]
        logger.info(f"Skipped: {', '.join(skip_strs)}")
    print()

    return passed


def write_progress(scan_id, strategy, timeframes, phase, total_combos, results, status='running'):
    """Write scan progress to JSON for dashboard consumption."""
    progress = {
        'scan_id': scan_id,
        'strategy': strategy,
        'timeframes': timeframes,
        'status': status,
        'started_at': getattr(write_progress, '_started', datetime.now().isoformat()),
        'phase': phase,
        'total_combos': total_combos,
        'completed': len(results),
        'results': results,
    }
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def run_phase2(combos, workers, fast, years, strategy, scan_id=None, timeframes=None):
    """Phase 2: Run pipeline on surviving combos. Returns list of result dicts."""
    total = len(combos)
    est_time_per = 18 if fast else 35
    est_total = (total * est_time_per) / workers
    logger.info(f"=== PHASE 2: {'Fast' if fast else 'Full'} Pipeline ({workers} workers) ===")
    logger.info(f"Running {total} combos. Estimated time: ~{est_total:.0f} min ({est_total/60:.1f} hours)")
    print()

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for pair, tf, _count in combos:
            spread = PAIR_SPREADS.get(pair, 2.0)
            future = executor.submit(
                run_pipeline_subprocess,
                pair, tf, spread, fast, years, strategy,
            )
            futures[future] = (pair, tf)

        for future in as_completed(futures):
            pair, tf = futures[future]
            completed += 1
            try:
                result = future.result()
                result['phase'] = 'fast_done' if fast else 'full_done'
                results.append(result)

                # Color-code rating
                rating_colors = {
                    'GREEN': '\033[92m', 'YELLOW': '\033[93m',
                    'RED': '\033[91m', 'ERROR': '\033[91m', 'TIMEOUT': '\033[91m',
                }
                color = rating_colors.get(result['rating'], '\033[0m')
                reset = '\033[0m'

                print(
                    f"  [{completed:>2}/{total}] {result['pair']:<10} {result['timeframe']:<4}: "
                    f"{color}{result['score']:>5.1f}/100 {result['rating']:<7}{reset} "
                    f"({result['elapsed_minutes']:.1f} min)"
                )

                # Update progress file
                if scan_id:
                    write_progress(scan_id, strategy, timeframes or [],
                                   'fast_scan' if fast else 'full_pipeline',
                                   total, results)

            except Exception as e:
                logger.error(f"  [{completed}/{total}] {pair} {tf}: Exception - {e}")
                results.append({
                    'pair': pair, 'timeframe': tf, 'spread': PAIR_SPREADS.get(pair, 2.0),
                    'score': 0.0, 'rating': 'ERROR',
                    'wf_pass_rate': 0.0, 'sharpe': 0.0, 'stability': 0.0,
                    'elapsed_minutes': 0.0, 'run_dir': '', 'success': False,
                    'error': str(e), 'phase': 'error',
                })

    return results


def run_phase3_full(fast_results, years, strategy, scan_id=None, timeframes=None):
    """Phase 3 (overnight): Run full pipelines on GREEN results sequentially."""
    green_results = sorted(
        [r for r in fast_results if r['rating'] == 'GREEN'],
        key=lambda x: x['score'], reverse=True,
    )

    if not green_results:
        logger.info("No GREEN results from fast scan. Skipping full pipeline phase.")
        return []

    logger.info(f"=== PHASE 3: Full Pipeline on {len(green_results)} GREEN results (sequential) ===")
    est_total = len(green_results) * 50
    logger.info(f"Estimated time: ~{est_total:.0f} min ({est_total/60:.1f} hours)")
    print()

    full_results = []
    for rank, r in enumerate(green_results, 1):
        pair, tf = r['pair'], r['timeframe']
        spread = PAIR_SPREADS.get(pair, 2.0)
        desc = f"scan:full #{rank} {pair} {tf} {strategy}"

        logger.info(f"  [{rank}/{len(green_results)}] Running full pipeline: {pair} {tf} (fast score: {r['score']:.1f})")

        result = run_pipeline_subprocess(
            pair, tf, spread, fast=False, years=years,
            strategy=strategy, description=desc,
        )
        result['phase'] = 'full_done'
        result['fast_score'] = r['score']
        result['fast_rating'] = r['rating']
        full_results.append(result)

        # Color-code
        rating_colors = {
            'GREEN': '\033[92m', 'YELLOW': '\033[93m',
            'RED': '\033[91m', 'ERROR': '\033[91m', 'TIMEOUT': '\033[91m',
        }
        color = rating_colors.get(result['rating'], '\033[0m')
        reset = '\033[0m'
        print(
            f"  [{rank:>2}/{len(green_results)}] {pair:<10} {tf:<4}: "
            f"{color}{result['score']:>5.1f}/100 {result['rating']:<7}{reset} "
            f"(fast: {r['score']:.1f}) ({result['elapsed_minutes']:.1f} min)"
        )

        # Update progress file
        if scan_id:
            write_progress(scan_id, strategy, timeframes or [],
                           f'full_pipeline ({rank}/{len(green_results)})',
                           len(green_results), full_results)

    return full_results


def print_rankings(results):
    """Print final ranked summary table."""
    # Sort by score descending
    ranked = sorted(results, key=lambda x: x['score'], reverse=True)

    print()
    print("=" * 95)
    print("  FINAL RANKINGS")
    print("=" * 95)
    print(f"  {'Rank':<5} {'Pair':<10} {'TF':<5} {'Score':<8} {'Rating':<8} "
          f"{'WF%':<7} {'Sharpe':<8} {'Stab%':<8} {'Spread':<7} {'Time':<6}")
    print("  " + "-" * 88)

    rating_colors = {
        'GREEN': '\033[92m', 'YELLOW': '\033[93m',
        'RED': '\033[91m', 'ERROR': '\033[91m', 'TIMEOUT': '\033[91m',
    }
    reset = '\033[0m'

    for i, r in enumerate(ranked, 1):
        color = rating_colors.get(r['rating'], '\033[0m')
        print(
            f"  {i:<5} {r['pair']:<10} {r['timeframe']:<5} "
            f"{color}{r['score']:>5.1f}{reset}   {color}{r['rating']:<8}{reset} "
            f"{r['wf_pass_rate']*100:>5.0f}%  "
            f"{r['sharpe']:>6.2f}  "
            f"{r['stability']*100:>5.1f}%  "
            f"{r['spread']:>4.1f}p   "
            f"{r['elapsed_minutes']:>4.0f}m"
        )

    print("  " + "-" * 88)

    # Summary counts
    n_green = sum(1 for r in ranked if r['rating'] == 'GREEN')
    n_yellow = sum(1 for r in ranked if r['rating'] == 'YELLOW')
    n_red = sum(1 for r in ranked if r['rating'] == 'RED')
    n_other = sum(1 for r in ranked if r['rating'] not in ('GREEN', 'YELLOW', 'RED'))
    total_time = sum(r['elapsed_minutes'] for r in ranked)

    print(f"\n  Results: {n_green} GREEN, {n_yellow} YELLOW, {n_red} RED"
          + (f", {n_other} other" if n_other else ""))
    print(f"  Total wall time: {total_time:.0f} min ({total_time/60:.1f} hours)")
    print("=" * 95)


def save_csv(results, output_path):
    """Save results to CSV."""
    ranked = sorted(results, key=lambda x: x['score'], reverse=True)
    fieldnames = ['rank', 'pair', 'timeframe', 'score', 'rating', 'wf_pass_rate',
                  'sharpe', 'stability', 'spread', 'elapsed_minutes', 'run_dir']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(ranked, 1):
            writer.writerow({
                'rank': i,
                'pair': r['pair'],
                'timeframe': r['timeframe'],
                'score': f"{r['score']:.1f}",
                'rating': r['rating'],
                'wf_pass_rate': f"{r['wf_pass_rate']*100:.0f}%",
                'sharpe': f"{r['sharpe']:.2f}",
                'stability': f"{r['stability']*100:.1f}%",
                'spread': r['spread'],
                'elapsed_minutes': f"{r['elapsed_minutes']:.1f}",
                'run_dir': r['run_dir'],
            })

    logger.info(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Pair Scanner - Find profitable pair/timeframe combos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--pairs', nargs='+', default=ALL_PAIRS,
                        help=f'Pairs to scan (default: all {len(ALL_PAIRS)})')
    parser.add_argument('--timeframes', nargs='+', default=DEFAULT_TIMEFRAMES,
                        choices=['M15', 'M30', 'H1', 'H4'],
                        help='Timeframes to scan (default: M15 H1)')
    parser.add_argument('--workers', '-w', type=int, default=2,
                        help='Parallel pipeline workers (default: 2)')
    parser.add_argument('--min-signals', type=int, default=20,
                        help='Minimum signals to proceed to pipeline (default: 20)')
    parser.add_argument('--skip-precheck', action='store_true',
                        help='Skip signal pre-check, run all combos')
    parser.add_argument('--fast', action='store_true', default=True,
                        help='Use --fast pipeline mode (default: True)')
    parser.add_argument('--full', action='store_true',
                        help='Use full pipeline mode (overrides --fast)')
    parser.add_argument('--overnight', action='store_true',
                        help='After fast scan, auto-run full pipelines on GREEN results')
    parser.add_argument('--years', type=float, default=4.0,
                        help='Years of data (default: 4.0)')
    parser.add_argument('--strategy', '-s', default='RSI_Divergence_v3',
                        help='Strategy name (default: RSI_Divergence_v3)')

    args = parser.parse_args()

    # --full overrides --fast
    use_fast = not args.full

    # Setup logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    # Scan ID for progress tracking
    scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    write_progress._started = datetime.now().isoformat()

    # Header
    print()
    print("=" * 95)
    print(f"  MULTI-PAIR SCANNER - {args.strategy}")
    print("=" * 95)
    logger.info(f"Pairs: {len(args.pairs)} | Timeframes: {args.timeframes} | Workers: {args.workers}")
    logger.info(f"Mode: {'FAST' if use_fast else 'FULL'}{' + OVERNIGHT' if args.overnight else ''} | Years: {args.years} | Min signals: {args.min_signals}")
    logger.info(f"Scan ID: {scan_id}")
    print()

    start_time = time.time()

    # Phase 1: Signal pre-check
    if args.skip_precheck:
        combos = [(pair, tf, -1) for pair in args.pairs for tf in args.timeframes]
        logger.info(f"Skipping pre-check. Running all {len(combos)} combos.")
    else:
        combos = run_phase1(args.pairs, args.timeframes, args.min_signals, args.years, args.strategy)

    if not combos:
        logger.error("No combos passed signal pre-check. Nothing to run.")
        write_progress(scan_id, args.strategy, args.timeframes, 'done', 0, [], status='completed')
        return 1

    # Initialize progress
    total_combos = len(combos)
    write_progress(scan_id, args.strategy, args.timeframes, 'fast_scan', total_combos, [])

    # Estimate time
    est_per = 18 if use_fast else 35
    est_total = (total_combos * est_per) / args.workers
    logger.info(f"Starting pipeline on {total_combos} combos (~{est_total:.0f} min / {est_total/60:.1f} hours estimated)")
    print()

    # Phase 2: Run pipelines
    results = run_phase2(combos, args.workers, use_fast, args.years, args.strategy,
                         scan_id=scan_id, timeframes=args.timeframes)

    # Final rankings (Phase 2)
    print_rankings(results)

    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = project_root / 'results' / f'scan_{timestamp}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(results, csv_path)

    # Phase 3: Overnight full pipelines on GREEN results
    full_results = []
    if args.overnight:
        full_results = run_phase3_full(
            results, args.years, args.strategy,
            scan_id=scan_id, timeframes=args.timeframes,
        )

        if full_results:
            print()
            logger.info("=== FULL PIPELINE RANKINGS ===")
            print_rankings(full_results)

            full_csv = project_root / 'results' / f'scan_full_{timestamp}.csv'
            save_csv(full_results, full_csv)

    # Mark scan complete
    all_results = results + full_results
    write_progress(scan_id, args.strategy, args.timeframes, 'done',
                   total_combos, results, status='completed')

    total_time = (time.time() - start_time) / 60
    n_green = sum(1 for r in results if r['rating'] == 'GREEN')
    print(f"\n  Total elapsed: {total_time:.0f} min ({total_time/60:.1f} hours)")
    if args.overnight:
        n_full_green = sum(1 for r in full_results if r['rating'] == 'GREEN')
        print(f"  Fast scan: {n_green} GREEN / {len(results)} total")
        print(f"  Full pipeline: {n_full_green} GREEN / {len(full_results)} total")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
