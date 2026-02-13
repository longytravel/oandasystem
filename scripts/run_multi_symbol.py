#!/usr/bin/env python
"""
Multi-Symbol Pipeline Runner

Run the same strategy across multiple currency pairs IN PARALLEL.
This is crucial for validating that a strategy has a real edge
(not just curve-fitted to one pair).

Usage:
    # Test RSI V3 on all 20 pairs
    python scripts/run_multi_symbol.py --strategy RSI_Divergence_v3

    # Test on specific pairs with multiple timeframes
    python scripts/run_multi_symbol.py --pairs GBP_USD EUR_USD --timeframes H1 M15

    # Fast mode with fewer trials
    python scripts/run_multi_symbol.py --fast --pairs GBP_USD EUR_USD USD_JPY

    # Quick test with custom trials
    python scripts/run_multi_symbol.py --strategy Simple_Trend --trials 1000 --final-trials 2000
"""
import sys
import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


# All tradeable pairs with realistic OANDA practice spreads (pips)
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


def run_pipeline(pair: str, strategy: str, timeframe: str, trials: int,
                 final_trials: int, fast: bool = False, spread: float = None) -> dict:
    """Run pipeline for a single pair. Returns result dict."""
    start = time.time()

    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'run_pipeline.py'),
        '--pair', pair,
        '--timeframe', timeframe,
        '--strategy', strategy,
        '--trials', str(trials),
        '--final-trials', str(final_trials),
    ]
    if fast:
        cmd.append('--fast')
    if spread is not None:
        cmd.extend(['--spread', str(spread)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour max
        )

        elapsed = time.time() - start

        # Parse output for key metrics
        output = result.stdout + result.stderr

        # Extract rating
        rating = 'UNKNOWN'
        if 'rating=RED' in output or 'RED' in output:
            rating = 'RED'
        elif 'rating=YELLOW' in output or 'YELLOW' in output:
            rating = 'YELLOW'
        elif 'rating=GREEN' in output or 'GREEN' in output:
            rating = 'GREEN'

        # Extract stability
        mean_stability = 0.0
        for line in output.split('\n'):
            if 'Mean stability:' in line:
                try:
                    mean_stability = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass

        # Extract candidates passed
        candidates_passed = 0
        for line in output.split('\n'):
            if 'Candidates passed:' in line:
                try:
                    candidates_passed = int(line.split(':')[1].strip().split('/')[0])
                except:
                    pass

        return {
            'pair': pair,
            'strategy': strategy,
            'timeframe': timeframe,
            'rating': rating,
            'mean_stability': mean_stability,
            'candidates_passed': candidates_passed,
            'elapsed_minutes': elapsed / 60,
            'success': result.returncode == 0,
            'error': None if result.returncode == 0 else result.stderr[-500:],
        }

    except subprocess.TimeoutExpired:
        return {
            'pair': pair,
            'strategy': strategy,
            'timeframe': timeframe,
            'rating': 'TIMEOUT',
            'mean_stability': 0,
            'candidates_passed': 0,
            'elapsed_minutes': 120,
            'success': False,
            'error': 'Pipeline timed out after 2 hours',
        }
    except Exception as e:
        return {
            'pair': pair,
            'strategy': strategy,
            'timeframe': timeframe,
            'rating': 'ERROR',
            'mean_stability': 0,
            'candidates_passed': 0,
            'elapsed_minutes': (time.time() - start) / 60,
            'success': False,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Run pipeline on multiple symbols in parallel')
    parser.add_argument('--strategy', '-s', default='RSI_Divergence_v3', help='Strategy to test')
    parser.add_argument('--pairs', nargs='+', default=ALL_PAIRS, help='Pairs to test')
    parser.add_argument('--timeframes', nargs='+', default=['H1'],
                        choices=['M15', 'M30', 'H1', 'H4'],
                        help='Timeframes to test (default: H1)')
    parser.add_argument('--trials', type=int, default=2000, help='Trials per stage')
    parser.add_argument('--final-trials', type=int, default=3000, help='Final optimization trials')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Parallel workers')
    parser.add_argument('--fast', action='store_true', help='Use --fast pipeline mode')

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    # Build combo list: pairs x timeframes
    combos = [(pair, tf) for pair in args.pairs for tf in args.timeframes]

    logger.info("=" * 70)
    logger.info("MULTI-SYMBOL PIPELINE TEST")
    logger.info("=" * 70)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info(f"Total combos: {len(combos)}")
    logger.info(f"Trials: {args.trials} / {args.final_trials}")
    logger.info(f"Parallel workers: {args.workers}")
    logger.info("=" * 70)

    start_time = time.time()
    results = []

    # Run all combos in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for pair, tf in combos:
            spread = PAIR_SPREADS.get(pair, 2.0)
            future = executor.submit(
                run_pipeline,
                pair, args.strategy, tf,
                args.trials, args.final_trials,
                args.fast, spread,
            )
            futures[future] = (pair, tf)

        for future in as_completed(futures):
            pair, tf = futures[future]
            try:
                result = future.result()
                results.append(result)

                # Print result immediately
                status = "OK" if result['success'] else "X"
                logger.info(
                    f"{status} {result['pair']} {result['timeframe']}: {result['rating']} "
                    f"(stability={result['mean_stability']:.1f}%, "
                    f"passed={result['candidates_passed']}, "
                    f"time={result['elapsed_minutes']:.1f}m)"
                )

            except Exception as e:
                logger.error(f"X {pair} {tf}: Exception - {e}")
                results.append({
                    'pair': pair,
                    'timeframe': tf,
                    'rating': 'ERROR',
                    'success': False,
                    'error': str(e),
                })

    # Summary
    total_time = (time.time() - start_time) / 60

    logger.info("\n" + "=" * 70)
    logger.info("MULTI-SYMBOL RESULTS SUMMARY")
    logger.info("=" * 70)

    # Results table
    print("\n")
    print(f"{'Pair':<12} {'TF':<5} {'Rating':<8} {'Stability':<12} {'Passed':<10} {'Time':<10}")
    print("-" * 57)

    green_count = 0
    yellow_count = 0
    red_count = 0

    for r in sorted(results, key=lambda x: (x['pair'], x.get('timeframe', ''))):
        rating_color = {
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'RED': '\033[91m',
        }.get(r['rating'], '\033[0m')

        print(
            f"{r['pair']:<12} "
            f"{r.get('timeframe', 'H1'):<5} "
            f"{rating_color}{r['rating']:<8}\033[0m "
            f"{r.get('mean_stability', 0):.1f}%{'':<8} "
            f"{r.get('candidates_passed', 0):<10} "
            f"{r.get('elapsed_minutes', 0):.1f}m"
        )

        if r['rating'] == 'GREEN':
            green_count += 1
        elif r['rating'] == 'YELLOW':
            yellow_count += 1
        else:
            red_count += 1

    print("-" * 57)
    print(f"\nTotal time: {total_time:.1f} minutes")
    print(f"Results: {green_count} GREEN, {yellow_count} YELLOW, {red_count} RED")

    # Verdict
    total_combos = len(combos)
    print("\n" + "=" * 70)
    if green_count >= total_combos // 2:
        print("VERDICT: STRATEGY SHOWS PROMISE - Works on multiple pairs")
    elif green_count > 0 or yellow_count >= total_combos // 2:
        print("VERDICT: MIXED RESULTS - May need refinement")
    else:
        print("VERDICT: STRATEGY LIKELY FLAWED - Fails across all pairs")
    print("=" * 70)

    # Save results
    results_file = project_root / 'results' / f'multi_symbol_{args.strategy}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        f.write(f"Strategy: {args.strategy}\n")
        f.write(f"Timeframes: {args.timeframes}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        for r in results:
            f.write(f"{r['pair']} {r.get('timeframe', 'H1')}: {r['rating']} "
                    f"(stability={r.get('mean_stability', 0):.1f}%)\n")

    logger.info(f"\nResults saved to: {results_file}")

    return 0 if green_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
