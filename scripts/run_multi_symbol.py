#!/usr/bin/env python
"""
Multi-Symbol Pipeline Runner

Run the same strategy across multiple currency pairs IN PARALLEL.
This is crucial for validating that a strategy has a real edge
(not just curve-fitted to one pair).

Usage:
    # Test Simple Trend on all 4 pairs
    python scripts/run_multi_symbol.py --strategy Simple_Trend

    # Test on specific pairs
    python scripts/run_multi_symbol.py --strategy Simple_Trend --pairs GBP_USD EUR_USD

    # Quick test with fewer trials
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


# All pairs we have M1 data for
ALL_PAIRS = ['GBP_USD', 'EUR_USD', 'EUR_JPY', 'USD_JPY']


def run_pipeline(pair: str, strategy: str, timeframe: str, trials: int, final_trials: int) -> dict:
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

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
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
            'elapsed_minutes': 60,
            'success': False,
            'error': 'Pipeline timed out after 1 hour',
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
    parser.add_argument('--strategy', '-s', default='Simple_Trend', help='Strategy to test')
    parser.add_argument('--pairs', nargs='+', default=ALL_PAIRS, help='Pairs to test')
    parser.add_argument('--timeframe', '-t', default='H1', help='Timeframe')
    parser.add_argument('--trials', type=int, default=2000, help='Trials per stage')
    parser.add_argument('--final-trials', type=int, default=3000, help='Final optimization trials')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Parallel workers')

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    logger.info("=" * 70)
    logger.info("MULTI-SYMBOL PIPELINE TEST")
    logger.info("=" * 70)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Trials: {args.trials} / {args.final_trials}")
    logger.info(f"Parallel workers: {args.workers}")
    logger.info("=" * 70)

    start_time = time.time()
    results = []

    # Run all pairs in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_pipeline,
                pair,
                args.strategy,
                args.timeframe,
                args.trials,
                args.final_trials
            ): pair
            for pair in args.pairs
        }

        for future in as_completed(futures):
            pair = futures[future]
            try:
                result = future.result()
                results.append(result)

                # Print result immediately
                status = "✓" if result['success'] else "✗"
                logger.info(
                    f"{status} {result['pair']}: {result['rating']} "
                    f"(stability={result['mean_stability']:.1f}%, "
                    f"passed={result['candidates_passed']}, "
                    f"time={result['elapsed_minutes']:.1f}m)"
                )

            except Exception as e:
                logger.error(f"✗ {pair}: Exception - {e}")
                results.append({
                    'pair': pair,
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
    print(f"{'Pair':<12} {'Rating':<8} {'Stability':<12} {'Passed':<10} {'Time':<10}")
    print("-" * 52)

    green_count = 0
    yellow_count = 0
    red_count = 0

    for r in sorted(results, key=lambda x: x['pair']):
        rating_color = {
            'GREEN': '\033[92m',  # Green
            'YELLOW': '\033[93m',  # Yellow
            'RED': '\033[91m',  # Red
        }.get(r['rating'], '\033[0m')

        print(
            f"{r['pair']:<12} "
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

    print("-" * 52)
    print(f"\nTotal time: {total_time:.1f} minutes")
    print(f"Results: {green_count} GREEN, {yellow_count} YELLOW, {red_count} RED")

    # Verdict
    print("\n" + "=" * 70)
    if green_count >= len(args.pairs) // 2:
        print("VERDICT: STRATEGY SHOWS PROMISE - Works on multiple pairs")
    elif green_count > 0 or yellow_count >= len(args.pairs) // 2:
        print("VERDICT: MIXED RESULTS - May need refinement")
    else:
        print("VERDICT: STRATEGY LIKELY FLAWED - Fails across all pairs")
    print("=" * 70)

    # Save results
    results_file = project_root / 'results' / f'multi_symbol_{args.strategy}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(results_file, 'w') as f:
        f.write(f"Strategy: {args.strategy}\n")
        f.write(f"Timeframe: {args.timeframe}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        for r in results:
            f.write(f"{r['pair']}: {r['rating']} (stability={r.get('mean_stability', 0):.1f}%)\n")

    logger.info(f"\nResults saved to: {results_file}")

    return 0 if green_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
