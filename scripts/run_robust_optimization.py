"""
Robust Optimization Script

Focuses on finding STABLE parameters rather than "best" parameters.
Tests parameter stability by checking if neighboring values also perform well.

A robust strategy:
- Performs well on backtest AND forward test
- Has stable neighbors (small parameter changes don't kill performance)
- Rating: ROBUST > MODERATE > FRAGILE > OVERFIT

Usage:
    python scripts/run_robust_optimization.py --pair EUR_USD --timeframe H1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from datetime import datetime
from loguru import logger

from strategies.rsi_full import RSIDivergenceFullFast
from optimization.unified_optimizer import UnifiedOptimizer


def load_data(pair: str, timeframe: str = 'H1') -> pd.DataFrame:
    """Load OANDA data for a pair."""
    data_dir = Path(__file__).parent.parent / 'data' / 'oanda'

    parquet_path = data_dir / f'{pair}_{timeframe}.parquet'
    csv_path = data_dir / f'{pair}_{timeframe}.csv'

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} bars from {parquet_path}")
    elif csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} bars from {csv_path}")
    else:
        raise FileNotFoundError(f"Data file not found: {parquet_path} or {csv_path}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


def split_data(df: pd.DataFrame, back_ratio: float = 0.7) -> tuple:
    """Split data into back and forward periods."""
    split_idx = int(len(df) * back_ratio)
    df_back = df.iloc[:split_idx].copy()
    df_forward = df.iloc[split_idx:].copy()

    logger.info(f"Back period: {df_back.index[0]} to {df_back.index[-1]} ({len(df_back)} bars)")
    logger.info(f"Forward period: {df_forward.index[0]} to {df_forward.index[-1]} ({len(df_forward)} bars)")

    return df_back, df_forward


def run_robust_optimization(
    pair: str,
    timeframe: str = 'H1',
    trials: int = 3000,
    n_candidates: int = 30,
    min_stability: float = 0.6,
):
    """Run robustness-focused optimization."""
    print("\n" + "="*80)
    print(f"ROBUST OPTIMIZATION")
    print(f"Pair: {pair} | Timeframe: {timeframe}")
    print(f"Goal: Find STABLE parameters, not just 'best' parameters")
    print("="*80)

    # Load and split data
    df = load_data(pair, timeframe)
    df_back, df_forward = split_data(df)

    # Initialize strategy and optimizer
    strategy = RSIDivergenceFullFast()
    strategy.set_pip_size(pair)

    optimizer = UnifiedOptimizer(
        strategy,
        min_forward_ratio=0.05,   # Reject if forward < 5% of back
        forward_rank_weight=2.0,  # Forward matters more
    )

    # Run robust optimization
    results = optimizer.run_robust_optimization(
        df_back, df_forward,
        n_trials=trials,
        n_candidates=n_candidates,
        min_stability=min_stability,
    )

    # Print results
    if results:
        print("\n" + "="*80)
        print("ROBUST RESULTS SUMMARY")
        print("="*80)
        print(f"{'Rank':<6} {'Back Score':<12} {'Fwd Score':<12} {'Stability':<12} {'Rating':<10}")
        print("-"*80)

        for i, r in enumerate(results[:10]):
            stability = r['stability']['overall']
            print(f"{i+1:<6} "
                  f"{r['back'].ontester_score:<12.1f} "
                  f"{r['forward'].ontester_score:<12.1f} "
                  f"{stability['mean_stability']:<12.1%} "
                  f"{stability['rating']:<10}")

        print("="*80)

        # Save results
        output_dir = Path(__file__).parent.parent / 'results' / 'robust'
        output_dir.mkdir(parents=True, exist_ok=True)
        optimizer.save_results(output_dir)

        # Print best robust params
        best = results[0]
        print("\n" + "-"*80)
        print("BEST ROBUST PARAMETERS:")
        print("-"*80)
        for k, v in sorted(best['params'].items()):
            print(f"  {k}: {v}")

        print("\n" + "-"*80)
        print("STABILITY DETAILS:")
        print("-"*80)
        for param, data in sorted(best['stability']['params'].items(),
                                   key=lambda x: x[1]['stability_ratio']):
            if data['stability_ratio'] < 0.7:
                print(f"  {param}: {data['stability_ratio']:.0%} stability (WATCH)")

    else:
        print("\n" + "="*80)
        print("NO ROBUST RESULTS FOUND")
        print("="*80)
        print(f"All top {n_candidates} candidates failed stability test (< {min_stability:.0%})")
        print("This indicates the strategy may be fundamentally overfit on this pair.")
        print("Consider:")
        print("  1. Using simpler parameters (fewer filters)")
        print("  2. Using more data")
        print("  3. Accepting that this pair may not be suitable for this strategy")

    return results


def main():
    parser = argparse.ArgumentParser(description='Robust optimization for RSI Divergence')
    parser.add_argument('--pair', default='EUR_USD', help='Currency pair')
    parser.add_argument('--timeframe', default='H1', help='Timeframe')
    parser.add_argument('--trials', type=int, default=3000, help='Optimization trials')
    parser.add_argument('--candidates', type=int, default=30, help='Candidates to stability test')
    parser.add_argument('--min-stability', type=float, default=0.6, help='Min stability ratio')

    args = parser.parse_args()

    run_robust_optimization(
        pair=args.pair,
        timeframe=args.timeframe,
        trials=args.trials,
        n_candidates=args.candidates,
        min_stability=args.min_stability,
    )


if __name__ == '__main__':
    main()
