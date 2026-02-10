"""
V1 vs V2 Comparison Script

Runs both RSI Divergence V1 and V2 on the same data to compare:
- Number of signals generated
- Forward performance
- Robustness (back vs forward consistency)

Usage:
    python scripts/compare_v1_v2.py --pair EUR_USD --timeframe H1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# Import strategies
from strategies.rsi_full import RSIDivergenceFullFast
from strategies.rsi_divergence_v2 import RSIDivergenceV2
from optimization.unified_optimizer import UnifiedOptimizer


def load_data(pair: str, timeframe: str = 'H1') -> pd.DataFrame:
    """Load OANDA data for a pair."""
    data_dir = Path(__file__).parent.parent / 'data' / 'oanda'

    # Try parquet first, then CSV
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

    # Ensure datetime index
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


def run_comparison(
    pair: str,
    timeframe: str = 'H1',
    trials: int = 5000,
    min_back_sharpe: float = 1.0,
):
    """Run V1 vs V2 comparison."""
    print("\n" + "="*80)
    print(f"RSI DIVERGENCE V1 vs V2 COMPARISON")
    print(f"Pair: {pair} | Timeframe: {timeframe}")
    print("="*80)

    # Load and split data
    df = load_data(pair, timeframe)
    df_back, df_forward = split_data(df)

    # Initialize strategies
    v1_strategy = RSIDivergenceFullFast()
    v2_strategy = RSIDivergenceV2()

    v1_strategy.set_pip_size(pair)
    v2_strategy.set_pip_size(pair)

    # === Run V1 Optimization ===
    print("\n" + "-"*80)
    print("V1: RSI Divergence Full (Original)")
    print("-"*80)

    v1_optimizer = UnifiedOptimizer(
        v1_strategy,
        min_forward_ratio=0.05,  # Same as V2 for fair comparison
        forward_rank_weight=2.0,  # Same as V2 for fair comparison
    )

    v1_results = v1_optimizer.run(
        df_back, df_forward,
        mode='staged',
        trials_per_stage=trials,
        final_trials=trials,
        min_back_sharpe=min_back_sharpe,
    )

    v1_signals_back = len(v1_optimizer.back_signals)
    v1_signals_fwd = len(v1_optimizer.fwd_signals)

    # === Run V2 Optimization ===
    print("\n" + "-"*80)
    print("V2: RSI Divergence V2 (Regime + Quality Filters)")
    print("-"*80)

    v2_optimizer = UnifiedOptimizer(
        v2_strategy,
        min_forward_ratio=0.05,  # V2: Reject if forward < 5% of back
        forward_rank_weight=2.0,  # V2: Forward matters more
    )

    v2_results = v2_optimizer.run(
        df_back, df_forward,
        mode='staged',
        trials_per_stage=trials,
        final_trials=trials,
        min_back_sharpe=min_back_sharpe,
    )

    v2_signals_back = len(v2_optimizer.back_signals)
    v2_signals_fwd = len(v2_optimizer.fwd_signals)

    # === Print Comparison ===
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    print(f"\n{'Metric':<30} {'V1':<20} {'V2':<20} {'Winner':<10}")
    print("-"*80)

    # Signal counts
    print(f"{'Signals (back)':<30} {v1_signals_back:<20} {v2_signals_back:<20}")
    print(f"{'Signals (forward)':<30} {v1_signals_fwd:<20} {v2_signals_fwd:<20}")

    # Valid results
    print(f"{'Valid results':<30} {len(v1_results):<20} {len(v2_results):<20}")

    if v1_results and v2_results:
        v1_best = v1_results[0]
        v2_best = v2_results[0]

        # Back performance
        v1_back_score = v1_best['back'].ontester_score
        v2_back_score = v2_best['back'].ontester_score
        back_winner = "V1" if v1_back_score > v2_back_score else "V2"
        print(f"{'Back OnTester (best)':<30} {v1_back_score:<20.1f} {v2_back_score:<20.1f} {back_winner:<10}")

        v1_back_sharpe = v1_best['back'].sharpe
        v2_back_sharpe = v2_best['back'].sharpe
        print(f"{'Back Sharpe':<30} {v1_back_sharpe:<20.2f} {v2_back_sharpe:<20.2f}")

        # Forward performance (THE KEY METRIC)
        v1_fwd_score = v1_best['forward'].ontester_score
        v2_fwd_score = v2_best['forward'].ontester_score
        fwd_winner = "V1" if v1_fwd_score > v2_fwd_score else "V2"
        print(f"{'Forward OnTester (best)':<30} {v1_fwd_score:<20.1f} {v2_fwd_score:<20.1f} {fwd_winner:<10}")

        v1_fwd_sharpe = v1_best['forward'].sharpe
        v2_fwd_sharpe = v2_best['forward'].sharpe
        fwd_sharpe_winner = "V1" if v1_fwd_sharpe > v2_fwd_sharpe else "V2"
        print(f"{'Forward Sharpe':<30} {v1_fwd_sharpe:<20.2f} {v2_fwd_sharpe:<20.2f} {fwd_sharpe_winner:<10}")

        # Robustness (forward/back ratio)
        v1_ratio = v1_fwd_score / v1_back_score if v1_back_score > 0 else 0
        v2_ratio = v2_fwd_score / v2_back_score if v2_back_score > 0 else 0
        ratio_winner = "V1" if v1_ratio > v2_ratio else "V2"
        print(f"{'Forward/Back Ratio':<30} {v1_ratio:<20.1%} {v2_ratio:<20.1%} {ratio_winner:<10}")

        # R² consistency
        v1_back_r2 = v1_best['back'].r_squared
        v1_fwd_r2 = v1_best['forward'].r_squared
        v2_back_r2 = v2_best['back'].r_squared
        v2_fwd_r2 = v2_best['forward'].r_squared
        print(f"{'Back R²':<30} {v1_back_r2:<20.3f} {v2_back_r2:<20.3f}")
        print(f"{'Forward R²':<30} {v1_fwd_r2:<20.3f} {v2_fwd_r2:<20.3f}")

        # Trade counts
        v1_back_trades = v1_best['back'].trades
        v1_fwd_trades = v1_best['forward'].trades
        v2_back_trades = v2_best['back'].trades
        v2_fwd_trades = v2_best['forward'].trades
        print(f"{'Back Trades':<30} {v1_back_trades:<20} {v2_back_trades:<20}")
        print(f"{'Forward Trades':<30} {v1_fwd_trades:<20} {v2_fwd_trades:<20}")

        # Drawdown
        v1_back_dd = v1_best['back'].max_dd
        v1_fwd_dd = v1_best['forward'].max_dd
        v2_back_dd = v2_best['back'].max_dd
        v2_fwd_dd = v2_best['forward'].max_dd
        dd_winner = "V2" if v2_fwd_dd < v1_fwd_dd else "V1"
        print(f"{'Back Max DD %':<30} {v1_back_dd:<20.1f} {v2_back_dd:<20.1f}")
        print(f"{'Forward Max DD %':<30} {v1_fwd_dd:<20.1f} {v2_fwd_dd:<20.1f} {dd_winner:<10}")

        print("\n" + "-"*80)
        print("BEST PARAMETERS")
        print("-"*80)

        print("\nV1 Best Params:")
        for k, v in sorted(v1_best['params'].items()):
            print(f"  {k}: {v}")

        print("\nV2 Best Params:")
        for k, v in sorted(v2_best['params'].items()):
            print(f"  {k}: {v}")

    else:
        if not v1_results:
            print("V1: No valid results!")
        if not v2_results:
            print("V2: No valid results!")

    print("\n" + "="*80)

    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if v1_results:
        v1_optimizer.save_results(output_dir / 'v1')
    if v2_results:
        v2_optimizer.save_results(output_dir / 'v2')

    return v1_results, v2_results


def main():
    parser = argparse.ArgumentParser(description='Compare V1 vs V2 RSI Divergence strategies')
    parser.add_argument('--pair', default='EUR_USD', help='Currency pair (default: EUR_USD)')
    parser.add_argument('--timeframe', default='H1', help='Timeframe (default: H1)')
    parser.add_argument('--trials', type=int, default=3000, help='Trials per stage (default: 3000)')
    parser.add_argument('--min-sharpe', type=float, default=1.0, help='Min back Sharpe (default: 1.0)')

    args = parser.parse_args()

    run_comparison(
        pair=args.pair,
        timeframe=args.timeframe,
        trials=args.trials,
        min_back_sharpe=args.min_sharpe,
    )


if __name__ == '__main__':
    main()
