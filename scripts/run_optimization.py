#!/usr/bin/env python
"""
Unified Optimization Entry Point.

Single script for ALL optimization modes:
- Quick: Basic SL/TP, ~700 trials/sec (for signal param tuning)
- Full: All trade management, ~400 trials/sec (for full param tuning)
- Staged: Group-by-group optimization with full features

Usage:
    # Quick test with 9-param strategy
    python scripts/run_optimization.py --quick

    # Full 35-param optimization
    python scripts/run_optimization.py --full --trials 30000

    # Staged group optimization (recommended for 35+ params)
    python scripts/run_optimization.py --staged --trials-per-stage 5000

    # List available strategies
    python scripts/run_optimization.py --list-strategies

    # Show parameter groups
    python scripts/run_optimization.py --show-groups --strategy rsi_full
"""
import sys
from pathlib import Path
import argparse
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from optimization.unified_optimizer import UnifiedOptimizer
from strategies.rsi_fast import get_strategy, list_strategies
from data.download import load_data
from config.settings import settings


def show_groups(strategy):
    """Display parameter groups for a strategy."""
    groups = strategy.get_parameter_groups()
    if not groups:
        print(f"\nStrategy '{strategy.name}' does not support staged optimization.")
        print("Use --quick or --full mode instead.\n")
        return

    print(f"\n{'='*70}")
    print(f"PARAMETER GROUPS - {strategy.name}")
    print(f"{'='*70}")

    total_params = 0
    total_combos = 1
    for name, group in groups.items():
        n_params = len(group.parameters)
        combos = group.get_space_size()
        total_params += n_params
        total_combos *= combos

        print(f"\n{name.upper()} - {group.description}")
        print("-" * 50)
        for param_name, param_def in group.parameters.items():
            vals = param_def.values
            if len(vals) > 6:
                vals_str = f"[{vals[0]}, {vals[1]}, ..., {vals[-1]}] ({len(vals)} values)"
            else:
                vals_str = str(vals)
            print(f"  {param_name:<25} default={param_def.default:<10} {vals_str}")
        print(f"  Combinations: {combos:,}")

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_params} parameters, {total_combos:,} combinations")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Strategy Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick 9-param optimization (~700 trials/sec)
  python scripts/run_optimization.py --quick --trials 10000

  # Full 35-param optimization (~400 trials/sec)
  python scripts/run_optimization.py --full --trials 30000

  # Staged group optimization (recommended)
  python scripts/run_optimization.py --staged --trials-per-stage 5000

  # Use grid sampling instead of Optuna
  python scripts/run_optimization.py --full --no-optuna

  # Show strategy parameter groups
  python scripts/run_optimization.py --show-groups --strategy rsi_full
"""
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true",
                          help="Quick mode: basic SL/TP only (~700 trials/sec)")
    mode_group.add_argument("--full", action="store_true",
                          help="Full mode: all trade management (~400 trials/sec)")
    mode_group.add_argument("--staged", action="store_true",
                          help="Staged mode: group-by-group optimization")

    # Strategy selection
    parser.add_argument("--strategy", default="rsi_full",
                       help="Strategy name (default: rsi_full)")
    parser.add_argument("--list-strategies", action="store_true",
                       help="List available strategies")
    parser.add_argument("--show-groups", action="store_true",
                       help="Show parameter groups for strategy")

    # Data options
    parser.add_argument("--pair", default="GBP_USD", help="Currency pair")
    parser.add_argument("--timeframe", default="H1", help="Timeframe")
    parser.add_argument("--forward-months", type=int, default=6,
                       help="Months of data for forward testing")

    # Optimization options
    parser.add_argument("--trials", type=int, default=10000,
                       help="Number of trials (quick/full mode)")
    parser.add_argument("--trials-per-stage", type=int, default=5000,
                       help="Trials per stage (staged mode)")
    parser.add_argument("--final-trials", type=int, default=10000,
                       help="Final stage trials (staged mode)")
    parser.add_argument("--top-n", type=int, default=200,
                       help="Top N to forward test")
    parser.add_argument("--min-trades", type=int, default=20,
                       help="Minimum trades for valid result")
    parser.add_argument("--min-sharpe", type=float, default=1.0,
                       help="Minimum back Sharpe to qualify")

    # Sampling options
    parser.add_argument("--no-optuna", action="store_true",
                       help="Use grid sampling instead of Optuna TPE")

    # Output options
    parser.add_argument("--show-importance", action="store_true",
                       help="Show parameter importance")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to file")

    args = parser.parse_args()

    # Handle list-strategies
    if args.list_strategies:
        print("\nAvailable strategies:")
        print("-" * 60)
        for name in list_strategies():
            strat = get_strategy(name)
            n_params = len(strat.get_parameter_space())
            groups = strat.get_parameter_groups()
            staged = "Yes" if groups else "No"
            print(f"  {name:<20} {strat.name} ({n_params} params, staged={staged})")
        print()
        return

    # Load strategy
    try:
        strategy = get_strategy(args.strategy)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Handle show-groups
    if args.show_groups:
        show_groups(strategy)
        return

    # Determine mode
    if args.staged:
        mode = 'staged'
    elif args.full:
        mode = 'full'
    elif args.quick:
        mode = 'quick'
    else:
        # Default based on strategy support
        if strategy.get_parameter_groups():
            mode = 'staged'
        else:
            mode = 'quick'

    # Check staged support
    if mode == 'staged' and not strategy.get_parameter_groups():
        print(f"Error: Strategy '{strategy.name}' doesn't support staged optimization")
        print("Use --quick or --full mode instead")
        return

    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {message}")

    n_params = len(strategy.get_parameter_space())
    use_optuna = not args.no_optuna

    # Print header
    print("\n" + "#" * 70)
    print(f"# UNIFIED OPTIMIZATION - Mode: {mode.upper()}")
    print(f"# Strategy: {strategy.name} ({n_params} parameters)")
    print(f"# {args.pair} {args.timeframe}")
    if mode == 'staged':
        groups = strategy.get_parameter_groups()
        print(f"# Stages: {len(groups)} groups, {args.trials_per_stage:,}/stage + {args.final_trials:,} final")
    else:
        sampling = "Optuna TPE" if use_optuna else "Grid"
        print(f"# Trials: {args.trials:,} | Sampling: {sampling}")
    print("#" * 70 + "\n")

    # Load data
    logger.info(f"Loading {args.pair} {args.timeframe}...")
    df = load_data(args.pair, args.timeframe)
    logger.info(f"Loaded {len(df):,} bars")

    # Split data
    total = len(df)
    fwd_bars = int(total * args.forward_months / 24)
    df_back = df.iloc[:total - fwd_bars]
    df_forward = df.iloc[total - fwd_bars:]

    print(f"Data split:")
    print(f"  Back:    {len(df_back):,} bars ({df_back.index[0].date()} to {df_back.index[-1].date()})")
    print(f"  Forward: {len(df_forward):,} bars ({df_forward.index[0].date()} to {df_forward.index[-1].date()})\n")

    # Set pip size
    strategy.set_pip_size(args.pair)

    # Create optimizer
    optimizer = UnifiedOptimizer(
        strategy=strategy,
        min_trades=args.min_trades,
    )

    # Run optimization
    start = time.time()

    results = optimizer.run(
        df_back=df_back,
        df_forward=df_forward,
        mode=mode,
        n_trials=args.trials,
        trials_per_stage=args.trials_per_stage,
        final_trials=args.final_trials,
        top_n=args.top_n,
        min_back_sharpe=args.min_sharpe,
        use_optuna=use_optuna,
    )

    total_time = time.time() - start

    if not results:
        print("\nNo valid results!")
        return

    # Print results
    if mode == 'staged':
        optimizer.print_stage_summary()

    optimizer.print_results(top_n=30)

    if args.show_importance:
        optimizer.print_param_importance()

    # Best result details
    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST FORWARD RESULT: Trial #{best['trial_id']}")
    print(f"{'='*70}")

    print(f"\nParameters ({len(best['params'])} total):")
    for k, v in sorted(best['params'].items()):
        print(f"  {k}: {v}")

    print(f"\nPerformance:")
    print(f"  {'Metric':<20} {'BACK':<15} {'FORWARD':<15}")
    print(f"  {'-'*50}")
    print(f"  {'Sharpe':<20} {best['back'].sharpe:<15.2f} {best['forward'].sharpe:<15.2f}")
    print(f"  {'Win Rate':<20} {best['back'].win_rate*100:<14.1f}% {best['forward'].win_rate*100:<14.1f}%")
    print(f"  {'Profit Factor':<20} {best['back'].profit_factor:<15.2f} {best['forward'].profit_factor:<15.2f}")
    print(f"  {'Max Drawdown':<20} {best['back'].max_dd:<14.1f}% {best['forward'].max_dd:<14.1f}%")
    print(f"  {'Trades':<20} {best['back'].trades:<15} {best['forward'].trades:<15}")

    # Save results
    if not args.no_save:
        output_dir = settings.RESULTS_DIR / "optimization"
        optimizer.save_results(output_dir)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Strategy:      {strategy.name}")
    print(f"Mode:          {mode}")
    print(f"Parameters:    {n_params}")
    print(f"Total time:    {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Back tested:   {len(optimizer.all_back_results):,}")
    print(f"Valid results: {len(results):,}")


if __name__ == "__main__":
    main()
