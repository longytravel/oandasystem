#!/usr/bin/env python
"""
OANDA Trading System - E2E Pipeline CLI

Run the full 7-stage validation pipeline for any strategy.

Usage:
    # Full pipeline
    python scripts/run_pipeline.py --pair GBP_USD --timeframe H1

    # With custom settings
    python scripts/run_pipeline.py --pair EUR_USD --timeframe H1 --trials 5000 --mc-iterations 1000

    # Resume from stage
    python scripts/run_pipeline.py --resume-from walkforward --run-dir results/pipelines/GBP_USD_H1_20260204/

    # Stop after specific stage
    python scripts/run_pipeline.py --pair GBP_USD --stop-after stability
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from pipeline.pipeline import Pipeline
from pipeline.config import PipelineConfig
from pipeline.state import STAGE_ORDER


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()

    level = "DEBUG" if verbose else "INFO"
    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>"

    logger.add(
        sys.stdout,
        format=fmt,
        level=level,
        colorize=True,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OANDA Trading System - E2E Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pair GBP_USD --timeframe H1
  %(prog)s --pair EUR_USD --trials 3000 --mc-iterations 500
  %(prog)s --resume-from stability --run-dir results/pipelines/GBP_USD_H1_20260204/
        """
    )

    # Basic settings
    parser.add_argument(
        '--pair', '-p',
        default='GBP_USD',
        help='Currency pair (default: GBP_USD)'
    )
    parser.add_argument(
        '--timeframe', '-t',
        default='H1',
        choices=['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D'],
        help='Timeframe (default: H1)'
    )
    parser.add_argument(
        '--strategy', '-s',
        default='RSI_Divergence_Full',
        help='Strategy name (default: RSI_Divergence_Full)'
    )

    # Data settings
    parser.add_argument(
        '--years',
        type=float,
        default=2.0,
        help='Years of historical data (default: 2.0)'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force full data download (ignore cache)'
    )

    # Optimization settings
    parser.add_argument(
        '--trials',
        type=int,
        default=5000,
        help='Trials per optimization stage (default: 5000)'
    )
    parser.add_argument(
        '--final-trials',
        type=int,
        default=10000,
        help='Final optimization trials (default: 10000)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=50,
        help='Top N candidates to validate (default: 50)'
    )

    # Walk-forward settings
    parser.add_argument(
        '--train-months',
        type=int,
        default=6,
        help='Training window months (default: 6)'
    )
    parser.add_argument(
        '--test-months',
        type=int,
        default=3,
        help='Test window months (default: 3)'
    )

    # Monte Carlo settings
    parser.add_argument(
        '--mc-iterations',
        type=int,
        default=500,
        help='Monte Carlo iterations (default: 500)'
    )

    # Performance settings
    parser.add_argument(
        '--n-jobs', '-j',
        type=int,
        default=1,
        help='Parallel workers for optimization (-1 = all cores, default: 1)'
    )

    # Flow control
    parser.add_argument(
        '--resume-from',
        choices=STAGE_ORDER,
        help='Resume from specific stage'
    )
    parser.add_argument(
        '--stop-after',
        choices=STAGE_ORDER,
        help='Stop after specific stage'
    )
    parser.add_argument(
        '--run-dir',
        type=Path,
        help='Existing run directory (for resuming)'
    )

    # ML exit
    parser.add_argument(
        '--ml-exit',
        action='store_true',
        help='Enable ML-based exit strategy (trains per walk-forward window)'
    )
    parser.add_argument(
        '--policy-mode',
        default='dual_model',
        choices=['dual_model', 'risk_only', 'hold_only'],
        help='ML exit policy mode (default: dual_model)'
    )
    parser.add_argument(
        '--ml-cooldown',
        type=int,
        default=10,
        help='Bars to skip new entries after ML exit (prevents trade inflation, default: 10)'
    )
    parser.add_argument(
        '--ml-mode',
        default='exit',
        choices=['exit', 'entry_filter', 'adaptive_sl', 'adaptive_sl_reg'],
        help='ML mode: exit = per-bar exit timing, entry_filter = signal filter, adaptive_sl = ML-tightened SL (classify), adaptive_sl_reg = ML-tightened SL (regress MAE) (default: exit)'
    )
    parser.add_argument(
        '--signal-filter-threshold',
        type=float,
        default=0.5,
        help='Probability threshold for signal filter (default: 0.5)'
    )

    # OOS holdout
    parser.add_argument(
        '--holdout-months',
        type=int,
        default=0,
        help='Reserve last N months as holdout for OOS validation (default: 0 = 80/20 split)'
    )

    # Run description
    parser.add_argument(
        '--description', '-d',
        default='',
        help='Human-readable run description (shown in leaderboard and report header)'
    )

    # Spread and slippage
    parser.add_argument(
        '--spread', type=float, default=None,
        help='Spread in pips (default: auto from config, typically 1.5)'
    )
    parser.add_argument(
        '--slippage', type=float, default=None,
        help='Slippage in pips (default: 0.5)'
    )

    # Speed presets
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: fewer trials (2000/stage, 5000 final, 250 MC). Good for dev/testing.'
    )

    # Output settings
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


def build_config(args) -> PipelineConfig:
    """Build pipeline config from CLI args."""
    config = PipelineConfig(
        pair=args.pair,
        timeframe=args.timeframe,
        strategy_name=args.strategy,
    )

    # Apply --fast preset before explicit args (explicit args override)
    if args.fast:
        config.optimization.trials_per_stage = 2000
        config.optimization.final_trials = 5000
        config.montecarlo.iterations = 250
        logger.info("FAST MODE: trials=2000/stage, final=5000, MC=250")

    # Data config
    config.data.years = args.years
    config.data.force_download = args.force_download

    # Optimization config (explicit CLI args override --fast defaults)
    if not args.fast or args.trials != 5000:  # 5000 is the argparse default
        config.optimization.trials_per_stage = args.trials
    if not args.fast or args.final_trials != 10000:
        config.optimization.final_trials = args.final_trials
    config.optimization.top_n_candidates = args.top_n
    config.optimization.n_jobs = args.n_jobs

    # Walk-forward config
    config.walkforward.train_months = args.train_months
    config.walkforward.test_months = args.test_months

    # Monte Carlo config (explicit CLI args override --fast defaults)
    if not args.fast or args.mc_iterations != 500:
        config.montecarlo.iterations = args.mc_iterations

    # Data holdout config
    config.data.holdout_months = args.holdout_months

    # ML exit config
    if args.ml_exit:
        config.ml_exit.enabled = True
    config.ml_exit.policy_mode = args.policy_mode
    config.ml_exit.ml_exit_cooldown_bars = args.ml_cooldown
    config.ml_exit.ml_mode = args.ml_mode
    config.ml_exit.signal_filter_threshold = args.signal_filter_threshold

    # Output config
    if args.output_dir:
        config.output_dir = args.output_dir

    # Spread and slippage overrides
    if args.spread is not None:
        config.spread_pips = args.spread
    if args.slippage is not None:
        config.slippage_pips = args.slippage

    # Description
    config.description = args.description

    return config


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Build config
    config = build_config(args)

    # Print startup info
    logger.info(f"Starting pipeline for {config.pair} {config.timeframe}")
    logger.info(f"Strategy: {config.strategy_name}")
    logger.info(f"Trials/stage: {config.optimization.trials_per_stage}")
    logger.info(f"Monte Carlo iterations: {config.montecarlo.iterations}")

    # Create and run pipeline
    pipeline = Pipeline(config)

    try:
        result = pipeline.run(
            resume_from=args.resume_from,
            stop_after=args.stop_after,
            run_dir=args.run_dir,
        )

        # Print result
        if result.get('report_path'):
            logger.info(f"\nReport generated: {result['report_path']}")
            logger.info(f"Open in browser: file://{result['report_path']}")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\nPipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
