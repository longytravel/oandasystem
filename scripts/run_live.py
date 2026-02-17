#!/usr/bin/env python
"""
Run live trading with configurable strategy.

Usage:
    # Pipeline-optimized V3 RSI (recommended):
    python scripts/run_live.py --strategy rsi_v3 --pair GBP_USD --timeframe M15 --dry-run --once
    python scripts/run_live.py --strategy rsi_v3 --pair GBP_USD --timeframe M15 --dry-run
    python scripts/run_live.py --strategy rsi_v3 --pair GBP_USD --timeframe M15  # Paper trade (practice account)

    # Load params from pipeline run:
    python scripts/run_live.py --strategy rsi_v3 --from-run GBP_USD_M15_20260210_063223 --dry-run

    # Legacy strategies:
    python scripts/run_live.py --pair GBP_USD --timeframe H1 --dry-run
    python scripts/run_live.py --pair GBP_USD --status
"""
import sys
import signal
from pathlib import Path
import argparse
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from loguru import logger

from live.trader import LiveTrader
from config.settings import settings


# Pipeline strategy map: name -> (module, class)
PIPELINE_STRATEGIES = {
    "rsi_v1": ("strategies.archive.rsi_full", "RSIDivergenceFullFast"),  # archived, kept for VPS compat
    "rsi_v3": ("strategies.rsi_full_v3", "RSIDivergenceFullFastV3"),
    "rsi_v4": ("strategies.rsi_full_v4", "RSIDivergenceFullFastV4"),
    "rsi_v5": ("strategies.rsi_full_v5", "RSIDivergenceFullFastV5"),
    "ema_cross": ("strategies.ema_cross_ml", "EMACrossMLStrategy"),
    "fair_price_ma": ("strategies.fair_price_ma", "FairPriceMAStrategy"),
    "donchian_breakout": ("strategies.donchian_breakout", "DonchianBreakoutStrategy"),
    "bollinger_squeeze": ("strategies.bollinger_squeeze", "BollingerSqueezeStrategy"),
    "london_breakout": ("strategies.london_breakout", "LondonBreakoutStrategy"),
    "stochastic_adx": ("strategies.stochastic_adx", "StochasticADXStrategy"),
}

# Legacy strategies (old Strategy interface) - all archived
LEGACY_STRATEGIES = {}


# Global trader instance for signal handling
trader: LiveTrader = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global trader
    if trader:
        logger.info("Shutdown signal received")
        trader.stop()


def setup_logging(verbose: bool = False, log_dir: Path = None):
    """Configure logging.

    Args:
        verbose: Enable DEBUG level output
        log_dir: Directory for log files (defaults to settings.LOGS_DIR)
    """
    logger.remove()

    level = "DEBUG" if verbose else "INFO"

    # Console logging
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    )

    # File logging
    log_dir = log_dir or settings.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "live_trading_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="30 days"
    )


def print_banner(args, pair=None, timeframe=None, strategy_name=None):
    """Print startup banner."""
    print("\n" + "=" * 60)
    print("   OANDA LIVE TRADING SYSTEM")
    print("=" * 60)
    print(f"   Strategy:  {strategy_name or args.strategy}")
    print(f"   Pair:      {pair or args.pair}")
    print(f"   Timeframe: {timeframe or args.timeframe}")
    print(f"   Mode:      {'DRY RUN (no trades)' if args.dry_run else 'PAPER TRADING (practice account)'}")
    print(f"   Risk:      {args.risk}% per trade")
    if args.from_run:
        print(f"   From Run:  {args.from_run}")
    if hasattr(args, 'instance_id') and args.instance_id:
        print(f"   Instance:  {args.instance_id}")
    print("=" * 60)

    if not args.dry_run:
        print("\n   WARNING: LIVE TRADING MODE")
        print("   Real money is at risk!")
        print("=" * 60)


def load_pipeline_run_params(run_id: str) -> dict:
    """Load best candidate params from a pipeline run directory."""
    run_dir = project_root / 'results' / 'pipelines' / run_id
    state_file = run_dir / 'state.json'

    if not state_file.exists():
        raise FileNotFoundError(f"Pipeline run not found: {state_file}")

    with open(state_file) as f:
        state = json.load(f)

    best = state.get('best_candidate', {})
    if not best or not best.get('params'):
        raise ValueError(f"No best candidate in {run_id}")

    return {
        'params': best['params'],
        'pair': state.get('pair', 'GBP_USD'),
        'timeframe': state.get('timeframe', 'H1'),
        'strategy_name': state.get('strategy_name', 'rsi_v3'),
        'score': state.get('final_score', 0),
        'rating': state.get('final_rating', 'RED'),
        'rank': best.get('rank', '?'),
    }


def create_strategy(args):
    """Create strategy from args, handling both pipeline and legacy strategies."""
    all_strategies = list(PIPELINE_STRATEGIES.keys()) + list(LEGACY_STRATEGIES.keys())

    # Load params from pipeline run
    params = None
    pair = args.pair
    timeframe = args.timeframe
    strategy_name = args.strategy

    if args.from_run:
        run_data = load_pipeline_run_params(args.from_run)
        params = run_data['params']
        pair = run_data['pair']
        timeframe = run_data['timeframe']
        # Map pipeline strategy_name to our key
        name_map = {
            'RSI_Divergence_Full': 'rsi_v1',
            'RSI_Divergence_v3': 'rsi_v3',
            'RSI_Divergence_v4': 'rsi_v4',
            'RSI_Divergence_v5': 'rsi_v5',
            'EMA_Cross_ML': 'ema_cross',
            'fair_price_ma': 'fair_price_ma',
            'Donchian_Breakout': 'donchian_breakout',
            'Bollinger_Squeeze': 'bollinger_squeeze',
            'London_Breakout': 'london_breakout',
            'Stochastic_ADX': 'stochastic_adx',
        }
        strategy_name = name_map.get(run_data['strategy_name'], args.strategy)
        logger.info(f"Loaded from pipeline run: {args.from_run}")
        logger.info(f"  Score: {run_data['score']}/100 {run_data['rating']} (Rank #{run_data['rank']})")
        logger.info(f"  Pair: {pair}, TF: {timeframe}, Strategy: {strategy_name}")
    elif args.params_file:
        with open(args.params_file) as f:
            data = json.load(f)
            params = data.get('params', data.get('best_params', data))
        logger.info(f"Loaded parameters from {args.params_file}")

    # Pipeline strategy (FastStrategy + adapter)
    if strategy_name in PIPELINE_STRATEGIES:
        import importlib
        mod_path, cls_name = PIPELINE_STRATEGIES[strategy_name]
        mod = importlib.import_module(mod_path)
        fast_strategy_cls = getattr(mod, cls_name)
        fast_strategy = fast_strategy_cls()

        if params is None:
            # Use defaults from parameter groups
            groups = fast_strategy.get_parameter_groups()
            params = {}
            for g in groups.values():
                params.update(g.get_defaults())
            logger.warning("No params file/run specified, using strategy defaults")

        from live.pipeline_adapter import PipelineStrategyAdapter
        strategy = PipelineStrategyAdapter(fast_strategy, params, pair=pair)
        return strategy, pair, timeframe

    # Legacy strategy
    if strategy_name in LEGACY_STRATEGIES:
        mod_path, cls_name = LEGACY_STRATEGIES[strategy_name]
        import importlib
        mod = importlib.import_module(mod_path)
        strategy_cls = getattr(mod, cls_name)
        strategy = strategy_cls(params)
        return strategy, pair, timeframe

    logger.error(f"Unknown strategy: {strategy_name}. Available: {all_strategies}")
    sys.exit(1)


def main():
    global trader

    all_strategies = list(PIPELINE_STRATEGIES.keys()) + list(LEGACY_STRATEGIES.keys())

    parser = argparse.ArgumentParser(description="Run live/paper trading")
    parser.add_argument("--pair", default="GBP_USD", help="Currency pair")
    parser.add_argument("--timeframe", default="H1", help="Timeframe (M5, M15, H1, H4)")
    parser.add_argument("--strategy", default="rsi_v3",
                       choices=all_strategies,
                       help="Strategy to use (default: rsi_v3)")
    parser.add_argument("--dry-run", action="store_true", help="Log signals without trading")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade (%%)")
    parser.add_argument("--params-file", help="Strategy parameters JSON file")
    parser.add_argument("--from-run", help="Load params from pipeline run ID (e.g., GBP_USD_M15_20260210_063223)")
    parser.add_argument("--candles", type=int, default=500, help="Candles to fetch for signals (default 500)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--once", action="store_true", help="Run once and exit (for testing)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--instance-id", help="Unique instance ID (e.g., rsi_v3_GBP_USD_M15) for multi-strategy isolation")
    parser.add_argument("--instance-dir", help="Base directory for instance state/logs (e.g., instances/rsi_v3_GBP_USD_M15)")

    args = parser.parse_args()

    # Resolve instance directory
    instance_dir = Path(args.instance_dir) if args.instance_dir else None
    instance_id = args.instance_id

    # Setup logging (use instance-specific log dir if provided)
    log_dir = (instance_dir / "logs") if instance_dir else None
    setup_logging(args.verbose, log_dir=log_dir)

    # Create strategy
    try:
        strategy, pair, timeframe = create_strategy(args)
    except Exception as e:
        logger.error(f"Failed to create strategy: {e}")
        sys.exit(1)

    # Override pair/timeframe if explicitly provided
    if args.pair != "GBP_USD" or not args.from_run:
        pair = args.pair
    if args.timeframe != "H1" or not args.from_run:
        timeframe = args.timeframe

    logger.info(f"Using strategy: {strategy.name}")

    # Create trader
    state_dir = str(instance_dir / "state") if instance_dir else None
    trader = LiveTrader(
        strategy=strategy,
        instrument=pair,
        timeframe=timeframe,
        dry_run=args.dry_run,
        candles_needed=args.candles,
        risk_per_trade=args.risk,
        state_dir=state_dir,
        instance_id=instance_id,
    )

    # Status only mode
    if args.status:
        status = trader.get_status()
        print("\n" + "=" * 60)
        print("TRADER STATUS")
        print("=" * 60)
        print(json.dumps(status, indent=2, default=str))
        return

    # Print banner
    print_banner(args, pair=pair, timeframe=timeframe, strategy_name=strategy.name)

    # Confirmation for live trading
    if not args.dry_run and not args.once and not args.yes:
        print("\n")
        confirm = input("Type 'LIVE' to confirm live trading: ")
        if confirm != "LIVE":
            print("Aborted.")
            sys.exit(0)

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    try:
        if args.once:
            # Single iteration for testing
            logger.info("Running single iteration...")
            result = trader.run_once()
            print("\nResult:")
            print(json.dumps(result, indent=2, default=str))
        else:
            # Main trading loop
            trader.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        # Save state
        trader.position_manager.save_state()

        # Print summary
        print("\n" + trader.position_manager.get_daily_summary())


if __name__ == "__main__":
    main()
