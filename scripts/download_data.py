#!/usr/bin/env python
"""
Download historical data from OANDA.

Uses intelligent caching - appends new data to existing cache.

Usage:
    python scripts/download_data.py --pair GBP_USD --years 3
    python scripts/download_data.py --pair EUR_USD --granularity H1 --years 2
    python scripts/download_data.py --pair GBP_USD --info  # Show cache info
    python scripts/download_data.py --pair GBP_USD --update  # Quick update
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from loguru import logger

from data.download import download_data, update_data, get_data_info


def main():
    parser = argparse.ArgumentParser(description="Download OANDA historical data")
    parser.add_argument("--pair", default="GBP_USD", help="Currency pair (e.g., GBP_USD)")
    parser.add_argument("--granularity", default="H1", help="Timeframe (M1, M5, M15, H1, H4, D)")
    parser.add_argument("--years", type=float, default=2, help="Years of history to download")
    parser.add_argument("--force", action="store_true", help="Force full re-download (ignore cache)")
    parser.add_argument("--info", action="store_true", help="Show cache info only")
    parser.add_argument("--update", action="store_true", help="Quick update (just new data)")

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    if args.info:
        # Show cache info
        info = get_data_info(args.pair, args.granularity)
        print(f"\n{'='*50}")
        print(f"CACHE INFO: {args.pair} {args.granularity}")
        print(f"{'='*50}")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print(f"{'='*50}\n")
        return

    if args.update:
        # Quick update
        print(f"\nUpdating {args.pair} {args.granularity}...")
        df = update_data(args.pair, args.granularity)
    else:
        # Full download (with append if cache exists)
        df = download_data(
            instrument=args.pair,
            granularity=args.granularity,
            years=args.years,
            force_full=args.force
        )

    # Print summary
    print(f"\n{'='*50}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    print(f"Pair:      {args.pair}")
    print(f"Timeframe: {args.granularity}")
    print(f"Candles:   {len(df):,}")
    print(f"Period:    {df.index[0].date()} to {df.index[-1].date()}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
