#!/usr/bin/env python
"""
Data Manager - Utility for managing OANDA data cache.

Commands:
    status      Show all cached data files and their coverage
    rebuild     Rebuild higher timeframes from M1 data
    update      Update M1 data to latest (incremental)
    download    Download M1 data for an instrument

OANDA Data Strategy:
====================
1. Download M1 data as the source of truth
2. Build higher timeframes (M5, M15, M30, H1, H4) from M1
3. Benefits: More accurate OHLC, single source, any TF on demand

OANDA Limitations:
- Candle data: M1 to Monthly, ~5+ years history
- Tick data: NOT available (max 24 ticks/min, aggregated)
- For true ticks: Use Dukascopy, TrueFX, or HistData.com
"""
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0].rsplit('\\', 2)[0])

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger

from data.download import (
    download_data, load_data, get_data_filepath,
    build_higher_timeframe, save_data
)
from config.settings import settings


def cmd_status(args):
    """Show status of all cached data."""
    data_dir = settings.DATA_DIR / "oanda"

    print("\n" + "=" * 80)
    print("OANDA DATA CACHE STATUS")
    print("=" * 80)

    if not data_dir.exists():
        print("No data directory found!")
        return

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print("No cached data files found!")
        return

    # Group by instrument
    instruments = {}
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            instr = f"{parts[0]}_{parts[1]}"
            tf = parts[2]
        else:
            instr = f.stem
            tf = "?"

        if instr not in instruments:
            instruments[instr] = {}

        df = pd.read_parquet(f)
        size_mb = f.stat().st_size / (1024 * 1024)
        instruments[instr][tf] = {
            'candles': len(df),
            'start': df.index[0],
            'end': df.index[-1],
            'size_mb': size_mb,
            'filepath': f
        }

    for instr, timeframes in sorted(instruments.items()):
        print(f"\n{instr}:")
        print("-" * 70)

        # Check if M1 exists (source of truth)
        has_m1 = 'M1' in timeframes
        if has_m1:
            m1 = timeframes['M1']
            print(f"  [SOURCE] M1: {m1['candles']:>10,} candles | "
                  f"{m1['start'].date()} to {m1['end'].date()} | {m1['size_mb']:.1f}MB")

        for tf in ['M5', 'M15', 'M30', 'H1', 'H4', 'D']:
            if tf in timeframes:
                info = timeframes[tf]
                status = "(can rebuild from M1)" if has_m1 else "(direct download)"
                print(f"  {tf:>6}: {info['candles']:>10,} candles | "
                      f"{info['start'].date()} to {info['end'].date()} | "
                      f"{info['size_mb']:.1f}MB {status}")

        if not has_m1:
            print(f"  [!] No M1 data - recommend: python scripts/data_manager.py download {instr}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("- Download M1 for any missing instruments")
    print("- Run 'rebuild' to regenerate higher TFs from M1")
    print("- Run 'update' to get latest M1 data")
    print("=" * 80 + "\n")


def cmd_rebuild(args):
    """Rebuild higher timeframes from M1 data."""
    instrument = args.instrument.upper().replace("/", "_")

    m1_path = get_data_filepath(instrument, "M1")
    if not m1_path.exists():
        logger.error(f"No M1 data for {instrument}! Download it first:")
        logger.error(f"  python scripts/data_manager.py download {instrument}")
        return

    logger.info(f"Loading M1 data for {instrument}...")
    df_m1 = pd.read_parquet(m1_path)
    logger.info(f"Loaded {len(df_m1):,} M1 candles")

    timeframes = args.timeframes.split(",") if args.timeframes else ["M5", "M15", "M30", "H1", "H4"]

    for tf in timeframes:
        tf = tf.strip().upper()
        if tf == "M1":
            continue

        logger.info(f"Building {tf} from M1...")
        df = build_higher_timeframe(df_m1, tf)

        # Save to cache
        save_data(df, instrument, tf)
        logger.info(f"  Saved {len(df):,} {tf} candles")

    logger.info("Rebuild complete!")


def cmd_update(args):
    """Update M1 data to latest."""
    instrument = args.instrument.upper().replace("/", "_")

    logger.info(f"Updating M1 data for {instrument}...")
    df = download_data(
        instrument=instrument,
        granularity="M1",
        years=0.1,  # Just get recent data (incremental)
        force_full=False
    )

    logger.info(f"M1 data updated: {len(df):,} candles")
    logger.info(f"Latest: {df.index[-1]}")

    # Optionally rebuild higher TFs
    if args.rebuild:
        args.timeframes = None
        cmd_rebuild(args)


def cmd_download(args):
    """Download M1 data for an instrument."""
    instrument = args.instrument.upper().replace("/", "_")
    years = args.years

    logger.info(f"Downloading {years} years of M1 data for {instrument}...")
    df = download_data(
        instrument=instrument,
        granularity="M1",
        years=years,
        force_full=args.force
    )

    logger.info(f"Downloaded {len(df):,} M1 candles")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Optionally build higher TFs
    if args.build:
        args.timeframes = None
        cmd_rebuild(args)


def main():
    parser = argparse.ArgumentParser(
        description="OANDA Data Manager - Manage cached data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all cached data
  python scripts/data_manager.py status

  # Download 3 years of M1 data for GBP/USD
  python scripts/data_manager.py download GBP_USD --years 3 --build

  # Rebuild H1 from existing M1 data
  python scripts/data_manager.py rebuild GBP_USD --timeframes H1,H4

  # Update M1 to latest and rebuild all TFs
  python scripts/data_manager.py update GBP_USD --rebuild

Data Strategy:
  1. Download M1 as source of truth (most granular OANDA offers)
  2. Build higher TFs from M1 for accurate OHLC values
  3. The pipeline auto-detects M1 and uses it when available

Note: OANDA doesn't provide true tick data (max 24 ticks/min).
For tick data, use Dukascopy, TrueFX, or HistData.com
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # status
    p_status = subparsers.add_parser('status', help='Show cached data status')
    p_status.set_defaults(func=cmd_status)

    # rebuild
    p_rebuild = subparsers.add_parser('rebuild', help='Rebuild higher TFs from M1')
    p_rebuild.add_argument('instrument', help='Instrument (e.g., GBP_USD)')
    p_rebuild.add_argument('--timeframes', '-t', default=None,
                          help='Timeframes to build (default: M5,M15,M30,H1,H4)')
    p_rebuild.set_defaults(func=cmd_rebuild)

    # update
    p_update = subparsers.add_parser('update', help='Update M1 data to latest')
    p_update.add_argument('instrument', help='Instrument (e.g., GBP_USD)')
    p_update.add_argument('--rebuild', '-r', action='store_true',
                         help='Also rebuild higher TFs after update')
    p_update.set_defaults(func=cmd_update)

    # download
    p_download = subparsers.add_parser('download', help='Download M1 data')
    p_download.add_argument('instrument', help='Instrument (e.g., GBP_USD)')
    p_download.add_argument('--years', '-y', type=float, default=2,
                           help='Years of history (default: 2)')
    p_download.add_argument('--force', '-f', action='store_true',
                           help='Force full re-download')
    p_download.add_argument('--build', '-b', action='store_true',
                           help='Also build higher TFs after download')
    p_download.set_defaults(func=cmd_download)

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    if args.command is None:
        # Default to status
        cmd_status(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
