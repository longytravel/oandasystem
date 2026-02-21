"""
Download historical data from OANDA with intelligent caching.

Key features:
- Downloads data once, caches to parquet files
- Incremental updates - only downloads new data since last update
- Progress tracking for long downloads
- Automatic retry on failures
- **Prefers M1 data when available** - builds higher TFs from M1 for accuracy

OANDA Data Capabilities:
========================
- Candle data (M1 to Monthly): Available, ~5+ years history
- Tick data: NOT truly available - OANDA aggregates to max 24 ticks/minute
- Rate limit: 5,000 candles per request (we chunk automatically)
- Demo accounts: May have shorter history limits

Data Strategy:
=============
1. Download M1 data once (largest, most granular)
2. Build any higher timeframe (M5, M15, M30, H1, H4, D) from M1
3. Benefits: More accurate OHLC, single source of truth, any TF on demand

For true tick data, use external sources:
- Dukascopy (free, back to 2003)
- TrueFX (free, limited pairs)
- HistData.com (free archives)
"""
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0].rsplit('\\', 2)[0])

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple
import time

import pandas as pd
from loguru import logger

from live.oanda_client import OandaClient
from config.settings import settings


# Canonical file naming - one file per instrument/timeframe
def get_data_filepath(instrument: str, timeframe: str, data_dir: Path = None) -> Path:
    """Get the canonical filepath for an instrument/timeframe."""
    data_dir = data_dir or settings.DATA_DIR / "oanda"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{instrument}_{timeframe}.parquet"


def load_cached_data(instrument: str, timeframe: str) -> Tuple[Optional[pd.DataFrame], Optional[datetime]]:
    """
    Load cached data and return the last timestamp.

    Returns:
        (DataFrame or None, last_timestamp or None)
    """
    filepath = get_data_filepath(instrument, timeframe)

    if not filepath.exists():
        return None, None

    try:
        df = pd.read_parquet(filepath)
        if df.empty:
            return None, None

        # Get the last timestamp
        last_time = df.index[-1]
        if hasattr(last_time, 'to_pydatetime'):
            last_time = last_time.to_pydatetime()
        if last_time.tzinfo is not None:
            last_time = last_time.replace(tzinfo=None)

        logger.info(f"Loaded {len(df)} cached candles, last: {last_time}")
        return df, last_time

    except Exception as e:
        logger.warning(f"Error loading cache: {e}")
        return None, None


def save_data(df: pd.DataFrame, instrument: str, timeframe: str) -> Path:
    """Save data to the canonical file."""
    filepath = get_data_filepath(instrument, timeframe)
    df.to_parquet(filepath)
    logger.info(f"Saved {len(df)} candles to {filepath}")
    return filepath


def download_with_progress(
    client: OandaClient,
    instrument: str,
    granularity: str,
    from_time: datetime,
    to_time: datetime,
) -> pd.DataFrame:
    """
    Download data with progress tracking and chunking.

    OANDA limits to 5000 candles per request. This function
    calculates proper chunk sizes based on timeframe and downloads
    in manageable pieces.

    Uses disk-based checkpointing to avoid segfaults from repeated
    pd.concat with pyarrow-backed DataFrames (pandas 3.x).
    """
    import tempfile
    import shutil

    current_from = from_time
    total_days = (to_time - from_time).days

    # Candles per minute for each timeframe
    tf_minutes = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D": 1440, "W": 10080
    }
    minutes_per_candle = tf_minutes.get(granularity, 60)

    # Calculate chunk duration to stay under 5000 candles (use 4000 for safety)
    max_candles_per_request = 4000
    chunk_minutes = max_candles_per_request * minutes_per_candle
    chunk_duration = timedelta(minutes=chunk_minutes)

    estimated_candles = int((to_time - from_time).total_seconds() / 60 / minutes_per_candle)
    estimated_requests = max(1, estimated_candles // max_candles_per_request + 1)

    logger.info(f"Downloading {instrument} {granularity}: ~{estimated_candles:,} candles")
    logger.info(f"Period: {from_time.date()} to {to_time.date()}")
    logger.info(f"Chunk size: {max_candles_per_request} candles (~{estimated_requests} requests)")

    # Use temp directory for intermediate parquet files to avoid segfaults
    # from repeated pd.concat with pyarrow-backed DataFrames (pandas 3.x)
    tmp_dir = Path(tempfile.mkdtemp(prefix="oanda_dl_"))
    logger.info(f"Temp dir: {tmp_dir}")

    request_count = 0
    total_candles = 0
    start_time = time.time()
    retry_count = 0
    max_retries = 3
    flush_every = 50  # Save to disk every N API requests
    pending_chunks = []
    part_files = []

    try:
        while current_from < to_time:
            # Calculate chunk end time
            chunk_end = min(current_from + chunk_duration, to_time)

            try:
                # Fetch chunk - no count parameter when using from/to
                df_chunk = client.get_candles(
                    instrument=instrument,
                    granularity=granularity,
                    from_time=current_from,
                    to_time=chunk_end,
                )

                if df_chunk.empty:
                    # No data in this chunk, move forward
                    current_from = chunk_end
                    continue

                pending_chunks.append(df_chunk)
                total_candles += len(df_chunk)
                request_count += 1
                retry_count = 0  # Reset retry counter on success

                # Flush to disk periodically - avoids accumulating DataFrames in memory
                if len(pending_chunks) >= flush_every:
                    part_df = pd.concat(pending_chunks)
                    part_path = tmp_dir / f"part_{len(part_files):04d}.parquet"
                    part_df.to_parquet(part_path)
                    part_files.append(part_path)
                    pending_chunks.clear()
                    logger.info(f"Saved part {len(part_files)} to disk ({len(part_df):,} rows)")

                # Update progress
                last_time = df_chunk.index[-1].to_pydatetime()
                if last_time.tzinfo is not None:
                    last_time = last_time.replace(tzinfo=None)

                downloaded_days = (last_time - from_time).days
                progress = min(100, downloaded_days / total_days * 100) if total_days > 0 else 100

                elapsed = time.time() - start_time
                rate = total_candles / elapsed if elapsed > 0 else 0

                # Log progress every request for long downloads
                logger.info(f"Progress: {progress:.0f}% | {total_candles:,} candles | "
                           f"{rate:.0f}/sec | up to {last_time.date()}")

                # Move to next chunk
                current_from = last_time + timedelta(minutes=minutes_per_candle)

                # Small delay to avoid rate limiting
                time.sleep(0.2)

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed after {max_retries} retries: {e}")
                    # Move forward anyway to avoid infinite loop
                    current_from = chunk_end
                    retry_count = 0
                else:
                    logger.warning(f"Error fetching chunk (attempt {retry_count}): {e}")
                    time.sleep(2)

        # Flush any remaining chunks
        if pending_chunks:
            part_df = pd.concat(pending_chunks)
            part_path = tmp_dir / f"part_{len(part_files):04d}.parquet"
            part_df.to_parquet(part_path)
            part_files.append(part_path)
            pending_chunks.clear()

        if not part_files:
            return pd.DataFrame()

        # Read all parts from disk and combine in one shot
        logger.info(f"Combining {len(part_files)} parts from disk...")
        all_parts = [pd.read_parquet(p) for p in part_files]
        result = pd.concat(all_parts)
        result = result[~result.index.duplicated(keep='last')]
        result = result.sort_index()

        elapsed = time.time() - start_time
        logger.info(f"Download complete: {len(result):,} candles in {elapsed:.1f}s "
                   f"({len(result)/elapsed:.0f}/sec)")

        return result

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


def download_data(
    instrument: str = "GBP_USD",
    granularity: str = "H1",
    years: float = 3,
    force_full: bool = False
) -> pd.DataFrame:
    """
    Download data with intelligent caching.

    If cached data exists, only downloads new data since last update.
    If force_full=True, downloads everything from scratch.

    Args:
        instrument: Currency pair (e.g., "GBP_USD")
        granularity: Timeframe (M1, M5, M15, H1, H4, D)
        years: Years of history to download
        force_full: Force full download, ignore cache

    Returns:
        DataFrame with OHLCV data
    """
    # Initialize client
    client = OandaClient()
    accounts = client.get_accounts()
    if not accounts:
        raise ValueError("No OANDA accounts found. Check your API key.")
    client.account_id = accounts[0]['id']

    # Calculate date range
    end_time = datetime.now(timezone.utc).replace(tzinfo=None)
    start_time = end_time - timedelta(days=365 * years)

    # Check for cached data
    cached_df, last_cached_time = load_cached_data(instrument, granularity)

    if cached_df is not None and not force_full:
        # Only download new data
        if last_cached_time >= end_time - timedelta(hours=2):
            logger.info("Cache is up to date (within 2 hours)")
            return cached_df

        logger.info(f"Updating cache from {last_cached_time}")
        new_df = download_with_progress(
            client, instrument, granularity,
            from_time=last_cached_time + timedelta(minutes=1),
            to_time=end_time
        )

        if not new_df.empty:
            # Merge with cached data
            df = pd.concat([cached_df, new_df])
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()

            # Trim to requested date range (handle timezone)
            start_ts = pd.Timestamp(start_time)
            if df.index.tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize('UTC')
            df = df[df.index >= start_ts]

            save_data(df, instrument, granularity)
            logger.info(f"Updated cache: {len(df):,} total candles")
            return df
        else:
            return cached_df
    else:
        # Full download
        logger.info(f"Full download: {instrument} {granularity}")
        df = download_with_progress(
            client, instrument, granularity,
            from_time=start_time,
            to_time=end_time
        )

        if not df.empty:
            save_data(df, instrument, granularity)

        return df


def update_data(instrument: str = "GBP_USD", granularity: str = "H1") -> pd.DataFrame:
    """
    Quick update - only download new data since last cache.

    This is faster than download_data() as it doesn't specify years
    and just updates from the last cached timestamp.
    """
    return download_data(instrument, granularity, years=0.1, force_full=False)


def build_higher_timeframe(df: pd.DataFrame, target_tf: str = "H1") -> pd.DataFrame:
    """
    Build higher timeframe candles from lower timeframe data.

    Args:
        df: DataFrame with OHLCV (any timeframe)
        target_tf: Target timeframe (M5, M15, M30, H1, H4, D)

    Returns:
        Resampled DataFrame
    """
    tf_map = {
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D": "1D",
        "W": "1W"
    }

    rule = tf_map.get(target_tf)
    if not rule:
        raise ValueError(f"Unknown timeframe: {target_tf}")

    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    return resampled


def load_data(
    instrument: str = "GBP_USD",
    timeframe: str = "H1",
    auto_download: bool = True,
    years: float = 4.0,
    prefer_m1: bool = True
) -> pd.DataFrame:
    """
    Load data from cache, downloading if needed.

    **Prefers M1 data when available** - builds higher timeframes from M1
    for more accurate OHLC values (captures true highs/lows).

    Args:
        instrument: Currency pair (e.g., "GBP_USD")
        timeframe: Target timeframe (M1, M5, M15, M30, H1, H4, D)
        auto_download: If True, download if not cached
        years: Years of data to download if needed
        prefer_m1: If True and M1 exists, build from M1 (default: True)

    Returns:
        DataFrame with OHLCV data indexed by datetime

    Data Source Priority (when prefer_m1=True):
        1. Build from M1 if M1 data exists (most accurate)
        2. Load direct timeframe cache if exists
        3. Auto-download if enabled
    """
    m1_filepath = get_data_filepath(instrument, "M1")
    filepath = get_data_filepath(instrument, timeframe)

    # Priority 1: Build from M1 if available and preferred (most accurate)
    if prefer_m1 and timeframe != "M1" and m1_filepath.exists():
        df_m1 = pd.read_parquet(m1_filepath)
        # Check if M1 data covers the requested period (within 20% tolerance)
        m1_span_years = (df_m1.index[-1] - df_m1.index[0]).days / 365.25 if len(df_m1) > 0 else 0
        if years and m1_span_years < years * 0.8:
            logger.info(f"M1 cache only covers {m1_span_years:.1f}yr, need {years}yr — skipping M1, using direct cache")
        else:
            # Trim M1 data to requested years before building higher timeframe
            if years and len(df_m1) > 0:
                cutoff = df_m1.index[-1] - pd.Timedelta(days=int(years * 365.25))
                df_m1_trimmed = df_m1[df_m1.index >= cutoff]
                if len(df_m1_trimmed) < len(df_m1):
                    logger.info(f"Trimmed M1 to last {years}yr: {len(df_m1_trimmed):,}/{len(df_m1):,} candles")
                    df_m1 = df_m1_trimmed
            df = build_higher_timeframe(df_m1, timeframe)
            logger.info(f"Built {timeframe} from M1 data: {len(df):,} candles")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"(M1 source: {len(df_m1):,} candles)")
            return df

    # Priority 2: Direct timeframe cache
    if filepath.exists():
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df):,} candles from cache")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        if prefer_m1:
            logger.warning(f"No M1 data available - using direct {timeframe} cache")
            logger.warning(f"For better accuracy, download M1: python data/download.py --instrument {instrument} --granularity M1 --years 3")
        return df

    # Priority 3: Auto-download
    if auto_download:
        logger.info(f"No cached data found, downloading {years} years...")
        # Download M1 if prefer_m1 is set and timeframe isn't M1
        if prefer_m1 and timeframe != "M1":
            logger.info(f"Downloading M1 data (will build {timeframe} from it)...")
            df_m1 = download_data(instrument, "M1", years=years)
            df = build_higher_timeframe(df_m1, timeframe)
            logger.info(f"Built {timeframe} from fresh M1: {len(df):,} candles")
            return df
        else:
            return download_data(instrument, timeframe, years=years)

    raise FileNotFoundError(f"No data found for {instrument} {timeframe}")


def get_data_info(instrument: str = "GBP_USD", timeframe: str = "H1") -> dict:
    """Get info about cached data."""
    filepath = get_data_filepath(instrument, timeframe)

    if not filepath.exists():
        return {"exists": False, "filepath": str(filepath)}

    df = pd.read_parquet(filepath)
    file_size = filepath.stat().st_size / (1024 * 1024)  # MB

    return {
        "exists": True,
        "filepath": str(filepath),
        "candles": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "days": (df.index[-1] - df.index[0]).days,
        "file_size_mb": round(file_size, 2)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download OANDA data")
    parser.add_argument("--instrument", default="GBP_USD", help="Instrument")
    parser.add_argument("--granularity", default="H1", help="Timeframe")
    parser.add_argument("--years", type=float, default=2, help="Years of history")
    parser.add_argument("--force", action="store_true", help="Force full download")
    parser.add_argument("--info", action="store_true", help="Show cache info only")
    parser.add_argument("--update", action="store_true", help="Quick update only")

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    if args.info:
        info = get_data_info(args.instrument, args.granularity)
        print(f"\nCache Info: {args.instrument} {args.granularity}")
        print("-" * 40)
        for k, v in info.items():
            print(f"  {k}: {v}")
    elif args.update:
        df = update_data(args.instrument, args.granularity)
        print(f"\nUpdated: {len(df):,} candles")
    else:
        df = download_data(
            instrument=args.instrument,
            granularity=args.granularity,
            years=args.years,
            force_full=args.force
        )
        print(f"\nDownloaded: {len(df):,} candles")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
