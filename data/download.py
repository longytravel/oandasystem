"""
Download historical data from OANDA.
"""
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0].rsplit('\\', 2)[0])

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from loguru import logger

from live.oanda_client import OandaClient
from config.settings import settings


def download_data(
    instrument: str = "GBP_USD",
    granularity: str = "M1",  # 1-minute for building higher TFs
    years: int = 3,
    output_dir: Path = None
) -> Path:
    """
    Download historical data from OANDA and save to parquet.

    Args:
        instrument: e.g., "GBP_USD"
        granularity: M1, M5, M15, H1, etc.
        years: How many years of history
        output_dir: Where to save (defaults to data/oanda/)

    Returns:
        Path to saved file
    """
    output_dir = output_dir or settings.DATA_DIR / "oanda"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OandaClient()

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=365 * years)

    logger.info(f"Downloading {instrument} {granularity}")
    logger.info(f"Period: {start_time.date()} to {end_time.date()}")

    # Fetch data (handles pagination internally)
    df = client.get_candles_range(
        instrument=instrument,
        granularity=granularity,
        from_time=start_time,
        to_time=end_time
    )

    if df.empty:
        logger.error("No data received!")
        return None

    logger.info(f"Downloaded {len(df)} candles")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Save to parquet
    filename = f"{instrument}_{granularity}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.parquet"
    filepath = output_dir / filename

    df.to_parquet(filepath)
    logger.info(f"Saved to {filepath}")

    return filepath


def build_higher_timeframe(
    df_m1: pd.DataFrame,
    target_tf: str = "H1"
) -> pd.DataFrame:
    """
    Build higher timeframe candles from 1-minute data.

    Args:
        df_m1: DataFrame with 1-minute OHLCV
        target_tf: Target timeframe (M5, M15, M30, H1, H4, D)

    Returns:
        Resampled DataFrame
    """
    # Map timeframe to pandas resample rule
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

    resampled = df_m1.resample(rule).agg({
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
    data_dir: Path = None
) -> pd.DataFrame:
    """
    Load data from parquet file, building higher TF if needed.
    """
    data_dir = data_dir or settings.DATA_DIR / "oanda"

    # Look for most recent file for this instrument
    pattern = f"{instrument}_M1_*.parquet"
    files = list(data_dir.glob(pattern))

    if not files:
        # Try looking for the exact timeframe
        pattern = f"{instrument}_{timeframe}_*.parquet"
        files = list(data_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No data files found for {instrument}")

    # Use most recent file
    filepath = sorted(files)[-1]
    logger.info(f"Loading {filepath}")

    df = pd.read_parquet(filepath)

    # Build higher timeframe if needed
    if timeframe != "M1" and "M1" in filepath.name:
        df = build_higher_timeframe(df, timeframe)
        logger.info(f"Built {timeframe} data: {len(df)} candles")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download OANDA data")
    parser.add_argument("--instrument", default="GBP_USD", help="Instrument")
    parser.add_argument("--granularity", default="M1", help="Timeframe")
    parser.add_argument("--years", type=int, default=3, help="Years of history")

    args = parser.parse_args()

    logger.add("logs/download.log", rotation="10 MB")

    download_data(
        instrument=args.instrument,
        granularity=args.granularity,
        years=args.years
    )
