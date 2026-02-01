#!/usr/bin/env python
"""
Test script to verify the trading system works end-to-end.

Run this after setting up to verify:
1. OANDA API connection works
2. Data download works
3. Strategy generates signals
4. Backtest runs successfully
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

# Remove default logger and add custom
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def test_api_connection():
    """Test OANDA API connection."""
    print("\n" + "=" * 60)
    print("TEST 1: OANDA API Connection")
    print("=" * 60)

    from live.oanda_client import OandaClient
    from config.settings import settings

    client = OandaClient()

    # Get accounts
    accounts = client.get_accounts()
    print(f"Accounts found: {len(accounts)}")

    if not accounts:
        print("ERROR: No accounts found!")
        return False

    # Get account summary
    client.account_id = accounts[0]['id']
    summary = client.get_account_summary()

    print(f"Account ID: {summary.get('id')}")
    print(f"Balance: {summary.get('balance')} {summary.get('currency')}")
    print(f"NAV: {summary.get('NAV')}")

    print("API Connection OK")
    return True


def test_data_download():
    """Test downloading historical data."""
    print("\n" + "=" * 60)
    print("TEST 2: Data Download")
    print("=" * 60)

    from live.oanda_client import OandaClient

    client = OandaClient()
    accounts = client.get_accounts()
    client.account_id = accounts[0]['id']

    # Fetch 100 H1 candles
    df = client.get_candles("GBP_USD", "H1", count=100)

    print(f"Downloaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.tail())

    if len(df) < 50:
        print("ERROR: Not enough data downloaded!")
        return False

    print("Data Download OK")
    return df


def test_strategy_signals(df: pd.DataFrame):
    """Test strategy signal generation."""
    print("\n" + "=" * 60)
    print("TEST 3: Strategy Signal Generation")
    print("=" * 60)

    from strategies.rsi_divergence import RSIDivergenceStrategy

    # Create strategy with default params
    strategy = RSIDivergenceStrategy()

    print(f"Strategy: {strategy.name}")
    print(f"Parameters: {strategy.params}")

    # Generate signals
    signals = strategy.generate_signals(df)

    print(f"\nSignals generated: {len(signals)}")

    if signals:
        print("\nLast few signals:")
        for signal in signals[-5:]:
            direction = "BUY" if signal.type.value == 1 else "SELL"
            print(f"  {signal.timestamp} | {direction} @ {signal.price:.5f} | "
                  f"SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")

    print("Strategy Signals OK")
    return signals


def test_backtest(df: pd.DataFrame):
    """Test backtesting engine."""
    print("\n" + "=" * 60)
    print("TEST 4: Backtest Engine")
    print("=" * 60)

    from strategies.rsi_divergence import RSIDivergenceStrategy
    from backtesting.engine import BacktestEngine

    # Create strategy
    strategy = RSIDivergenceStrategy()

    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=10000.0,
        spread_pips=1.5,
        slippage_pips=0.5
    )

    # Run backtest
    result = engine.run(strategy, df, risk_per_trade=1.0)

    print(result.summary())

    if result.total_trades == 0:
        print("NOTE: No trades executed (may need more data or different params)")

    print("Backtest Engine OK")
    return result


def test_larger_dataset():
    """Test with more data to get actual trades."""
    print("\n" + "=" * 60)
    print("TEST 5: Larger Dataset Backtest")
    print("=" * 60)

    from live.oanda_client import OandaClient
    from strategies.rsi_divergence import RSIDivergenceStrategy
    from backtesting.engine import BacktestEngine

    # Get more data (6 months of H1)
    client = OandaClient()
    accounts = client.get_accounts()
    client.account_id = accounts[0]['id']

    print("Downloading 6 months of H1 data...")
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=180)

    df = client.get_candles_range(
        instrument="GBP_USD",
        granularity="H1",
        from_time=start_time,
        to_time=end_time
    )

    print(f"Downloaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Run backtest
    strategy = RSIDivergenceStrategy()
    engine = BacktestEngine(initial_capital=10000.0)

    result = engine.run(strategy, df, risk_per_trade=1.0)

    print(result.summary())

    # Show some trades
    if result.trades:
        print("\nSample trades:")
        for trade in result.trades[:10]:
            direction = "BUY" if trade.direction.value == 1 else "SELL"
            print(f"  {trade.entry_time} | {direction} | "
                  f"Entry: {trade.entry_price:.5f} | Exit: {trade.exit_price:.5f} | "
                  f"PnL: {trade.pnl:.2f} ({trade.exit_reason})")

    return result


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("#  FOREX PYTHON TRADING SYSTEM - TEST SUITE")
    print("#" * 60)

    try:
        # Test 1: API
        if not test_api_connection():
            print("\n FAILED: API connection")
            return

        # Test 2: Data
        df = test_data_download()
        if df is None or len(df) == 0:
            print("\n FAILED: Data download")
            return

        # Test 3: Strategy
        signals = test_strategy_signals(df)

        # Test 4: Backtest (small dataset)
        result = test_backtest(df)

        # Test 5: Larger dataset
        result = test_larger_dataset()

        print("\n" + "#" * 60)
        print("#  ALL TESTS PASSED!")
        print("#" * 60)
        print("\nNext steps:")
        print("1. Download more historical data: python data/download.py")
        print("2. Run optimization: python scripts/run_optimization.py")
        print("3. Check PROJECT_PLAN.md for full workflow")

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
