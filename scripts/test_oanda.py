#!/usr/bin/env python
"""
Test OANDA API connection end-to-end before live trading.

Usage:
    python scripts/test_oanda.py                 # Read-only tests
    python scripts/test_oanda.py --test-trade    # Place small test trade (requires confirmation)
"""
import sys
from pathlib import Path
import argparse
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from live.oanda_client import OandaClient
from config.settings import settings


def test_connection(client: OandaClient) -> bool:
    """Test 1: Basic API connection."""
    print("\n" + "=" * 50)
    print("TEST 1: API Connection")
    print("=" * 50)

    try:
        accounts = client.get_accounts()
        print(f"  Status: PASS")
        print(f"  Accounts found: {len(accounts)}")
        for acc in accounts:
            print(f"    - {acc['id']}")
        return True
    except Exception as e:
        print(f"  Status: FAIL")
        print(f"  Error: {e}")
        return False


def test_account_info(client: OandaClient) -> bool:
    """Test 2: Account info retrieval."""
    print("\n" + "=" * 50)
    print("TEST 2: Account Info")
    print("=" * 50)

    try:
        account = client.get_account_summary()
        print(f"  Status: PASS")
        print(f"  Balance: {float(account.get('balance', 0)):,.2f} {account.get('currency', 'USD')}")
        print(f"  NAV: {float(account.get('NAV', 0)):,.2f}")
        print(f"  Unrealized P&L: {float(account.get('unrealizedPL', 0)):,.2f}")
        print(f"  Margin Used: {float(account.get('marginUsed', 0)):,.2f}")
        print(f"  Margin Available: {float(account.get('marginAvailable', 0)):,.2f}")
        print(f"  Open Trades: {account.get('openTradeCount', 0)}")
        print(f"  Open Positions: {account.get('openPositionCount', 0)}")
        return True
    except Exception as e:
        print(f"  Status: FAIL")
        print(f"  Error: {e}")
        return False


def test_live_price(client: OandaClient, instrument: str = "GBP_USD") -> bool:
    """Test 3: Live price fetch."""
    print("\n" + "=" * 50)
    print(f"TEST 3: Live Price ({instrument})")
    print("=" * 50)

    try:
        price = client.get_price(instrument)
        if not price:
            print(f"  Status: FAIL")
            print(f"  Error: No price data returned")
            return False

        bid = float(price.get('bids', [{}])[0].get('price', 0))
        ask = float(price.get('asks', [{}])[0].get('price', 0))
        spread = (ask - bid) * 10000  # Convert to pips

        print(f"  Status: PASS")
        print(f"  Bid: {bid:.5f}")
        print(f"  Ask: {ask:.5f}")
        print(f"  Spread: {spread:.1f} pips")
        print(f"  Time: {price.get('time', 'N/A')}")
        return True
    except Exception as e:
        print(f"  Status: FAIL")
        print(f"  Error: {e}")
        return False


def test_historical_candles(client: OandaClient, instrument: str = "GBP_USD") -> bool:
    """Test 4: Historical candles."""
    print("\n" + "=" * 50)
    print(f"TEST 4: Historical Candles ({instrument} H1)")
    print("=" * 50)

    try:
        df = client.get_candles(instrument, "H1", count=10)
        if df.empty:
            print(f"  Status: FAIL")
            print(f"  Error: No candle data returned")
            return False

        print(f"  Status: PASS")
        print(f"  Candles received: {len(df)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Latest candle:")
        latest = df.iloc[-1]
        print(f"    Open:  {latest['open']:.5f}")
        print(f"    High:  {latest['high']:.5f}")
        print(f"    Low:   {latest['low']:.5f}")
        print(f"    Close: {latest['close']:.5f}")
        print(f"    Volume: {latest['volume']}")
        return True
    except Exception as e:
        print(f"  Status: FAIL")
        print(f"  Error: {e}")
        return False


def test_open_positions(client: OandaClient) -> bool:
    """Test 5: Open positions query."""
    print("\n" + "=" * 50)
    print("TEST 5: Open Positions")
    print("=" * 50)

    try:
        positions = client.get_open_positions()
        trades = client.get_open_trades()

        print(f"  Status: PASS")
        print(f"  Open positions: {len(positions)}")
        print(f"  Open trades: {len(trades)}")

        if positions:
            print(f"  Positions:")
            for pos in positions:
                instrument = pos.get('instrument', 'N/A')
                long_units = pos.get('long', {}).get('units', '0')
                short_units = pos.get('short', {}).get('units', '0')
                unrealized_pl = float(pos.get('unrealizedPL', 0))
                print(f"    - {instrument}: Long {long_units} / Short {short_units} | P&L: {unrealized_pl:+.2f}")

        if trades:
            print(f"  Trades:")
            for trade in trades:
                trade_id = trade.get('id', 'N/A')
                instrument = trade.get('instrument', 'N/A')
                units = trade.get('currentUnits', '0')
                unrealized_pl = float(trade.get('unrealizedPL', 0))
                print(f"    - #{trade_id} {instrument}: {units} units | P&L: {unrealized_pl:+.2f}")

        return True
    except Exception as e:
        print(f"  Status: FAIL")
        print(f"  Error: {e}")
        return False


def test_place_trade(client: OandaClient, instrument: str = "GBP_USD") -> bool:
    """Test 6: Place and close a small test trade."""
    print("\n" + "=" * 50)
    print(f"TEST 6: Test Trade ({instrument})")
    print("=" * 50)

    # Get current price for SL/TP calculation
    try:
        price_data = client.get_price(instrument)
        if not price_data:
            print(f"  Status: FAIL")
            print(f"  Error: Could not get current price")
            return False

        ask = float(price_data.get('asks', [{}])[0].get('price', 0))

        # Calculate SL/TP (20 pips each)
        pip_size = 0.0001 if 'JPY' not in instrument else 0.01
        sl_price = ask - (20 * pip_size)
        tp_price = ask + (20 * pip_size)

        print(f"  Current Ask: {ask:.5f}")
        print(f"  Stop Loss: {sl_price:.5f}")
        print(f"  Take Profit: {tp_price:.5f}")
        print(f"  Units: 1 (minimum)")

    except Exception as e:
        print(f"  Status: FAIL")
        print(f"  Error getting price: {e}")
        return False

    # Place the order
    try:
        print(f"\n  Placing BUY order...")
        result = client.market_order(
            instrument=instrument,
            units=1,  # Minimum size
            stop_loss_price=sl_price,
            take_profit_price=tp_price
        )

        if 'orderFillTransaction' in result:
            fill = result['orderFillTransaction']
            trade_id = fill.get('tradeOpened', {}).get('tradeID')
            fill_price = float(fill.get('price', 0))
            print(f"  Order filled at: {fill_price:.5f}")
            print(f"  Trade ID: {trade_id}")
        elif 'orderCancelTransaction' in result:
            cancel = result['orderCancelTransaction']
            reason = cancel.get('reason', 'Unknown')
            print(f"  Status: FAIL")
            print(f"  Order cancelled: {reason}")
            return False
        else:
            print(f"  Status: FAIL")
            print(f"  Unexpected response: {result}")
            return False

    except Exception as e:
        print(f"  Status: FAIL")
        print(f"  Error placing order: {e}")
        return False

    # Wait a moment
    print(f"\n  Waiting 2 seconds before closing...")
    time.sleep(2)

    # Close the trade
    try:
        print(f"  Closing trade #{trade_id}...")
        close_result = client.close_trade(trade_id)

        if 'orderFillTransaction' in close_result:
            fill = close_result['orderFillTransaction']
            close_price = float(fill.get('price', 0))
            realized_pl = float(fill.get('pl', 0))
            print(f"  Closed at: {close_price:.5f}")
            print(f"  Realized P&L: {realized_pl:+.4f}")
            print(f"\n  Status: PASS")
            return True
        else:
            print(f"  Status: FAIL")
            print(f"  Unexpected close response: {close_result}")
            return False

    except Exception as e:
        print(f"  Status: FAIL")
        print(f"  Error closing trade: {e}")
        print(f"  WARNING: Trade #{trade_id} may still be open!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test OANDA API connection")
    parser.add_argument("--test-trade", action="store_true",
                        help="Place a small test trade (requires confirmation)")
    parser.add_argument("--instrument", default="GBP_USD",
                        help="Instrument for price/candle tests")

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="WARNING")

    print("\n" + "#" * 60)
    print("# OANDA API CONNECTION TEST")
    print("#" * 60)
    print(f"\nAPI URL: {settings.OANDA_API_URL}")
    print(f"Account Type: {settings.OANDA_ACCOUNT_TYPE}")
    print(f"Account ID: {settings.OANDA_ACCOUNT_ID or 'Not set (will auto-detect)'}")

    # Initialize client
    try:
        client = OandaClient()

        # Auto-set account ID if not configured
        if not client.account_id:
            accounts = client.get_accounts()
            if accounts:
                client.account_id = accounts[0]['id']
                print(f"Auto-detected Account ID: {client.account_id}")
            else:
                print("\nERROR: No accounts found. Check your API key.")
                sys.exit(1)

    except Exception as e:
        print(f"\nERROR: Failed to initialize client: {e}")
        sys.exit(1)

    # Run tests
    results = []

    results.append(("API Connection", test_connection(client)))
    results.append(("Account Info", test_account_info(client)))
    results.append(("Live Price", test_live_price(client, args.instrument)))
    results.append(("Historical Candles", test_historical_candles(client, args.instrument)))
    results.append(("Open Positions", test_open_positions(client)))

    # Optional test trade
    if args.test_trade:
        print("\n" + "!" * 60)
        print("! WARNING: This will place a real trade on your account!")
        print("!" * 60)
        confirm = input("\nType 'YES' to confirm: ")
        if confirm == 'YES':
            results.append(("Test Trade", test_place_trade(client, args.instrument)))
        else:
            print("Test trade skipped.")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests passed! Ready for live trading.")
        sys.exit(0)
    else:
        print("\n  Some tests failed. Review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
