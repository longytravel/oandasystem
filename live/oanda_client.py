"""
OANDA API client wrapper.

Handles all communication with OANDA REST API v20.
"""
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pandas as pd
from loguru import logger

from config.settings import settings


class OandaClient:
    """
    Client for OANDA REST API v20.

    Usage:
        client = OandaClient()

        # Get account info
        account = client.get_account()

        # Get historical data
        df = client.get_candles("GBP_USD", "H1", count=500)

        # Place order
        client.market_order("GBP_USD", units=1000)  # Buy
        client.market_order("GBP_USD", units=-1000) # Sell
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        practice: bool = True
    ):
        """
        Initialize OANDA client.

        Args:
            api_key: OANDA API key (defaults to settings)
            account_id: OANDA account ID (defaults to settings)
            practice: Use practice/demo account (default True)
        """
        self.api_key = api_key or settings.OANDA_API_KEY
        self.account_id = account_id or settings.OANDA_ACCOUNT_ID

        if practice or settings.OANDA_ACCOUNT_TYPE == "practice":
            self.api_url = "https://api-fxpractice.oanda.com"
            self.stream_url = "https://stream-fxpractice.oanda.com"
        else:
            self.api_url = "https://api-fxtrade.oanda.com"
            self.stream_url = "https://stream-fxtrade.oanda.com"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"OANDA client initialized - {self.api_url}")

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict:
        """Make API request."""
        url = f"{self.api_url}{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data,
                timeout=(10, 30)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            logger.error(f"Response: {response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    # ==================== Account ====================

    def get_accounts(self) -> List[Dict]:
        """Get list of accounts for this API key."""
        response = self._request("GET", "/v3/accounts")
        return response.get("accounts", [])

    def get_account(self, account_id: Optional[str] = None) -> Dict:
        """Get account details."""
        acc_id = account_id or self.account_id
        if not acc_id:
            raise ValueError("Account ID not set. Check .env file.")
        response = self._request("GET", f"/v3/accounts/{acc_id}")
        return response.get("account", {})

    def get_account_summary(self, account_id: Optional[str] = None) -> Dict:
        """Get account summary (balance, equity, etc.)."""
        acc_id = account_id or self.account_id
        if not acc_id:
            raise ValueError("Account ID not set. Check .env file.")
        response = self._request("GET", f"/v3/accounts/{acc_id}/summary")
        return response.get("account", {})

    # ==================== Instruments ====================

    def get_instruments(self, account_id: Optional[str] = None) -> List[Dict]:
        """Get tradeable instruments."""
        acc_id = account_id or self.account_id
        response = self._request("GET", f"/v3/accounts/{acc_id}/instruments")
        return response.get("instruments", [])

    def get_price(self, instrument: str) -> Dict:
        """Get current price for an instrument."""
        response = self._request(
            "GET",
            f"/v3/accounts/{self.account_id}/pricing",
            params={"instruments": instrument}
        )
        prices = response.get("prices", [])
        return prices[0] if prices else {}

    # ==================== Candles / Historical Data ====================

    def get_candles(
        self,
        instrument: str,
        granularity: str = "H1",
        count: Optional[int] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        price: str = "M"  # M=mid, B=bid, A=ask, BA=bid+ask
    ) -> pd.DataFrame:
        """
        Get historical candlestick data.

        Args:
            instrument: e.g., "GBP_USD"
            granularity: S5, S10, S15, S30, M1, M2, M4, M5, M10, M15, M30,
                        H1, H2, H3, H4, H6, H8, H12, D, W, M
            count: Number of candles (max 5000)
            from_time: Start time
            to_time: End time
            price: M=mid, B=bid, A=ask

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        params = {
            "granularity": granularity,
            "price": price
        }

        # OANDA: Can't use count when from/to are specified
        if from_time and to_time:
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif count:
            params["count"] = min(count, 5000)
            if from_time:
                params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            if to_time:
                params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        response = self._request(
            "GET",
            f"/v3/instruments/{instrument}/candles",
            params=params
        )

        candles = response.get("candles", [])

        if not candles:
            return pd.DataFrame()

        # Parse candles into DataFrame
        data = []
        for candle in candles:
            if not candle.get("complete", True):
                continue  # Skip incomplete candles

            mid = candle.get("mid", {})
            data.append({
                "time": pd.to_datetime(candle["time"]),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(candle.get("volume", 0))
            })

        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index("time", inplace=True)

        return df

    def get_candles_range(
        self,
        instrument: str,
        granularity: str,
        from_time: datetime,
        to_time: datetime
    ) -> pd.DataFrame:
        """
        Get candles for a date range (handles pagination).

        OANDA limits to 5000 candles per request, this method
        fetches in chunks if needed.
        """
        all_candles = []
        current_from = from_time

        while current_from < to_time:
            # Note: OANDA doesn't allow count when from/to are specified
            df = self.get_candles(
                instrument=instrument,
                granularity=granularity,
                from_time=current_from,
                to_time=to_time,
                count=None  # Don't use count with from/to
            )

            if df.empty:
                break

            all_candles.append(df)

            # Move to next chunk (handle timezone-aware timestamps)
            # FIX: Advance by the correct granularity duration, not just 1 minute
            tf_minutes = {
                "S5": 5/60, "S10": 10/60, "S15": 15/60, "S30": 30/60,
                "M1": 1, "M2": 2, "M4": 4, "M5": 5, "M10": 10, "M15": 15, "M30": 30,
                "H1": 60, "H2": 120, "H3": 180, "H4": 240, "H6": 360, "H8": 480, "H12": 720,
                "D": 1440, "W": 10080, "M": 43200
            }
            candle_minutes = tf_minutes.get(granularity, 60)
            last_time = df.index[-1].to_pydatetime()
            if last_time.tzinfo is not None:
                last_time = last_time.replace(tzinfo=None)
            current_from = last_time + timedelta(minutes=candle_minutes)

            logger.debug(f"Fetched {len(df)} candles, up to {df.index[-1]}")

        if not all_candles:
            return pd.DataFrame()

        result = pd.concat(all_candles)
        result = result[~result.index.duplicated(keep='last')]
        return result.sort_index()

    # ==================== Orders ====================

    def market_order(
        self,
        instrument: str,
        units: int,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        strategy_tag: Optional[str] = None
    ) -> Dict:
        """
        Place a market order.

        Args:
            instrument: e.g., "GBP_USD"
            units: Positive for buy, negative for sell
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            strategy_tag: Strategy identifier for clientExtensions (e.g., "rsi_v3_GBP_USD_M15")

        Returns:
            Order response
        """
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",  # Fill or kill
                "positionFill": "DEFAULT"
            }
        }

        if stop_loss_price is not None:
            order_data["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss_price:.5f}"
            }

        if take_profit_price is not None:
            order_data["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit_price:.5f}"
            }

        if strategy_tag:
            # OANDA clientExtensions: tag (max 128 chars), comment (max 128 chars)
            order_data["order"]["clientExtensions"] = {
                "tag": strategy_tag[:128],
                "comment": strategy_tag[:128],
            }
            # Also tag the trade itself so closed trades retain the strategy ID
            order_data["order"]["tradeClientExtensions"] = {
                "tag": strategy_tag[:128],
                "comment": strategy_tag[:128],
            }

        return self._request(
            "POST",
            f"/v3/accounts/{self.account_id}/orders",
            data=order_data
        )

    # ==================== Positions ====================

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        response = self._request(
            "GET",
            f"/v3/accounts/{self.account_id}/openPositions"
        )
        return response.get("positions", [])

    def get_position(self, instrument: str) -> Dict:
        """Get position for a specific instrument."""
        response = self._request(
            "GET",
            f"/v3/accounts/{self.account_id}/positions/{instrument}"
        )
        return response.get("position", {})

    def close_position(self, instrument: str, long_units: Optional[str] = None, short_units: Optional[str] = None) -> Dict:
        """Close a position."""
        data = {}
        if long_units is not None:
            data["longUnits"] = long_units
        if short_units is not None:
            data["shortUnits"] = short_units

        return self._request(
            "PUT",
            f"/v3/accounts/{self.account_id}/positions/{instrument}/close",
            data=data
        )

    # ==================== Trades ====================

    def get_open_trades(self) -> List[Dict]:
        """Get all open trades."""
        response = self._request(
            "GET",
            f"/v3/accounts/{self.account_id}/openTrades"
        )
        return response.get("trades", [])

    def get_trade(self, trade_id: str) -> Dict:
        """
        Get details for a single trade (open or closed).

        Returns trade with realizedPL, averageClosePrice, closeTime, state, etc.
        """
        response = self._request(
            "GET",
            f"/v3/accounts/{self.account_id}/trades/{trade_id}"
        )
        return response.get("trade", {})

    def close_trade(self, trade_id: str, units: Optional[str] = None) -> Dict:
        """Close a specific trade."""
        data = {"units": units or "ALL"}
        return self._request(
            "PUT",
            f"/v3/accounts/{self.account_id}/trades/{trade_id}/close",
            data=data
        )

    def modify_trade(
        self,
        trade_id: str,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> Dict:
        """Modify SL/TP on an open trade."""
        data = {}

        if stop_loss_price is not None:
            data["stopLoss"] = {"price": f"{stop_loss_price:.5f}"}

        if take_profit_price is not None:
            data["takeProfit"] = {"price": f"{take_profit_price:.5f}"}

        return self._request(
            "PUT",
            f"/v3/accounts/{self.account_id}/trades/{trade_id}/orders",
            data=data
        )


# Quick test
if __name__ == "__main__":
    from loguru import logger
    import sys

    logger.remove()
    logger.add(sys.stdout, level="DEBUG")

    client = OandaClient()

    # Test 1: Get accounts (to find account ID)
    print("\n" + "=" * 50)
    print("TESTING OANDA API CONNECTION")
    print("=" * 50)

    try:
        accounts = client.get_accounts()
        print(f"\nAccounts found: {len(accounts)}")
        for acc in accounts:
            print(f"  - {acc['id']} ({acc.get('mt4AccountID', 'N/A')})")

        if accounts and not settings.OANDA_ACCOUNT_ID:
            print(f"\n*** Add this to your .env file: ***")
            print(f"OANDA_ACCOUNT_ID={accounts[0]['id']}")

        # Test 2: Get some candles
        if accounts:
            client.account_id = accounts[0]['id']

            print(f"\nFetching GBP_USD H1 candles...")
            df = client.get_candles("GBP_USD", "H1", count=10)
            print(df)

            print("\nAPI connection successful!")

    except Exception as e:
        print(f"\nAPI Error: {e}")
        print("\nCheck your API key in .env file")
