"""
RSI Hidden Divergence Strategy.

Concept (similar to MQL5 RSI_Divergence_Pro_V3):
- Hidden Bullish: Price makes higher low, RSI makes lower low -> BUY
- Hidden Bearish: Price makes lower high, RSI makes higher high -> SELL

This indicates trend continuation.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger

from strategies.base import Strategy, Signal, SignalType


@dataclass
class SwingPoint:
    """Represents a swing high or low."""
    bar_index: int
    price: float
    rsi: float
    time: pd.Timestamp


class RSIDivergenceStrategy(Strategy):
    """
    Hidden RSI Divergence Strategy.

    Detects hidden divergences between price and RSI:
    - Hidden Bullish: Price HL + RSI LL -> continuation up
    - Hidden Bearish: Price LH + RSI HH -> continuation down
    """

    name = "RSI_Divergence"
    version = "1.0"

    def get_default_parameters(self) -> Dict[str, Any]:
        """Default strategy parameters."""
        return {
            # RSI settings
            "rsi_period": 14,

            # Swing detection
            "swing_strength": 5,       # Bars each side for swing
            "min_bars_between": 5,     # Min bars between swings
            "max_bars_between": 50,    # Max bars between swings

            # Stop loss / Take profit (in pips)
            "stop_loss_pips": 50.0,
            "tp_multiplier": 2.0,      # TP = SL * multiplier

            # Filters
            "use_slope_filter": True,
            "min_price_slope": 10.0,   # Minimum slope angle (degrees)
            "max_price_slope": 80.0,   # Maximum slope angle

            # Risk
            "risk_percent": 1.0,

            # Trading hours (broker time)
            "use_trading_hours": True,
            "trade_start_hour": 2,
            "trade_end_hour": 22,
            "avoid_friday_close": True,
            "friday_close_hour": 20,
        }

    def get_parameter_space(self) -> Dict[str, Any]:
        """Parameter ranges for Optuna optimization."""
        return {
            "rsi_period": (5, 30),
            "swing_strength": (2, 10),
            "min_bars_between": (3, 15),
            "max_bars_between": (20, 100),
            "stop_loss_pips": (20.0, 80.0),
            "tp_multiplier": (1.0, 4.0),
            "min_price_slope": (5.0, 30.0),
            "max_price_slope": (50.0, 85.0),
        }

    def calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def find_swing_highs(
        self,
        highs: pd.Series,
        rsi: pd.Series,
        strength: int
    ) -> List[SwingPoint]:
        """Find swing high points."""
        swings = []

        for i in range(strength, len(highs) - strength):
            is_swing = True

            # Check bars to the left
            for j in range(1, strength + 1):
                if highs.iloc[i - j] >= highs.iloc[i]:
                    is_swing = False
                    break

            # Check bars to the right
            if is_swing:
                for j in range(1, strength + 1):
                    if highs.iloc[i + j] >= highs.iloc[i]:
                        is_swing = False
                        break

            if is_swing:
                swings.append(SwingPoint(
                    bar_index=i,
                    price=highs.iloc[i],
                    rsi=rsi.iloc[i],
                    time=highs.index[i]
                ))

        return swings

    def find_swing_lows(
        self,
        lows: pd.Series,
        rsi: pd.Series,
        strength: int
    ) -> List[SwingPoint]:
        """Find swing low points."""
        swings = []

        for i in range(strength, len(lows) - strength):
            is_swing = True

            # Check bars to the left
            for j in range(1, strength + 1):
                if lows.iloc[i - j] <= lows.iloc[i]:
                    is_swing = False
                    break

            # Check bars to the right
            if is_swing:
                for j in range(1, strength + 1):
                    if lows.iloc[i + j] <= lows.iloc[i]:
                        is_swing = False
                        break

            if is_swing:
                swings.append(SwingPoint(
                    bar_index=i,
                    price=lows.iloc[i],
                    rsi=rsi.iloc[i],
                    time=lows.index[i]
                ))

        return swings

    def calculate_slope_angle(
        self,
        value1: float,
        value2: float,
        bars: int,
        scale: float = 10000.0
    ) -> float:
        """Calculate slope angle in degrees."""
        if bars <= 0:
            return 0.0

        slope = (value2 - value1) / bars * scale
        return abs(np.arctan(slope) * 180.0 / np.pi)

    def is_within_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within trading hours."""
        if not self.params["use_trading_hours"]:
            return True

        hour = timestamp.hour
        day_of_week = timestamp.dayofweek  # Monday=0, Sunday=6

        # No trading on Sunday
        if day_of_week == 6:
            return False

        # Friday close
        if self.params["avoid_friday_close"]:
            if day_of_week == 4 and hour >= self.params["friday_close_hour"]:
                return False

        # Normal hours
        start = self.params["trade_start_hour"]
        end = self.params["trade_end_hour"]

        if start < end:
            return start <= hour < end
        else:
            return hour >= start or hour < end

    def check_hidden_bullish(
        self,
        swings: List[SwingPoint],
        bar_index: int
    ) -> Optional[Tuple[SwingPoint, SwingPoint]]:
        """
        Check for hidden bullish divergence.

        Hidden Bullish: Price makes HIGHER low, RSI makes LOWER low
        """
        # Need at least 2 swing lows
        recent_swings = [s for s in swings if s.bar_index <= bar_index]
        if len(recent_swings) < 2:
            return None

        # Get the two most recent swings
        recent = recent_swings[-1]
        previous = recent_swings[-2]

        # Check bar distance
        bars_between = recent.bar_index - previous.bar_index
        if bars_between < self.params["min_bars_between"]:
            return None
        if bars_between > self.params["max_bars_between"]:
            return None

        # Hidden bullish: Price HL, RSI LL
        if recent.price > previous.price and recent.rsi < previous.rsi:
            # Optional: slope filter
            if self.params["use_slope_filter"]:
                price_angle = self.calculate_slope_angle(
                    previous.price, recent.price, bars_between
                )
                if price_angle < self.params["min_price_slope"]:
                    return None
                if price_angle > self.params["max_price_slope"]:
                    return None

            return (previous, recent)

        return None

    def check_hidden_bearish(
        self,
        swings: List[SwingPoint],
        bar_index: int
    ) -> Optional[Tuple[SwingPoint, SwingPoint]]:
        """
        Check for hidden bearish divergence.

        Hidden Bearish: Price makes LOWER high, RSI makes HIGHER high
        """
        # Need at least 2 swing highs
        recent_swings = [s for s in swings if s.bar_index <= bar_index]
        if len(recent_swings) < 2:
            return None

        # Get the two most recent swings
        recent = recent_swings[-1]
        previous = recent_swings[-2]

        # Check bar distance
        bars_between = recent.bar_index - previous.bar_index
        if bars_between < self.params["min_bars_between"]:
            return None
        if bars_between > self.params["max_bars_between"]:
            return None

        # Hidden bearish: Price LH, RSI HH
        if recent.price < previous.price and recent.rsi > previous.rsi:
            # Optional: slope filter
            if self.params["use_slope_filter"]:
                price_angle = self.calculate_slope_angle(
                    previous.price, recent.price, bars_between
                )
                if price_angle < self.params["min_price_slope"]:
                    return None
                if price_angle > self.params["max_price_slope"]:
                    return None

            return (previous, recent)

        return None

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals from historical data.

        Args:
            df: DataFrame with OHLCV columns, datetime index

        Returns:
            List of Signal objects
        """
        signals = []

        # Calculate RSI
        rsi = self.calculate_rsi(df['close'], self.params["rsi_period"])

        # Find all swing points
        swing_highs = self.find_swing_highs(
            df['high'], rsi, self.params["swing_strength"]
        )
        swing_lows = self.find_swing_lows(
            df['low'], rsi, self.params["swing_strength"]
        )

        logger.debug(f"Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")

        # Track which swings we've already signaled on
        signaled_bullish = set()
        signaled_bearish = set()

        # Iterate through bars (starting after warmup period)
        warmup = self.params["rsi_period"] + self.params["swing_strength"] * 2
        pip_size = 0.0001 if 'JPY' not in df.index.name else 0.01

        for i in range(warmup, len(df)):
            bar_time = df.index[i]
            close_price = df['close'].iloc[i]

            # Check trading hours
            if not self.is_within_trading_hours(bar_time):
                continue

            # Check for hidden bullish divergence (BUY)
            bullish = self.check_hidden_bullish(swing_lows, i)
            if bullish:
                swing_key = (bullish[0].bar_index, bullish[1].bar_index)
                if swing_key not in signaled_bullish:
                    signaled_bullish.add(swing_key)

                    sl_pips = self.params["stop_loss_pips"]
                    tp_pips = sl_pips * self.params["tp_multiplier"]

                    signals.append(Signal(
                        type=SignalType.BUY,
                        price=close_price,
                        timestamp=bar_time,
                        stop_loss=close_price - sl_pips * pip_size,
                        take_profit=close_price + tp_pips * pip_size,
                        metadata={
                            "swing1_bar": bullish[0].bar_index,
                            "swing2_bar": bullish[1].bar_index,
                            "swing1_price": bullish[0].price,
                            "swing2_price": bullish[1].price,
                            "swing1_rsi": bullish[0].rsi,
                            "swing2_rsi": bullish[1].rsi,
                        }
                    ))

            # Check for hidden bearish divergence (SELL)
            bearish = self.check_hidden_bearish(swing_highs, i)
            if bearish:
                swing_key = (bearish[0].bar_index, bearish[1].bar_index)
                if swing_key not in signaled_bearish:
                    signaled_bearish.add(swing_key)

                    sl_pips = self.params["stop_loss_pips"]
                    tp_pips = sl_pips * self.params["tp_multiplier"]

                    signals.append(Signal(
                        type=SignalType.SELL,
                        price=close_price,
                        timestamp=bar_time,
                        stop_loss=close_price + sl_pips * pip_size,
                        take_profit=close_price - tp_pips * pip_size,
                        metadata={
                            "swing1_bar": bearish[0].bar_index,
                            "swing2_bar": bearish[1].bar_index,
                            "swing1_price": bearish[0].price,
                            "swing2_price": bearish[1].price,
                            "swing1_rsi": bearish[0].rsi,
                            "swing2_rsi": bearish[1].rsi,
                        }
                    ))

        logger.info(f"Generated {len(signals)} signals "
                   f"({sum(1 for s in signals if s.type == SignalType.BUY)} buys, "
                   f"{sum(1 for s in signals if s.type == SignalType.SELL)} sells)")

        return signals


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

    from data.download import load_data

    logger.add(sys.stdout, level="DEBUG")

    # Test with sample data
    strategy = RSIDivergenceStrategy()
    print(f"Strategy: {strategy}")
    print(f"Parameters: {strategy.params}")
    print(f"Parameter space: {strategy.get_parameter_space()}")
