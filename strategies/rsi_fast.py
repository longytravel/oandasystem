"""
RSI Hidden Divergence Strategy - Fast Optimization Version.

This implements the FastStrategy interface for ultra-fast optimization.
Pre-computes ALL divergences once, then filters per trial.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.fast_strategy import FastStrategy, FastSignal


class RSIDivergenceFast(FastStrategy):
    """
    RSI Hidden Divergence - optimized for fast backtesting.

    Hidden Bullish: Price HL + RSI LL -> BUY
    Hidden Bearish: Price LH + RSI HH -> SELL
    """

    name = "RSI_Divergence_Fast"
    version = "2.0"

    # Pre-compute for all these RSI periods and swing strengths
    RSI_PERIODS = [5, 7, 9, 11, 14, 18, 21, 25]
    SWING_STRENGTHS = [3, 5, 7, 9]

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Return parameter grid for optimization."""
        return {
            'min_rsi_diff': [3.0, 5.0, 8.0, 12.0],
            'min_bars_between': [3, 5, 8, 12],
            'max_bars_between': [40, 60, 80, 100],
            'stop_loss_pips': [25, 35, 50],
            'tp_multiplier': [1.0, 1.5, 2.0, 2.5],
            'min_price_slope': [5.0, 15.0, 25.0],
            'max_price_slope': [50.0, 65.0, 80.0],
            'trade_start_hour': [0, 2, 4, 6],
            'trade_end_hour': [20, 22, 23],
        }

    def _calc_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Fast RSI calculation using numpy."""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)

        if len(close) <= period:
            return np.full_like(close, 50.0)

        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])

        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period

        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0)
        return 100.0 - (100.0 / (1.0 + rs))

    def _find_swings(self, highs: np.ndarray, lows: np.ndarray, strength: int) -> tuple:
        """Find all swing high and low points."""
        swing_highs = []
        swing_lows = []
        n = len(highs)

        for i in range(strength, n - strength):
            # Swing high
            is_high = True
            for j in range(1, strength + 1):
                if highs[i - j] >= highs[i] or highs[i + j] >= highs[i]:
                    is_high = False
                    break
            if is_high:
                swing_highs.append(i)

            # Swing low
            is_low = True
            for j in range(1, strength + 1):
                if lows[i - j] <= lows[i] or lows[i + j] <= lows[i]:
                    is_low = False
                    break
            if is_low:
                swing_lows.append(i)

        return swing_highs, swing_lows

    def precompute(self, df: pd.DataFrame) -> List[FastSignal]:
        """
        Pre-compute ALL possible divergences for all RSI periods and swing strengths.
        """
        signals = []
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        hours = df.index.hour.values
        days = df.index.dayofweek.values

        for rsi_period in self.RSI_PERIODS:
            rsi = self._calc_rsi(closes, rsi_period)

            for strength in self.SWING_STRENGTHS:
                swing_highs, swing_lows = self._find_swings(highs, lows, strength)

                # Bullish divergences (from swing lows)
                for i in range(1, len(swing_lows)):
                    prev_bar = swing_lows[i - 1]
                    curr_bar = swing_lows[i]
                    bars_between = curr_bar - prev_bar

                    if bars_between < 2 or bars_between > 120:
                        continue

                    prev_price = lows[prev_bar]
                    curr_price = lows[curr_bar]
                    prev_rsi = rsi[prev_bar]
                    curr_rsi = rsi[curr_bar]

                    # Hidden bullish: Price HL, RSI LL
                    if curr_price > prev_price and curr_rsi < prev_rsi:
                        rsi_diff = abs(prev_rsi - curr_rsi)
                        slope = (curr_price - prev_price) / bars_between * 10000
                        slope_angle = abs(np.arctan(slope) * 180 / np.pi)

                        signal_bar = curr_bar + 1
                        if signal_bar < len(closes):
                            signals.append(FastSignal(
                                bar=signal_bar,
                                direction=1,
                                price=closes[signal_bar],
                                hour=hours[signal_bar],
                                day=days[signal_bar],
                                attributes={
                                    'rsi_diff': rsi_diff,
                                    'bars_between': bars_between,
                                    'slope': slope_angle,
                                }
                            ))

                # Bearish divergences (from swing highs)
                for i in range(1, len(swing_highs)):
                    prev_bar = swing_highs[i - 1]
                    curr_bar = swing_highs[i]
                    bars_between = curr_bar - prev_bar

                    if bars_between < 2 or bars_between > 120:
                        continue

                    prev_price = highs[prev_bar]
                    curr_price = highs[curr_bar]
                    prev_rsi = rsi[prev_bar]
                    curr_rsi = rsi[curr_bar]

                    # Hidden bearish: Price LH, RSI HH
                    if curr_price < prev_price and curr_rsi > prev_rsi:
                        rsi_diff = abs(curr_rsi - prev_rsi)
                        slope = abs(prev_price - curr_price) / bars_between * 10000
                        slope_angle = abs(np.arctan(slope) * 180 / np.pi)

                        signal_bar = curr_bar + 1
                        if signal_bar < len(closes):
                            signals.append(FastSignal(
                                bar=signal_bar,
                                direction=-1,
                                price=closes[signal_bar],
                                hour=hours[signal_bar],
                                day=days[signal_bar],
                                attributes={
                                    'rsi_diff': rsi_diff,
                                    'bars_between': bars_between,
                                    'slope': slope_angle,
                                }
                            ))

        # Remove duplicates (same bar, same direction)
        seen = set()
        unique = []
        for s in signals:
            key = (s.bar, s.direction)
            if key not in seen:
                seen.add(key)
                unique.append(s)

        # Sort by bar
        unique.sort(key=lambda x: x.bar)
        return unique

    def filter_signals(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any]
    ) -> List[FastSignal]:
        """
        Filter pre-computed signals based on parameters.
        This runs PER TRIAL so must be fast.
        """
        min_rsi_diff = params['min_rsi_diff']
        min_bars = params['min_bars_between']
        max_bars = params['max_bars_between']
        min_slope = params['min_price_slope']
        max_slope = params['max_price_slope']
        start_hour = params['trade_start_hour']
        end_hour = params['trade_end_hour']

        result = []
        for s in signals:
            attr = s.attributes

            # RSI filter
            if attr['rsi_diff'] < min_rsi_diff:
                continue

            # Bars between filter
            if attr['bars_between'] < min_bars or attr['bars_between'] > max_bars:
                continue

            # Slope filter
            if attr['slope'] < min_slope or attr['slope'] > max_slope:
                continue

            # Day filter (no Sunday, Friday close)
            if s.day == 6:  # Sunday
                continue
            if s.day == 4 and s.hour >= 20:  # Friday close
                continue

            # Hour filter
            if start_hour < end_hour:
                if s.hour < start_hour or s.hour >= end_hour:
                    continue
            else:
                if s.hour < start_hour and s.hour >= end_hour:
                    continue

            result.append(s)

        return result

    def compute_sl_tp(
        self,
        signal: FastSignal,
        params: Dict[str, Any],
        pip_size: float
    ) -> tuple:
        """Compute stop loss and take profit prices."""
        sl_pips = params['stop_loss_pips']
        tp_pips = sl_pips * params['tp_multiplier']

        if signal.direction == 1:  # Buy
            sl = signal.price - sl_pips * pip_size
            tp = signal.price + tp_pips * pip_size
        else:  # Sell
            sl = signal.price + sl_pips * pip_size
            tp = signal.price - tp_pips * pip_size

        return sl, tp


# Registry of available fast strategies
FAST_STRATEGIES = {
    'rsi_divergence': RSIDivergenceFast,
}

# Import full strategy (adds itself to registry)
try:
    from strategies.rsi_full import RSIDivergenceFullFast
    FAST_STRATEGIES['rsi_full'] = RSIDivergenceFullFast
except ImportError:
    pass

# Import V2 strategy (adds itself to registry)
try:
    from strategies.rsi_divergence_v2 import RSIDivergenceV2
    FAST_STRATEGIES['rsi_v2'] = RSIDivergenceV2
except ImportError:
    pass

# Import V3 strategy (adds itself to registry)
try:
    from strategies.rsi_full_v3 import RSIDivergenceFullFastV3
    FAST_STRATEGIES['rsi_v3'] = RSIDivergenceFullFastV3
except ImportError:
    pass


def get_strategy(name: str) -> FastStrategy:
    """Get a fast strategy by name."""
    if name not in FAST_STRATEGIES:
        available = ', '.join(FAST_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return FAST_STRATEGIES[name]()


def list_strategies() -> List[str]:
    """List all available fast strategies."""
    return list(FAST_STRATEGIES.keys())
