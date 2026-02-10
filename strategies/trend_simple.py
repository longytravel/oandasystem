"""
Simple Trend Following Strategy

PHILOSOPHY: Fewer parameters = more robust. Based on proven edge (trend following).

Core idea:
- Trade WITH the trend (price vs slow MA)
- Enter on pullbacks to fast MA
- Exit with ATR-based stops

Only 6 parameters (vs 35 in RSI divergence):
- slow_ma_period: Trend filter (100, 150, 200)
- fast_ma_period: Entry trigger (20, 50)
- atr_period: Volatility measure (14)
- sl_atr_mult: Stop loss distance (1.5, 2.0, 2.5)
- tp_atr_mult: Take profit distance (2.0, 3.0, 4.0)
- use_trailing: Trail stop or fixed (True/False)

WHY THIS MIGHT WORK:
- Trend following has 40+ years of CTA track record
- Based on behavioral finance (herding, anchoring, slow information diffusion)
- Simple strategies generalize better than complex ones
- Works across asset classes (commodities, FX, equities)

Created: 2026-02-04
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.fast_strategy import FastStrategy, FastSignal, ParameterGroup


class SimpleTrendStrategy(FastStrategy):
    """
    Simple Trend Following - Minimal parameters, maximum robustness.

    Entry Logic:
    1. Determine trend: Price > slow_ma = uptrend, Price < slow_ma = downtrend
    2. Wait for pullback: Price touches fast_ma
    3. Enter in trend direction

    Exit Logic:
    - Stop loss: ATR-based
    - Take profit: ATR-based (or trailing)

    Only 6 parameters vs 35 in RSI divergence.
    """

    name = "Simple_Trend"
    version = "1.0"

    ATR_PERIOD = 14  # Fixed - no need to optimize

    def get_parameter_groups(self) -> Optional[Dict[str, ParameterGroup]]:
        """Minimal parameter groups."""
        groups = {}

        # === Group 1: Trend Detection ===
        trend = ParameterGroup('trend', 'Trend detection parameters')
        trend.add_param('slow_ma_period', [100, 150, 200], default=200)
        trend.add_param('fast_ma_period', [20, 35, 50], default=50)
        groups['trend'] = trend

        # === Group 2: Risk Management ===
        risk = ParameterGroup('risk', 'Stop loss and take profit')
        risk.add_param('sl_atr_mult', [1.5, 2.0, 2.5, 3.0], default=2.0)
        risk.add_param('tp_atr_mult', [2.0, 3.0, 4.0, 5.0], default=3.0)
        risk.add_param('use_trailing', [True, False], default=True)
        risk.add_param('trail_atr_mult', [1.5, 2.0, 2.5], default=2.0)
        groups['risk'] = risk

        return groups

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Return parameter grid - only 6 parameters!"""
        groups = self.get_parameter_groups()
        space = {}
        for group in groups.values():
            space.update(group.get_param_space())
        return space

    def _calc_sma(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        sma = np.zeros_like(close)
        for i in range(period - 1, len(close)):
            sma[i] = np.mean(close[i-period+1:i+1])
        return sma

    def _calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate ATR."""
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]

        atr = np.zeros_like(tr)
        if len(tr) > period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return atr

    def precompute(self, df: pd.DataFrame) -> List[FastSignal]:
        """
        Pre-compute ALL possible pullback signals.

        Signal generated when:
        1. Clear trend exists (price away from slow MA)
        2. Price pulls back to touch fast MA
        3. Direction matches trend
        """
        signals = []
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        hours = df.index.hour.values
        days = df.index.dayofweek.values

        # Pre-compute ATR
        atr = self._calc_atr(highs, lows, closes, self.ATR_PERIOD)

        # Pre-compute all MA combinations
        ma_periods = [20, 35, 50, 100, 150, 200]
        mas = {p: self._calc_sma(closes, p) for p in ma_periods}

        # Generate signals for each MA combination
        for slow_period in [100, 150, 200]:
            for fast_period in [20, 35, 50]:
                if fast_period >= slow_period:
                    continue

                slow_ma = mas[slow_period]
                fast_ma = mas[fast_period]

                # Need enough bars for slow MA
                start_bar = slow_period + 1

                for i in range(start_bar, len(closes) - 1):
                    # Skip if MAs not valid yet
                    if slow_ma[i] == 0 or fast_ma[i] == 0:
                        continue

                    # Determine trend
                    # Uptrend: price > slow_ma AND fast_ma > slow_ma
                    # Downtrend: price < slow_ma AND fast_ma < slow_ma

                    price = closes[i]
                    trend_up = price > slow_ma[i] and fast_ma[i] > slow_ma[i]
                    trend_down = price < slow_ma[i] and fast_ma[i] < slow_ma[i]

                    if not trend_up and not trend_down:
                        continue  # No clear trend

                    # Check for pullback to fast MA
                    # Pullback = price was away from fast_ma, now touching it

                    # For uptrend: low touched or crossed below fast_ma
                    # For downtrend: high touched or crossed above fast_ma

                    prev_low = lows[i-1]
                    prev_high = highs[i-1]
                    curr_low = lows[i]
                    curr_high = highs[i]

                    if trend_up:
                        # Bullish pullback: low touches fast_ma from above
                        # Previous bar was above fast_ma, current bar touches it
                        touched_ma = curr_low <= fast_ma[i] * 1.002  # Within 0.2%
                        was_above = prev_low > fast_ma[i-1]
                        price_above_ma = price > fast_ma[i]  # Closed above

                        if touched_ma and price_above_ma:
                            # Valid bullish pullback
                            signal_bar = i + 1  # Enter next bar
                            if signal_bar < len(closes):
                                signals.append(FastSignal(
                                    bar=signal_bar,
                                    direction=1,  # Buy
                                    price=closes[signal_bar],
                                    hour=hours[signal_bar],
                                    day=days[signal_bar],
                                    attributes={
                                        'slow_ma_period': slow_period,
                                        'fast_ma_period': fast_period,
                                        'slow_ma': slow_ma[signal_bar],
                                        'fast_ma': fast_ma[signal_bar],
                                        'atr': atr[signal_bar],
                                        'atr_pips': atr[signal_bar] / self._pip_size,
                                        'trend_strength': (price - slow_ma[i]) / atr[i] if atr[i] > 0 else 0,
                                    }
                                ))

                    elif trend_down:
                        # Bearish pullback: high touches fast_ma from below
                        touched_ma = curr_high >= fast_ma[i] * 0.998  # Within 0.2%
                        was_below = prev_high < fast_ma[i-1]
                        price_below_ma = price < fast_ma[i]  # Closed below

                        if touched_ma and price_below_ma:
                            # Valid bearish pullback
                            signal_bar = i + 1  # Enter next bar
                            if signal_bar < len(closes):
                                signals.append(FastSignal(
                                    bar=signal_bar,
                                    direction=-1,  # Sell
                                    price=closes[signal_bar],
                                    hour=hours[signal_bar],
                                    day=days[signal_bar],
                                    attributes={
                                        'slow_ma_period': slow_period,
                                        'fast_ma_period': fast_period,
                                        'slow_ma': slow_ma[signal_bar],
                                        'fast_ma': fast_ma[signal_bar],
                                        'atr': atr[signal_bar],
                                        'atr_pips': atr[signal_bar] / self._pip_size,
                                        'trend_strength': (slow_ma[i] - price) / atr[i] if atr[i] > 0 else 0,
                                    }
                                ))

        # Remove duplicates (same bar, same direction, same MA combo)
        seen = set()
        unique = []
        for s in signals:
            key = (s.bar, s.direction, s.attributes['slow_ma_period'], s.attributes['fast_ma_period'])
            if key not in seen:
                seen.add(key)
                unique.append(s)

        unique.sort(key=lambda x: x.bar)
        return unique

    def filter_signals(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any]
    ) -> List[FastSignal]:
        """Filter signals by MA parameters."""
        result = []

        target_slow = params.get('slow_ma_period', 200)
        target_fast = params.get('fast_ma_period', 50)

        for s in signals:
            attr = s.attributes

            # Match MA periods
            if attr['slow_ma_period'] != target_slow:
                continue
            if attr['fast_ma_period'] != target_fast:
                continue

            # Optional: filter by trend strength (require strong trend)
            # Disabled for simplicity - let the optimizer decide
            # if abs(attr['trend_strength']) < 1.0:
            #     continue

            result.append(s)

        return result

    def compute_sl_tp(
        self,
        signal: FastSignal,
        params: Dict[str, Any],
        pip_size: float
    ) -> Tuple[float, float]:
        """Compute ATR-based stop loss and take profit."""
        attr = signal.attributes
        atr = attr['atr']

        # Stop loss
        sl_mult = params.get('sl_atr_mult', 2.0)
        sl_distance = atr * sl_mult

        # Take profit
        tp_mult = params.get('tp_atr_mult', 3.0)
        tp_distance = atr * tp_mult

        if signal.direction == 1:  # Buy
            sl = signal.price - sl_distance
            tp = signal.price + tp_distance
        else:  # Sell
            sl = signal.price + sl_distance
            tp = signal.price - tp_distance

        return sl, tp

    def get_management_arrays(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Get trade management arrays."""
        n = len(signals)
        if n == 0:
            return None

        use_trailing = params.get('use_trailing', True)
        trail_mult = params.get('trail_atr_mult', 2.0)

        # Trailing stop arrays
        use_trailing_arr = np.full(n, use_trailing, dtype=np.bool_)

        # Trail start and step based on ATR
        trail_start = np.zeros(n, dtype=np.float64)
        trail_step = np.zeros(n, dtype=np.float64)

        for i, sig in enumerate(signals):
            atr_pips = sig.attributes.get('atr_pips', 35.0)
            trail_start[i] = atr_pips * 1.5  # Start trailing after 1.5 ATR profit
            trail_step[i] = atr_pips * 0.5   # Step by 0.5 ATR

        # No breakeven or partial closes - keep it simple
        use_be = np.full(n, False, dtype=np.bool_)
        be_trigger = np.zeros(n, dtype=np.float64)
        be_offset = np.zeros(n, dtype=np.float64)
        use_partial = np.full(n, False, dtype=np.bool_)
        partial_pct = np.zeros(n, dtype=np.float64)
        partial_target = np.zeros(n, dtype=np.float64)
        max_bars = np.zeros(n, dtype=np.int64)

        return {
            'use_trailing': use_trailing_arr,
            'trail_start_pips': trail_start,
            'trail_step_pips': trail_step,
            'use_breakeven': use_be,
            'be_trigger_pips': be_trigger,
            'be_offset_pips': be_offset,
            'use_partial': use_partial,
            'partial_pct': partial_pct,
            'partial_target_pips': partial_target,
            'max_bars': max_bars,
        }
