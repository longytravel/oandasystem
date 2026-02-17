"""
Donchian Channel Breakout Strategy (Turtle Trading)

Classic trend-following system made famous by Richard Dennis's Turtle Traders.

Entry: Price breaks above N-period high (buy) or below N-period low (sell).
Exit: SL at ATR multiple or opposite channel edge; TP at RR multiple.

The Donchian Channel is simply the highest high and lowest low over N periods.
When price breaks out of this range, it signals a potential new trend.

Parameter groups (3 groups, 10 total params):
  1. signal (3 params): channel_period, min_channel_width_atr, use_exit_channel
  2. filters (4 params): use_trend_filter, use_time_filter, trade_start_hour, trade_end_hour
  3. risk (3 params): sl_atr_mult, tp_rr_ratio, max_hold_bars

Created: 2026-02-17
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.fast_strategy import FastStrategy, FastSignal, ParameterGroup, ParameterDef


class DonchianBreakoutStrategy(FastStrategy):
    """
    Donchian Channel Breakout (Turtle Trading).

    Entry: Close breaks above N-period high (buy) or below N-period low (sell).
    Trend filter: 200 SMA direction agreement.
    SL: ATR-based. TP: Risk-reward ratio.
    """

    name = "Donchian_Breakout"
    version = "1.0"

    ATR_PERIOD = 14

    # Pre-compute channels for these periods
    CHANNEL_PERIODS = [10, 20, 30, 50, 70]

    def get_parameter_groups(self) -> Optional[Dict[str, ParameterGroup]]:
        groups = {}

        # === Group 1: Signal (3 params) ===
        signal = ParameterGroup('signal', 'Donchian channel signal parameters')
        signal.add_param('channel_period', [10, 20, 30, 50, 70], default=20)
        signal.add_param('min_channel_width_atr', [0.5, 1.0, 1.5, 2.0], default=1.0)
        signal.add_param('confirm_bars', [1, 2, 3], default=1)
        groups['signal'] = signal

        # === Group 2: Filters (4 params) ===
        filters = ParameterGroup('filters', 'Entry filter parameters')
        filters.add_param('use_trend_filter', [True, False], default=False)
        filters.add_param('use_time_filter', [True, False], default=False)
        filters.add_param('trade_start_hour', [0, 4, 8], default=0)
        filters.add_param('trade_end_hour', [18, 22, 23], default=23)
        groups['filters'] = filters

        # === Group 3: Risk (3 params) ===
        risk = ParameterGroup('risk', 'Stop loss and take profit')
        risk.add_param('sl_atr_mult', [1.0, 1.5, 2.0, 3.0], default=2.0)
        risk.add_param('tp_rr_ratio', [1.5, 2.0, 3.0, 5.0, 7.5], default=5.0)
        risk.add_param('max_hold_bars', [0, 50, 100, 200], default=0)
        groups['risk'] = risk

        return groups

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        groups = self.get_parameter_groups()
        space = {}
        for group in groups.values():
            space.update(group.get_param_space())
        return space

    def _calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
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
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    def _calc_ma(self, close: np.ndarray, period: int) -> np.ndarray:
        ma = np.zeros_like(close)
        for i in range(period - 1, len(close)):
            ma[i] = np.mean(close[i - period + 1:i + 1])
        return ma

    def precompute(self, df: pd.DataFrame) -> List[FastSignal]:
        """Pre-compute breakout signals for all channel periods."""
        signals = []
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        hours = df.index.hour.values
        days = df.index.dayofweek.values
        n = len(closes)

        atr = self._calc_atr(highs, lows, closes, self.ATR_PERIOD)
        ma200 = self._calc_ma(closes, 200)

        for period in self.CHANNEL_PERIODS:
            if n <= period + 1:
                continue

            # Compute rolling highest high and lowest low
            upper = np.zeros(n, dtype=np.float64)
            lower = np.zeros(n, dtype=np.float64)
            for i in range(period, n):
                upper[i] = np.max(highs[i - period:i])
                lower[i] = np.min(lows[i - period:i])

            # Detect breakouts: close breaks above upper or below lower
            for i in range(period + 1, n):
                if atr[i] <= 0 or upper[i] <= 0:
                    continue

                channel_width = upper[i] - lower[i]
                channel_width_atr = channel_width / atr[i] if atr[i] > 0 else 0

                # Bullish breakout: close above previous upper channel
                if closes[i] > upper[i] and closes[i - 1] <= upper[i]:
                    signals.append(FastSignal(
                        bar=i,
                        direction=1,
                        price=closes[i],
                        hour=hours[i],
                        day=days[i],
                        attributes={
                            'channel_period': period,
                            'channel_width_atr': channel_width_atr,
                            'atr': atr[i],
                            'atr_pips': atr[i] / self._pip_size if self._pip_size > 0 else 0,
                            'ma200': ma200[i],
                            'channel_upper': upper[i],
                            'channel_lower': lower[i],
                        }
                    ))

                # Bearish breakout: close below previous lower channel
                if closes[i] < lower[i] and closes[i - 1] >= lower[i]:
                    signals.append(FastSignal(
                        bar=i,
                        direction=-1,
                        price=closes[i],
                        hour=hours[i],
                        day=days[i],
                        attributes={
                            'channel_period': period,
                            'channel_width_atr': channel_width_atr,
                            'atr': atr[i],
                            'atr_pips': atr[i] / self._pip_size if self._pip_size > 0 else 0,
                            'ma200': ma200[i],
                            'channel_upper': upper[i],
                            'channel_lower': lower[i],
                        }
                    ))

        # Remove duplicates (same bar, same direction)
        seen = set()
        unique = []
        for s in signals:
            key = (s.bar, s.direction, s.attributes['channel_period'])
            if key not in seen:
                seen.add(key)
                unique.append(s)

        unique.sort(key=lambda x: x.bar)
        return unique

    def filter_signals(self, signals: List[FastSignal], params: Dict[str, Any]) -> List[FastSignal]:
        result = []
        channel_period = params.get('channel_period', 20)
        min_width = params.get('min_channel_width_atr', 1.0)

        for s in signals:
            attr = s.attributes
            if attr['channel_period'] != channel_period:
                continue
            if attr['channel_width_atr'] < min_width:
                continue

            # Trend filter: 200 MA direction
            if params.get('use_trend_filter', False):
                if attr['ma200'] > 0:
                    if s.direction == 1 and s.price < attr['ma200']:
                        continue
                    if s.direction == -1 and s.price > attr['ma200']:
                        continue

            # Time filter
            if params.get('use_time_filter', False):
                start = params.get('trade_start_hour', 0)
                end = params.get('trade_end_hour', 23)
                if start < end:
                    if s.hour < start or s.hour >= end:
                        continue
                else:
                    if s.hour < start and s.hour >= end:
                        continue

            result.append(s)
        return result

    def compute_sl_tp(self, signal: FastSignal, params: Dict[str, Any], pip_size: float) -> Tuple[float, float]:
        attr = signal.attributes
        atr_pips = attr['atr'] / pip_size if pip_size > 0 else 35.0
        sl_pips = max(atr_pips * params.get('sl_atr_mult', 2.0), 10.0)
        tp_pips = max(sl_pips * params.get('tp_rr_ratio', 5.0), sl_pips)

        if signal.direction == 1:
            sl = signal.price - sl_pips * pip_size
            tp = signal.price + tp_pips * pip_size
        else:
            sl = signal.price + sl_pips * pip_size
            tp = signal.price - tp_pips * pip_size
        return sl, tp

    # === VECTORIZED PATH ===

    def _build_signal_arrays(self, signals: list) -> Dict[str, np.ndarray]:
        n = len(signals)
        if n == 0:
            return {k: np.array([], dtype=d) for k, d in [
                ('bars', np.int64), ('prices', np.float64), ('directions', np.int64),
                ('hours', np.int64), ('days', np.int64), ('channel_periods', np.int64),
                ('channel_width_atr', np.float64), ('atr', np.float64),
                ('atr_pips', np.float64), ('ma200', np.float64),
            ]}

        bars = np.empty(n, dtype=np.int64)
        prices = np.empty(n, dtype=np.float64)
        directions = np.empty(n, dtype=np.int64)
        hours = np.empty(n, dtype=np.int64)
        days_arr = np.empty(n, dtype=np.int64)
        channel_periods = np.empty(n, dtype=np.int64)
        channel_width_atr = np.empty(n, dtype=np.float64)
        atr = np.empty(n, dtype=np.float64)
        atr_pips = np.empty(n, dtype=np.float64)
        ma200 = np.empty(n, dtype=np.float64)

        for i, s in enumerate(signals):
            bars[i] = s.bar
            prices[i] = s.price
            directions[i] = s.direction
            hours[i] = s.hour
            days_arr[i] = s.day
            a = s.attributes
            channel_periods[i] = a['channel_period']
            channel_width_atr[i] = a['channel_width_atr']
            atr[i] = a['atr']
            atr_pips[i] = a['atr_pips']
            ma200[i] = a['ma200']

        return {
            'bars': bars, 'prices': prices, 'directions': directions,
            'hours': hours, 'days': days_arr, 'channel_periods': channel_periods,
            'channel_width_atr': channel_width_atr, 'atr': atr,
            'atr_pips': atr_pips, 'ma200': ma200,
        }

    def _filter_vectorized(self, params: Dict[str, Any]) -> np.ndarray:
        va = self._vec_arrays
        n = len(va['bars'])
        if n == 0:
            return np.array([], dtype=np.bool_)

        mask = np.ones(n, dtype=np.bool_)
        mask &= va['channel_periods'] == params.get('channel_period', 20)
        mask &= va['channel_width_atr'] >= params.get('min_channel_width_atr', 1.0)

        if params.get('use_trend_filter', False):
            ma_valid = va['ma200'] > 0
            long_fail = (va['directions'] == 1) & ma_valid & (va['prices'] < va['ma200'])
            short_fail = (va['directions'] == -1) & ma_valid & (va['prices'] > va['ma200'])
            mask &= ~(long_fail | short_fail)

        if params.get('use_time_filter', False):
            start = params.get('trade_start_hour', 0)
            end = params.get('trade_end_hour', 23)
            if start < end:
                mask &= (va['hours'] >= start) & (va['hours'] < end)
            else:
                mask &= (va['hours'] >= start) | (va['hours'] < end)

        return mask

    def _compute_sl_tp_vectorized(self, mask: np.ndarray, params: Dict[str, Any], pip_size: float) -> Tuple[np.ndarray, np.ndarray]:
        va = self._vec_arrays
        prices = va['prices'][mask]
        directions = va['directions'][mask]
        atr_pips = va['atr_pips'][mask]
        n = len(prices)

        sl_pips = np.maximum(atr_pips * params.get('sl_atr_mult', 2.0), 10.0)
        tp_pips = np.maximum(sl_pips * params.get('tp_rr_ratio', 5.0), sl_pips)

        is_long = directions == 1
        sl_prices = np.where(is_long, prices - sl_pips * pip_size, prices + sl_pips * pip_size)
        tp_prices = np.where(is_long, prices + tp_pips * pip_size, prices - tp_pips * pip_size)
        return sl_prices, tp_prices

    def _get_management_arrays_vectorized(self, mask: np.ndarray, params: Dict[str, Any], sl_prices: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        n = int(np.sum(mask))
        if n == 0:
            return None
        max_bars_val = params.get('max_hold_bars', 0)
        return {
            'use_trailing': np.full(n, False, dtype=np.bool_),
            'trail_start_pips': np.zeros(n, dtype=np.float64),
            'trail_step_pips': np.zeros(n, dtype=np.float64),
            'use_breakeven': np.full(n, False, dtype=np.bool_),
            'be_trigger_pips': np.zeros(n, dtype=np.float64),
            'be_offset_pips': np.zeros(n, dtype=np.float64),
            'use_partial': np.full(n, False, dtype=np.bool_),
            'partial_pct': np.zeros(n, dtype=np.float64),
            'partial_target_pips': np.zeros(n, dtype=np.float64),
            'max_bars': np.full(n, max_bars_val, dtype=np.int64),
        }
