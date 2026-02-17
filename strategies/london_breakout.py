"""
London Session Breakout Strategy

Time-based breakout strategy exploiting the volatility surge at London open.

Concept: During the Asian session (00:00-07:00 UTC), price forms a range.
When London opens (07:00-08:00 UTC), the increased volume and liquidity
often causes price to break out of this range directionally.

Entry: Price breaks above/below the Asian session range during London open.
SL: Opposite end of the range (or ATR-based minimum).
TP: Risk-reward ratio based.

This is a completely different paradigm from technical indicator strategies -
it's purely time-based and exploits known forex market microstructure.

Parameter groups (3 groups, 10 total params):
  1. signal (4 params): asian_start_hour, asian_end_hour, entry_window_hours, min_range_atr
  2. filters (3 params): use_trend_filter, max_range_atr, require_inside_bar
  3. risk (3 params): sl_mode (range/atr), sl_atr_mult, tp_rr_ratio

Created: 2026-02-17
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.fast_strategy import FastStrategy, FastSignal, ParameterGroup, ParameterDef


class LondonBreakoutStrategy(FastStrategy):
    """
    London Session Breakout.

    Identifies Asian session range and trades breakouts at London open.
    """

    name = "London_Breakout"
    version = "1.0"

    ATR_PERIOD = 14

    # Pre-compute ranges for these Asian session configs
    ASIAN_STARTS = [0, 1, 2]
    ASIAN_ENDS = [6, 7, 8]

    def get_parameter_groups(self) -> Optional[Dict[str, ParameterGroup]]:
        groups = {}

        signal = ParameterGroup('signal', 'Session range and breakout parameters')
        signal.add_param('asian_start_hour', [0, 1, 2], default=0)
        signal.add_param('asian_end_hour', [6, 7, 8], default=7)
        signal.add_param('entry_window_hours', [2, 3, 4, 6], default=3)
        signal.add_param('min_range_atr', [0.3, 0.5, 0.8, 1.0], default=0.5)
        groups['signal'] = signal

        filters = ParameterGroup('filters', 'Entry filter parameters')
        filters.add_param('use_trend_filter', [True, False], default=False)
        filters.add_param('max_range_atr', [2.0, 3.0, 5.0], default=3.0)
        filters.add_param('breakout_buffer_pct', [0, 10, 25], default=10)
        groups['filters'] = filters

        risk = ParameterGroup('risk', 'Stop loss and take profit')
        risk.add_param('sl_mode', ['range', 'atr'], default='range')
        risk.add_param('sl_atr_mult', [1.0, 1.5, 2.0, 3.0], default=1.5)
        risk.add_param('tp_rr_ratio', [1.5, 2.0, 3.0, 5.0, 7.5], default=5.0)
        groups['risk'] = risk

        return groups

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        groups = self.get_parameter_groups()
        space = {}
        for group in groups.values():
            space.update(group.get_param_space())
        return space

    def _calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        tr = np.maximum(high - low, np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ))
        tr[0] = high[0] - low[0]
        atr = np.zeros_like(tr)
        if len(tr) > period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    def _calc_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        ma = np.zeros_like(data)
        for i in range(period - 1, len(data)):
            ma[i] = np.mean(data[i - period + 1:i + 1])
        return ma

    def precompute(self, df: pd.DataFrame) -> List[FastSignal]:
        """Pre-compute session range breakout signals."""
        signals = []
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        hours = df.index.hour.values
        days = df.index.dayofweek.values
        n = len(closes)

        atr = self._calc_atr(highs, lows, closes, self.ATR_PERIOD)
        ma200 = self._calc_sma(closes, 200)

        for asian_start in self.ASIAN_STARTS:
            for asian_end in self.ASIAN_ENDS:
                if asian_start >= asian_end:
                    continue

                # Track session ranges day by day
                # A "session" is from asian_start to asian_end on the same day
                current_range_high = 0.0
                current_range_low = float('inf')
                in_range = False
                range_complete = False
                range_high = 0.0
                range_low = float('inf')
                entry_deadline = 0  # hour after which we don't enter

                for i in range(1, n):
                    hour = hours[i]
                    day = days[i]

                    # Skip weekends
                    if day >= 5:
                        in_range = False
                        range_complete = False
                        continue

                    # New day or range start
                    if hour == asian_start and hours[i - 1] != asian_start:
                        current_range_high = highs[i]
                        current_range_low = lows[i]
                        in_range = True
                        range_complete = False

                    # Building the range
                    if in_range and asian_start <= hour < asian_end:
                        current_range_high = max(current_range_high, highs[i])
                        current_range_low = min(current_range_low, lows[i])

                    # Range complete at asian_end
                    if in_range and hour >= asian_end and not range_complete:
                        range_high = current_range_high
                        range_low = current_range_low
                        range_complete = True
                        in_range = False

                    # Check for breakout during entry window
                    if range_complete and asian_end <= hour < asian_end + 6:
                        if atr[i] <= 0 or range_high <= range_low:
                            continue

                        range_size = range_high - range_low
                        range_atr = range_size / atr[i]

                        # Bullish breakout
                        if closes[i] > range_high and closes[i - 1] <= range_high:
                            signals.append(FastSignal(
                                bar=i,
                                direction=1,
                                price=closes[i],
                                hour=hours[i],
                                day=days[i],
                                attributes={
                                    'asian_start': asian_start,
                                    'asian_end': asian_end,
                                    'range_high': range_high,
                                    'range_low': range_low,
                                    'range_size': range_size,
                                    'range_atr': range_atr,
                                    'atr': atr[i],
                                    'atr_pips': atr[i] / self._pip_size if self._pip_size > 0 else 0,
                                    'ma200': ma200[i],
                                    'hours_after_range': hour - asian_end,
                                }
                            ))

                        # Bearish breakout
                        if closes[i] < range_low and closes[i - 1] >= range_low:
                            signals.append(FastSignal(
                                bar=i,
                                direction=-1,
                                price=closes[i],
                                hour=hours[i],
                                day=days[i],
                                attributes={
                                    'asian_start': asian_start,
                                    'asian_end': asian_end,
                                    'range_high': range_high,
                                    'range_low': range_low,
                                    'range_size': range_size,
                                    'range_atr': range_atr,
                                    'atr': atr[i],
                                    'atr_pips': atr[i] / self._pip_size if self._pip_size > 0 else 0,
                                    'ma200': ma200[i],
                                    'hours_after_range': hour - asian_end,
                                }
                            ))

                    # Reset after entry window passes
                    if range_complete and hour >= asian_end + 6:
                        range_complete = False

        # Remove duplicates
        seen = set()
        unique = []
        for s in signals:
            key = (s.bar, s.direction, s.attributes['asian_start'], s.attributes['asian_end'])
            if key not in seen:
                seen.add(key)
                unique.append(s)

        unique.sort(key=lambda x: x.bar)
        return unique

    def filter_signals(self, signals: List[FastSignal], params: Dict[str, Any]) -> List[FastSignal]:
        result = []
        target_start = params.get('asian_start_hour', 0)
        target_end = params.get('asian_end_hour', 7)
        entry_window = params.get('entry_window_hours', 3)
        min_range = params.get('min_range_atr', 0.5)
        max_range = params.get('max_range_atr', 3.0)
        buffer_pct = params.get('breakout_buffer_pct', 10) / 100.0

        for s in signals:
            attr = s.attributes
            if attr['asian_start'] != target_start or attr['asian_end'] != target_end:
                continue

            # Entry window filter
            if attr['hours_after_range'] >= entry_window:
                continue

            # Range size filters
            if attr['range_atr'] < min_range:
                continue
            if attr['range_atr'] > max_range:
                continue

            # Buffer: price must break by buffer_pct of range
            if buffer_pct > 0:
                buffer = attr['range_size'] * buffer_pct
                if s.direction == 1 and s.price < attr['range_high'] + buffer:
                    continue
                if s.direction == -1 and s.price > attr['range_low'] - buffer:
                    continue

            # Trend filter
            if params.get('use_trend_filter', False):
                if attr['ma200'] > 0:
                    if s.direction == 1 and s.price < attr['ma200']:
                        continue
                    if s.direction == -1 and s.price > attr['ma200']:
                        continue

            result.append(s)
        return result

    def compute_sl_tp(self, signal: FastSignal, params: Dict[str, Any], pip_size: float) -> Tuple[float, float]:
        attr = signal.attributes
        sl_mode = params.get('sl_mode', 'range')

        if sl_mode == 'range':
            # SL at opposite end of range
            if signal.direction == 1:
                sl_price = attr['range_low']
                sl_pips = (signal.price - sl_price) / pip_size
            else:
                sl_price = attr['range_high']
                sl_pips = (sl_price - signal.price) / pip_size
            sl_pips = max(sl_pips, 10.0)
        else:
            atr_pips = attr['atr'] / pip_size if pip_size > 0 else 35.0
            sl_pips = max(atr_pips * params.get('sl_atr_mult', 1.5), 10.0)

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
                ('hours', np.int64), ('days', np.int64),
                ('asian_starts', np.int64), ('asian_ends', np.int64),
                ('range_highs', np.float64), ('range_lows', np.float64),
                ('range_sizes', np.float64), ('range_atrs', np.float64),
                ('atr', np.float64), ('atr_pips', np.float64),
                ('ma200', np.float64), ('hours_after_range', np.int64),
            ]}

        arrays = {
            'bars': np.empty(n, dtype=np.int64),
            'prices': np.empty(n, dtype=np.float64),
            'directions': np.empty(n, dtype=np.int64),
            'hours': np.empty(n, dtype=np.int64),
            'days': np.empty(n, dtype=np.int64),
            'asian_starts': np.empty(n, dtype=np.int64),
            'asian_ends': np.empty(n, dtype=np.int64),
            'range_highs': np.empty(n, dtype=np.float64),
            'range_lows': np.empty(n, dtype=np.float64),
            'range_sizes': np.empty(n, dtype=np.float64),
            'range_atrs': np.empty(n, dtype=np.float64),
            'atr': np.empty(n, dtype=np.float64),
            'atr_pips': np.empty(n, dtype=np.float64),
            'ma200': np.empty(n, dtype=np.float64),
            'hours_after_range': np.empty(n, dtype=np.int64),
        }

        for i, s in enumerate(signals):
            arrays['bars'][i] = s.bar
            arrays['prices'][i] = s.price
            arrays['directions'][i] = s.direction
            arrays['hours'][i] = s.hour
            arrays['days'][i] = s.day
            a = s.attributes
            arrays['asian_starts'][i] = a['asian_start']
            arrays['asian_ends'][i] = a['asian_end']
            arrays['range_highs'][i] = a['range_high']
            arrays['range_lows'][i] = a['range_low']
            arrays['range_sizes'][i] = a['range_size']
            arrays['range_atrs'][i] = a['range_atr']
            arrays['atr'][i] = a['atr']
            arrays['atr_pips'][i] = a['atr_pips']
            arrays['ma200'][i] = a['ma200']
            arrays['hours_after_range'][i] = a['hours_after_range']

        return arrays

    def _filter_vectorized(self, params: Dict[str, Any]) -> np.ndarray:
        va = self._vec_arrays
        n = len(va['bars'])
        if n == 0:
            return np.array([], dtype=np.bool_)

        mask = np.ones(n, dtype=np.bool_)
        mask &= va['asian_starts'] == params.get('asian_start_hour', 0)
        mask &= va['asian_ends'] == params.get('asian_end_hour', 7)
        mask &= va['hours_after_range'] < params.get('entry_window_hours', 3)
        mask &= va['range_atrs'] >= params.get('min_range_atr', 0.5)
        mask &= va['range_atrs'] <= params.get('max_range_atr', 3.0)

        # Buffer filter
        buffer_pct = params.get('breakout_buffer_pct', 10) / 100.0
        if buffer_pct > 0:
            buffer = va['range_sizes'] * buffer_pct
            long_fail = (va['directions'] == 1) & (va['prices'] < va['range_highs'] + buffer)
            short_fail = (va['directions'] == -1) & (va['prices'] > va['range_lows'] - buffer)
            mask &= ~(long_fail | short_fail)

        if params.get('use_trend_filter', False):
            ma_valid = va['ma200'] > 0
            long_fail = (va['directions'] == 1) & ma_valid & (va['prices'] < va['ma200'])
            short_fail = (va['directions'] == -1) & ma_valid & (va['prices'] > va['ma200'])
            mask &= ~(long_fail | short_fail)

        return mask

    def _compute_sl_tp_vectorized(self, mask: np.ndarray, params: Dict[str, Any], pip_size: float) -> Tuple[np.ndarray, np.ndarray]:
        va = self._vec_arrays
        prices = va['prices'][mask]
        directions = va['directions'][mask]
        atr_pips = va['atr_pips'][mask]
        range_highs = va['range_highs'][mask]
        range_lows = va['range_lows'][mask]
        n = len(prices)

        sl_mode = params.get('sl_mode', 'range')

        if sl_mode == 'range':
            # SL at opposite range edge
            is_long = directions == 1
            sl_pips = np.where(
                is_long,
                (prices - range_lows) / pip_size,
                (range_highs - prices) / pip_size
            )
            sl_pips = np.maximum(sl_pips, 10.0)
        else:
            sl_pips = np.maximum(atr_pips * params.get('sl_atr_mult', 1.5), 10.0)

        tp_pips = np.maximum(sl_pips * params.get('tp_rr_ratio', 5.0), sl_pips)

        is_long = directions == 1
        sl_prices = np.where(is_long, prices - sl_pips * pip_size, prices + sl_pips * pip_size)
        tp_prices = np.where(is_long, prices + tp_pips * pip_size, prices - tp_pips * pip_size)
        return sl_prices, tp_prices

    def _get_management_arrays_vectorized(self, mask: np.ndarray, params: Dict[str, Any], sl_prices: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        n = int(np.sum(mask))
        if n == 0:
            return None
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
            'max_bars': np.zeros(n, dtype=np.int64),
        }
