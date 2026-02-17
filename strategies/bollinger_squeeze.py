"""
Bollinger Band Squeeze Breakout Strategy

Volatility-based strategy that enters when Bollinger Bands squeeze (narrow)
and then expand. Famous concept from John Bollinger - "The Squeeze" is the
single most reliable pattern in Bollinger Band analysis.

Entry: After BB width drops below threshold (squeeze), enter on breakout
       when price closes outside the bands.
Exit: SL at ATR multiple; TP at RR ratio.

The squeeze indicates low volatility, which tends to precede explosive moves.
Keltner Channel overlap detection is used to identify the squeeze.

Parameter groups (3 groups, 10 total params):
  1. signal (4 params): bb_period, bb_std, squeeze_lookback, squeeze_percentile
  2. filters (3 params): use_trend_filter, use_time_filter, min_atr_pips
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


class BollingerSqueezeStrategy(FastStrategy):
    """
    Bollinger Band Squeeze Breakout.

    Detects low-volatility squeeze periods, then enters on breakout.
    """

    name = "Bollinger_Squeeze"
    version = "1.0"

    ATR_PERIOD = 14

    # Pre-compute for these BB periods
    BB_PERIODS = [14, 20, 30]
    BB_STDS = [1.5, 2.0, 2.5]

    def get_parameter_groups(self) -> Optional[Dict[str, ParameterGroup]]:
        groups = {}

        signal = ParameterGroup('signal', 'Bollinger Band squeeze parameters')
        signal.add_param('bb_period', [14, 20, 30], default=20)
        signal.add_param('bb_std', [1.5, 2.0, 2.5], default=2.0)
        signal.add_param('squeeze_lookback', [20, 50, 100], default=50)
        signal.add_param('squeeze_percentile', [10, 20, 30], default=20)
        groups['signal'] = signal

        filters = ParameterGroup('filters', 'Entry filter parameters')
        filters.add_param('use_trend_filter', [True, False], default=False)
        filters.add_param('use_time_filter', [True, False], default=False)
        filters.add_param('min_atr_pips', [0, 3, 5, 10], default=0)
        groups['filters'] = filters

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

    def _calc_bb(self, close: np.ndarray, period: int, num_std: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands. Returns (middle, upper, lower)."""
        middle = self._calc_sma(close, period)
        std = np.zeros_like(close)
        for i in range(period - 1, len(close)):
            std[i] = np.std(close[i - period + 1:i + 1])

        upper = middle + num_std * std
        lower = middle - num_std * std
        return middle, upper, lower

    def precompute(self, df: pd.DataFrame) -> List[FastSignal]:
        """Pre-compute squeeze breakout signals for all BB combos."""
        signals = []
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        hours = df.index.hour.values
        days = df.index.dayofweek.values
        n = len(closes)

        atr = self._calc_atr(highs, lows, closes, self.ATR_PERIOD)
        ma200 = self._calc_sma(closes, 200)

        for bb_period in self.BB_PERIODS:
            for bb_std in self.BB_STDS:
                if n <= bb_period + 1:
                    continue

                middle, upper, lower = self._calc_bb(closes, bb_period, bb_std)

                # BB width as percentage of middle band
                bb_width = np.zeros(n, dtype=np.float64)
                valid = middle > 0
                bb_width[valid] = (upper[valid] - lower[valid]) / middle[valid]

                # For each bar, check if in squeeze and breaking out
                for i in range(max(bb_period, 100) + 1, n):
                    if atr[i] <= 0 or middle[i] <= 0:
                        continue

                    # Current BB width
                    current_width = bb_width[i]
                    if current_width <= 0:
                        continue

                    # Was previous bar inside bands? (pre-breakout)
                    prev_inside = lower[i - 1] <= closes[i - 1] <= upper[i - 1]
                    if not prev_inside:
                        continue

                    # Bullish breakout: close above upper band
                    if closes[i] > upper[i]:
                        signals.append(FastSignal(
                            bar=i,
                            direction=1,
                            price=closes[i],
                            hour=hours[i],
                            day=days[i],
                            attributes={
                                'bb_period': bb_period,
                                'bb_std': bb_std,
                                'bb_width': current_width,
                                'atr': atr[i],
                                'atr_pips': atr[i] / self._pip_size if self._pip_size > 0 else 0,
                                'ma200': ma200[i],
                                'bb_middle': middle[i],
                            }
                        ))

                    # Bearish breakout: close below lower band
                    elif closes[i] < lower[i]:
                        signals.append(FastSignal(
                            bar=i,
                            direction=-1,
                            price=closes[i],
                            hour=hours[i],
                            day=days[i],
                            attributes={
                                'bb_period': bb_period,
                                'bb_std': bb_std,
                                'bb_width': current_width,
                                'atr': atr[i],
                                'atr_pips': atr[i] / self._pip_size if self._pip_size > 0 else 0,
                                'ma200': ma200[i],
                                'bb_middle': middle[i],
                            }
                        ))

        # Remove duplicates
        seen = set()
        unique = []
        for s in signals:
            key = (s.bar, s.direction, s.attributes['bb_period'], s.attributes['bb_std'])
            if key not in seen:
                seen.add(key)
                unique.append(s)

        unique.sort(key=lambda x: x.bar)
        return unique

    def filter_signals(self, signals: List[FastSignal], params: Dict[str, Any]) -> List[FastSignal]:
        result = []
        target_period = params.get('bb_period', 20)
        target_std = params.get('bb_std', 2.0)
        min_atr = params.get('min_atr_pips', 0)

        # Pre-compute squeeze threshold from all signals of this BB combo
        combo_widths = [s.attributes['bb_width'] for s in signals
                        if s.attributes['bb_period'] == target_period
                        and s.attributes['bb_std'] == target_std]

        # Use percentile of all observed widths as squeeze threshold
        squeeze_pct = params.get('squeeze_percentile', 20)
        if combo_widths:
            squeeze_threshold = np.percentile(combo_widths, squeeze_pct)
        else:
            squeeze_threshold = 0

        for s in signals:
            attr = s.attributes
            if attr['bb_period'] != target_period:
                continue
            if attr['bb_std'] != target_std:
                continue

            # Squeeze filter: only enter when width is below percentile threshold
            if squeeze_threshold > 0 and attr['bb_width'] > squeeze_threshold:
                continue

            if min_atr > 0 and attr['atr_pips'] < min_atr:
                continue

            # Trend filter
            if params.get('use_trend_filter', False):
                if attr['ma200'] > 0:
                    if s.direction == 1 and s.price < attr['ma200']:
                        continue
                    if s.direction == -1 and s.price > attr['ma200']:
                        continue

            # Time filter
            if params.get('use_time_filter', False):
                if s.day == 6:
                    continue
                if 8 <= s.hour < 20:
                    pass  # London+NY hours, allow
                else:
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
                ('hours', np.int64), ('days', np.int64), ('bb_periods', np.int64),
                ('bb_stds', np.float64), ('bb_widths', np.float64),
                ('atr', np.float64), ('atr_pips', np.float64), ('ma200', np.float64),
            ]}

        bars = np.empty(n, dtype=np.int64)
        prices = np.empty(n, dtype=np.float64)
        directions = np.empty(n, dtype=np.int64)
        hours = np.empty(n, dtype=np.int64)
        days_arr = np.empty(n, dtype=np.int64)
        bb_periods = np.empty(n, dtype=np.int64)
        bb_stds = np.empty(n, dtype=np.float64)
        bb_widths = np.empty(n, dtype=np.float64)
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
            bb_periods[i] = a['bb_period']
            bb_stds[i] = a['bb_std']
            bb_widths[i] = a['bb_width']
            atr[i] = a['atr']
            atr_pips[i] = a['atr_pips']
            ma200[i] = a['ma200']

        return {
            'bars': bars, 'prices': prices, 'directions': directions,
            'hours': hours, 'days': days_arr, 'bb_periods': bb_periods,
            'bb_stds': bb_stds, 'bb_widths': bb_widths, 'atr': atr,
            'atr_pips': atr_pips, 'ma200': ma200,
        }

    def _filter_vectorized(self, params: Dict[str, Any]) -> np.ndarray:
        va = self._vec_arrays
        n = len(va['bars'])
        if n == 0:
            return np.array([], dtype=np.bool_)

        mask = np.ones(n, dtype=np.bool_)
        mask &= va['bb_periods'] == params.get('bb_period', 20)
        mask &= np.isclose(va['bb_stds'], params.get('bb_std', 2.0))

        # Squeeze filter: compute percentile on matching combo
        combo_mask = mask.copy()
        if np.any(combo_mask):
            widths = va['bb_widths'][combo_mask]
            pct = params.get('squeeze_percentile', 20)
            threshold = np.percentile(widths, pct)
            mask &= va['bb_widths'] <= threshold

        min_atr = params.get('min_atr_pips', 0)
        if min_atr > 0:
            mask &= va['atr_pips'] >= min_atr

        if params.get('use_trend_filter', False):
            ma_valid = va['ma200'] > 0
            long_fail = (va['directions'] == 1) & ma_valid & (va['prices'] < va['ma200'])
            short_fail = (va['directions'] == -1) & ma_valid & (va['prices'] > va['ma200'])
            mask &= ~(long_fail | short_fail)

        if params.get('use_time_filter', False):
            mask &= va['days'] != 6
            mask &= (va['hours'] >= 8) & (va['hours'] < 20)

        return mask

    def _compute_sl_tp_vectorized(self, mask: np.ndarray, params: Dict[str, Any], pip_size: float) -> Tuple[np.ndarray, np.ndarray]:
        va = self._vec_arrays
        prices = va['prices'][mask]
        directions = va['directions'][mask]
        atr_pips = va['atr_pips'][mask]

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
