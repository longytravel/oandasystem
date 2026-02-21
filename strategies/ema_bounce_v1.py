"""
EMA Bounce Strategy V1

Source: MQL5 article "From Novice to Expert: Automating Intraday Strategies"
https://www.mql5.com/en/articles/21283

Concept: Trade price bounces off the EMA during a trend.
- Buy: Price in uptrend (above EMA), candle dips below EMA, closes back above
- Sell: Price in downtrend (below EMA), candle pokes above EMA, closes back below
The EMA acts as dynamic support/resistance.

Filters: pin bar confirmation, tolerance (near-miss), EMA slope (trend strength),
minimum ATR (avoid chop).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.fast_strategy import FastStrategy, FastSignal, ParameterGroup
from research.strategy_helpers import (
    standard_risk_group, standard_management_group, standard_time_group,
    build_core_arrays, extract_attribute_arrays,
    vectorized_sl_tp, vectorized_management, vectorized_time_filter,
    calc_atr, calc_ema,
)


class EMABounceFastV1(FastStrategy):
    """
    EMA Bounce - trade rejections off EMA as dynamic support/resistance.

    Parameter groups:
    1. signal (3 params): EMA period, pin bar settings
    2. filters (4 params): Tolerance, trend strength, ATR minimum
    3. risk (7 params): Standard SL/TP
    4. management (8 params): Standard trailing/BE/partial
    5. time (6 params): Standard time filters

    Total: 28 parameters
    """

    name = "EMA_Bounce_v1"
    version = "1.0"

    ATR_PERIOD = 14
    # Precompute signals for all EMA periods — filter selects which
    EMA_PERIODS = [20, 30, 50, 100]
    # Maximum tolerance for precomputation (generous; filter_signals narrows)
    MAX_TOLERANCE_PIPS = 15

    def get_parameter_groups(self) -> Optional[Dict[str, ParameterGroup]]:
        groups = {}

        # === Signal (3 params) ===
        signal = ParameterGroup('signal', 'EMA bounce signal parameters')
        signal.add_param('ema_period', self.EMA_PERIODS, default=50)
        signal.add_param('use_pin_bar', [True, False], default=False)
        signal.add_param('pin_bar_ratio', [1.5, 2.0, 3.0], default=2.0)
        groups['signal'] = signal

        # === Filters (4 params) ===
        filters = ParameterGroup('filters', 'Entry confirmation filters')
        filters.add_param('use_tolerance', [True, False], default=False)
        filters.add_param('tolerance_pips', [3, 5, 10], default=5)
        filters.add_param('use_trend_strength', [True, False], default=False)
        filters.add_param('min_ema_slope', [0.5, 1.0, 2.0], default=1.0)
        groups['filters'] = filters

        # === Standard groups ===
        groups['risk'] = standard_risk_group()
        groups['management'] = standard_management_group()
        groups['time'] = standard_time_group()

        return groups

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        groups = self.get_parameter_groups()
        space = {}
        for group in groups.values():
            space.update(group.get_param_space())
        return space

    def precompute(self, df: pd.DataFrame) -> List[FastSignal]:
        """
        Precompute EMA bounce signals for ALL EMA periods.

        A bounce signal occurs when:
        - Buy: prev close > EMA (uptrend), current low pierces below or near EMA,
               current close > EMA (bounced back)
        - Sell: prev close < EMA (downtrend), current high pierces above or near EMA,
                current close < EMA (bounced back)

        We use a generous tolerance for precomputation so filter_signals can
        narrow it per trial.
        """
        signals = []
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        hours = df.index.hour.values
        days = df.index.dayofweek.values
        n = len(closes)

        atr = calc_atr(highs, lows, closes, self.ATR_PERIOD)
        pip = self._pip_size
        max_tol = self.MAX_TOLERANCE_PIPS * pip

        # Precompute EMAs for all periods
        emas = {}
        for period in self.EMA_PERIODS:
            emas[period] = calc_ema(closes, period)

        # Precompute EMA slopes (5-bar lookback, in pips per bar)
        ema_slopes = {}
        for period in self.EMA_PERIODS:
            ema = emas[period]
            slope = np.zeros(n, dtype=np.float64)
            for i in range(5, n):
                slope[i] = (ema[i] - ema[i - 5]) / (5 * pip)
            ema_slopes[period] = slope

        # Scan for bounce signals
        start_bar = max(self.EMA_PERIODS) + 5  # ensure all EMAs are warm

        for period in self.EMA_PERIODS:
            ema = emas[period]
            slope = ema_slopes[period]

            for i in range(start_bar, n):
                if atr[i] < pip:  # skip if ATR is essentially zero
                    continue

                # === BUY: uptrend bounce off EMA ===
                # Previous close above EMA (uptrend)
                if closes[i - 1] > ema[i - 1]:
                    # Current low dips to or below EMA (with tolerance)
                    if lows[i] <= ema[i] + max_tol:
                        # Current close back above EMA
                        if closes[i] > ema[i]:
                            # Touch distance: negative = pierced through, positive = near miss
                            touch_dist = (lows[i] - ema[i]) / pip

                            # Pin bar metrics
                            body = abs(closes[i] - opens[i])
                            lower_wick = min(opens[i], closes[i]) - lows[i]
                            pin_ratio = (lower_wick / body) if body > 0 else (10.0 if lower_wick > 0 else 0.0)

                            signals.append(FastSignal(
                                bar=i,
                                direction=1,
                                price=closes[i],
                                hour=hours[i],
                                day=days[i],
                                attributes={
                                    'atr': atr[i],
                                    'atr_pips': atr[i] / pip,
                                    'ema_period': period,
                                    'touch_distance': touch_dist,
                                    'pin_ratio': pin_ratio,
                                    'ema_slope': slope[i],
                                }
                            ))

                # === SELL: downtrend bounce off EMA ===
                if closes[i - 1] < ema[i - 1]:
                    if highs[i] >= ema[i] - max_tol:
                        if closes[i] < ema[i]:
                            touch_dist = (ema[i] - highs[i]) / pip

                            body = abs(closes[i] - opens[i])
                            upper_wick = highs[i] - max(opens[i], closes[i])
                            pin_ratio = (upper_wick / body) if body > 0 else (10.0 if upper_wick > 0 else 0.0)

                            signals.append(FastSignal(
                                bar=i,
                                direction=-1,
                                price=closes[i],
                                hour=hours[i],
                                day=days[i],
                                attributes={
                                    'atr': atr[i],
                                    'atr_pips': atr[i] / pip,
                                    'ema_period': period,
                                    'touch_distance': touch_dist,
                                    'pin_ratio': pin_ratio,
                                    'ema_slope': slope[i],
                                }
                            ))

        # Deduplicate (same bar + direction, keep first = smallest EMA period)
        seen = set()
        unique = []
        for s in signals:
            key = (s.bar, s.direction, s.attributes['ema_period'])
            if key not in seen:
                seen.add(key)
                unique.append(s)

        unique.sort(key=lambda x: x.bar)
        return unique

    def filter_signals(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any],
    ) -> List[FastSignal]:
        result = []
        ema_period = params.get('ema_period', 50)
        use_pin_bar = params.get('use_pin_bar', False)
        pin_bar_ratio = params.get('pin_bar_ratio', 2.0)
        use_tolerance = params.get('use_tolerance', False)
        tolerance_pips = params.get('tolerance_pips', 5)
        use_trend_strength = params.get('use_trend_strength', False)
        min_ema_slope = params.get('min_ema_slope', 1.0)

        for s in signals:
            attr = s.attributes

            # EMA period filter
            if attr['ema_period'] != ema_period:
                continue

            # Tolerance filter
            touch = attr['touch_distance']
            if use_tolerance:
                # Allow near-miss up to tolerance_pips
                if touch > tolerance_pips:
                    continue
            else:
                # Require actual pierce (touch_distance <= 0)
                if touch > 0:
                    continue

            # Pin bar filter
            if use_pin_bar:
                if attr['pin_ratio'] < pin_bar_ratio:
                    continue

            # Trend strength filter (EMA slope)
            if use_trend_strength:
                slope = attr['ema_slope']
                if s.direction == 1 and slope < min_ema_slope:
                    continue
                if s.direction == -1 and slope > -min_ema_slope:
                    continue

            # Time filters
            if params.get('use_time_filter', False):
                if s.day == 0 and not params.get('trade_monday', False):
                    continue
                if s.day == 4 and not params.get('trade_friday', False):
                    continue
                if s.day == 4 and s.hour >= params.get('friday_close_hour', 18):
                    continue
                if s.day == 6:
                    continue
                start_h = params.get('trade_start_hour', 4)
                end_h = params.get('trade_end_hour', 22)
                if start_h < end_h:
                    if s.hour < start_h or s.hour >= end_h:
                        continue
                else:
                    if s.hour < start_h and s.hour >= end_h:
                        continue

            result.append(s)

        return result

    def compute_sl_tp(
        self,
        signal: FastSignal,
        params: Dict[str, Any],
        pip_size: float,
    ) -> Tuple[float, float]:
        attr = signal.attributes

        # === STOP LOSS ===
        sl_mode = params.get('sl_mode', 'atr')
        if sl_mode == 'fixed':
            sl_pips = params.get('sl_fixed_pips', 50)
        elif sl_mode == 'atr':
            sl_pips = attr['atr_pips'] * params.get('sl_atr_pct', 100) / 100.0
        else:
            sl_pips = params.get('sl_fixed_pips', 50)
        sl_pips = max(sl_pips, 10)

        # === TAKE PROFIT ===
        tp_mode = params.get('tp_mode', 'rr')
        if tp_mode == 'rr':
            tp_pips = sl_pips * params.get('tp_rr_ratio', 2.0)
        elif tp_mode == 'atr':
            tp_pips = attr['atr_pips'] * params.get('tp_atr_mult', 3.0)
        elif tp_mode == 'fixed':
            tp_pips = params.get('tp_fixed_pips', 30)
        else:
            tp_pips = sl_pips * params.get('tp_rr_ratio', 2.0)
        tp_pips = max(tp_pips, sl_pips)

        if signal.direction == 1:
            return signal.price - sl_pips * pip_size, signal.price + tp_pips * pip_size
        else:
            return signal.price + sl_pips * pip_size, signal.price - tp_pips * pip_size

    # ===================================================================
    # VECTORIZED PATH
    # ===================================================================

    def _build_signal_arrays(self, signals: list) -> Dict[str, np.ndarray]:
        arrays = build_core_arrays(signals)
        attrs = extract_attribute_arrays(signals, [
            'atr', 'atr_pips', 'ema_period', 'touch_distance',
            'pin_ratio', 'ema_slope',
        ])
        arrays.update(attrs)
        return arrays

    def _filter_vectorized(self, params: Dict[str, Any]) -> np.ndarray:
        va = self._vec_arrays
        n = len(va['bars'])
        if n == 0:
            return np.array([], dtype=np.bool_)

        mask = np.ones(n, dtype=np.bool_)

        # EMA period
        mask &= va['ema_period'] == params.get('ema_period', 50)

        # Tolerance
        if params.get('use_tolerance', False):
            mask &= va['touch_distance'] <= params.get('tolerance_pips', 5)
        else:
            mask &= va['touch_distance'] <= 0

        # Pin bar
        if params.get('use_pin_bar', False):
            mask &= va['pin_ratio'] >= params.get('pin_bar_ratio', 2.0)

        # Trend strength
        if params.get('use_trend_strength', False):
            min_slope = params.get('min_ema_slope', 1.0)
            is_long = va['directions'] == 1
            is_short = va['directions'] == -1
            slope_fail_long = is_long & (va['ema_slope'] < min_slope)
            slope_fail_short = is_short & (va['ema_slope'] > -min_slope)
            mask &= ~(slope_fail_long | slope_fail_short)

        # Standard time filter
        mask = vectorized_time_filter(va, mask, params)

        return mask

    def _compute_sl_tp_vectorized(
        self,
        mask: np.ndarray,
        params: Dict[str, Any],
        pip_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return vectorized_sl_tp(self._vec_arrays, mask, params, pip_size)

    def _get_management_arrays_vectorized(
        self,
        mask: np.ndarray,
        params: Dict[str, Any],
        sl_prices: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        return vectorized_management(
            self._vec_arrays, mask, params, sl_prices, self._pip_size
        )

    # ===================================================================
    # NON-VECTORIZED MANAGEMENT (fallback)
    # ===================================================================

    def get_management_arrays(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any],
    ) -> Optional[Dict[str, np.ndarray]]:
        n = len(signals)
        if n == 0:
            return None

        atr_pips_arr = np.array([s.attributes['atr_pips'] for s in signals], dtype=np.float64)

        # Trailing (ATR-scaled)
        use_trailing = np.full(n, params.get('use_trailing', False), dtype=np.bool_)
        trail_start = atr_pips_arr * params.get('trail_start_atr', 1.0)
        trail_step = atr_pips_arr * params.get('trail_step_atr', 0.3)

        # Breakeven (ATR-scaled)
        use_be = np.full(n, params.get('use_break_even', False), dtype=np.bool_)
        be_trigger = atr_pips_arr * params.get('be_atr_mult', 0.5)
        be_offset = atr_pips_arr * params.get('be_offset_atr', 0)

        # Partial close
        use_partial = np.full(n, params.get('use_partial_close', False), dtype=np.bool_)
        partial_pct = np.full(n, params.get('partial_close_pct', 0.3), dtype=np.float64)

        # Partial target as ratio of TP distance
        partial_target_rr = params.get('partial_target_rr', 0.5)
        partial_target = np.zeros(n, dtype=np.float64)
        for i, sig in enumerate(signals):
            sl_pips = self._get_sl_pips(sig, params)
            tp_pips = self._get_tp_pips(sl_pips, params, sig.attributes['atr_pips'])
            partial_target[i] = tp_pips * partial_target_rr

        # Max bars
        if params.get('use_max_bars', False):
            max_bars = np.full(n, params.get('max_bars_limit', 100), dtype=np.int64)
        else:
            max_bars = np.zeros(n, dtype=np.int64)

        return {
            'use_trailing': use_trailing,
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

    def _get_sl_pips(self, signal: FastSignal, params: Dict[str, Any]) -> float:
        sl_mode = params.get('sl_mode', 'atr')
        if sl_mode == 'fixed':
            return params.get('sl_fixed_pips', 50)
        elif sl_mode == 'atr':
            return signal.attributes['atr_pips'] * params.get('sl_atr_pct', 100) / 100.0
        return params.get('sl_fixed_pips', 50)

    def _get_tp_pips(self, sl_pips: float, params: Dict[str, Any], atr_pips: float) -> float:
        tp_mode = params.get('tp_mode', 'rr')
        if tp_mode == 'rr':
            return sl_pips * params.get('tp_rr_ratio', 2.0)
        elif tp_mode == 'atr':
            return atr_pips * params.get('tp_atr_mult', 3.0)
        elif tp_mode == 'fixed':
            return params.get('tp_fixed_pips', 30)
        return sl_pips * params.get('tp_rr_ratio', 2.0)
