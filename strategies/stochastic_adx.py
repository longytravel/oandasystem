"""
Stochastic Oscillator + ADX Strategy

Classic momentum + trend strength combination. One of the most widely-used
forex strategies, popularized in trading literature.

Entry:
- Stochastic %K crosses %D in oversold zone (buy) or overbought zone (sell)
- ADX must be above threshold confirming strong trend
- Optional: +DI/-DI directional agreement

Exit: SL at ATR multiple; TP at RR ratio.

The logic: Stochastic identifies overbought/oversold conditions (momentum),
while ADX confirms whether the market is trending strongly enough to trade.

Parameter groups (3 groups, 10 total params):
  1. signal (4 params): stoch_k, stoch_d, stoch_smooth, adx_period
  2. filters (3 params): adx_min, overbought, oversold
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


class StochasticADXStrategy(FastStrategy):
    """
    Stochastic + ADX Strategy.

    Entry on stochastic crossover in OB/OS zones, confirmed by ADX trend strength.
    """

    name = "Stochastic_ADX"
    version = "1.0"

    ATR_PERIOD = 14

    # Pre-compute for these stochastic K periods
    STOCH_K_PERIODS = [5, 9, 14, 21]
    STOCH_D_PERIODS = [3, 5]
    ADX_PERIODS = [14, 20]

    def get_parameter_groups(self) -> Optional[Dict[str, ParameterGroup]]:
        groups = {}

        signal = ParameterGroup('signal', 'Stochastic and ADX parameters')
        signal.add_param('stoch_k', [5, 9, 14, 21], default=14)
        signal.add_param('stoch_d', [3, 5], default=3)
        signal.add_param('stoch_smooth', [1, 3], default=3)
        signal.add_param('adx_period', [14, 20], default=14)
        groups['signal'] = signal

        filters = ParameterGroup('filters', 'Entry filter parameters')
        filters.add_param('adx_min', [15, 20, 25, 30], default=20)
        filters.add_param('overbought', [75, 80, 85], default=80)
        filters.add_param('oversold', [15, 20, 25], default=20)
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
        ma = np.zeros_like(data, dtype=np.float64)
        for i in range(period - 1, len(data)):
            ma[i] = np.mean(data[i - period + 1:i + 1])
        return ma

    def _calc_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          k_period: int, d_period: int, smooth: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic %K and %D."""
        n = len(close)
        raw_k = np.zeros(n, dtype=np.float64)

        for i in range(k_period - 1, n):
            highest = np.max(high[i - k_period + 1:i + 1])
            lowest = np.min(low[i - k_period + 1:i + 1])
            if highest != lowest:
                raw_k[i] = (close[i] - lowest) / (highest - lowest) * 100
            else:
                raw_k[i] = 50.0

        # Smooth %K
        if smooth > 1:
            k = self._calc_sma(raw_k, smooth)
        else:
            k = raw_k

        # %D is SMA of %K
        d = self._calc_sma(k, d_period)

        return k, d

    def _calc_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ADX, +DI, -DI."""
        n = len(close)
        plus_dm = np.zeros(n, dtype=np.float64)
        minus_dm = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        tr = np.maximum(high - low, np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ))
        tr[0] = high[0] - low[0]

        # Smoothed TR, +DM, -DM using Wilder's smoothing
        atr = np.zeros(n, dtype=np.float64)
        smooth_plus = np.zeros(n, dtype=np.float64)
        smooth_minus = np.zeros(n, dtype=np.float64)

        if n > period:
            atr[period] = np.sum(tr[1:period + 1])
            smooth_plus[period] = np.sum(plus_dm[1:period + 1])
            smooth_minus[period] = np.sum(minus_dm[1:period + 1])

            for i in range(period + 1, n):
                atr[i] = atr[i - 1] - atr[i - 1] / period + tr[i]
                smooth_plus[i] = smooth_plus[i - 1] - smooth_plus[i - 1] / period + plus_dm[i]
                smooth_minus[i] = smooth_minus[i - 1] - smooth_minus[i - 1] / period + minus_dm[i]

        # +DI and -DI
        plus_di = np.zeros(n, dtype=np.float64)
        minus_di = np.zeros(n, dtype=np.float64)
        valid = atr > 0
        plus_di[valid] = 100 * smooth_plus[valid] / atr[valid]
        minus_di[valid] = 100 * smooth_minus[valid] / atr[valid]

        # DX and ADX
        dx = np.zeros(n, dtype=np.float64)
        di_sum = plus_di + minus_di
        di_valid = di_sum > 0
        dx[di_valid] = 100 * np.abs(plus_di[di_valid] - minus_di[di_valid]) / di_sum[di_valid]

        adx = np.zeros(n, dtype=np.float64)
        if n > 2 * period:
            adx[2 * period] = np.mean(dx[period + 1:2 * period + 1])
            for i in range(2 * period + 1, n):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        return adx, plus_di, minus_di

    def precompute(self, df: pd.DataFrame) -> List[FastSignal]:
        """Pre-compute stochastic crossover signals with ADX."""
        signals = []
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        hours = df.index.hour.values
        days = df.index.dayofweek.values
        n = len(closes)

        atr = self._calc_atr(highs, lows, closes, self.ATR_PERIOD)

        # Pre-compute ADX for all periods
        adx_cache = {}
        for adx_period in self.ADX_PERIODS:
            adx_cache[adx_period] = self._calc_adx(highs, lows, closes, adx_period)

        # Pre-compute stochastic for all combos
        for k_period in self.STOCH_K_PERIODS:
            for d_period in self.STOCH_D_PERIODS:
                for smooth in [1, 3]:
                    k, d = self._calc_stochastic(highs, lows, closes, k_period, d_period, smooth)

                    for adx_period in self.ADX_PERIODS:
                        adx, plus_di, minus_di = adx_cache[adx_period]

                        # Find crossover signals
                        for i in range(1, n):
                            if atr[i] <= 0:
                                continue

                            # Bullish: %K crosses above %D
                            if k[i] > d[i] and k[i - 1] <= d[i - 1]:
                                signals.append(FastSignal(
                                    bar=i,
                                    direction=1,
                                    price=closes[i],
                                    hour=hours[i],
                                    day=days[i],
                                    attributes={
                                        'stoch_k_period': k_period,
                                        'stoch_d_period': d_period,
                                        'stoch_smooth': smooth,
                                        'adx_period': adx_period,
                                        'k_value': k[i],
                                        'd_value': d[i],
                                        'adx_value': adx[i],
                                        'plus_di': plus_di[i],
                                        'minus_di': minus_di[i],
                                        'atr': atr[i],
                                        'atr_pips': atr[i] / self._pip_size if self._pip_size > 0 else 0,
                                    }
                                ))

                            # Bearish: %K crosses below %D
                            if k[i] < d[i] and k[i - 1] >= d[i - 1]:
                                signals.append(FastSignal(
                                    bar=i,
                                    direction=-1,
                                    price=closes[i],
                                    hour=hours[i],
                                    day=days[i],
                                    attributes={
                                        'stoch_k_period': k_period,
                                        'stoch_d_period': d_period,
                                        'stoch_smooth': smooth,
                                        'adx_period': adx_period,
                                        'k_value': k[i],
                                        'd_value': d[i],
                                        'adx_value': adx[i],
                                        'plus_di': plus_di[i],
                                        'minus_di': minus_di[i],
                                        'atr': atr[i],
                                        'atr_pips': atr[i] / self._pip_size if self._pip_size > 0 else 0,
                                    }
                                ))

        # Remove duplicates (same bar, direction, and param combo)
        seen = set()
        unique = []
        for s in signals:
            a = s.attributes
            key = (s.bar, s.direction, a['stoch_k_period'], a['stoch_d_period'],
                   a['stoch_smooth'], a['adx_period'])
            if key not in seen:
                seen.add(key)
                unique.append(s)

        unique.sort(key=lambda x: x.bar)
        return unique

    def filter_signals(self, signals: List[FastSignal], params: Dict[str, Any]) -> List[FastSignal]:
        result = []
        target_k = params.get('stoch_k', 14)
        target_d = params.get('stoch_d', 3)
        target_smooth = params.get('stoch_smooth', 3)
        target_adx_period = params.get('adx_period', 14)
        adx_min = params.get('adx_min', 20)
        overbought = params.get('overbought', 80)
        oversold = params.get('oversold', 20)

        for s in signals:
            attr = s.attributes
            if attr['stoch_k_period'] != target_k:
                continue
            if attr['stoch_d_period'] != target_d:
                continue
            if attr['stoch_smooth'] != target_smooth:
                continue
            if attr['adx_period'] != target_adx_period:
                continue

            # ADX must be above minimum
            if attr['adx_value'] < adx_min:
                continue

            # Stochastic zone filter
            if s.direction == 1:
                # Buy: stochastic should be in oversold zone
                if attr['k_value'] > oversold + 20:  # Allow some buffer above oversold
                    continue
            else:
                # Sell: stochastic should be in overbought zone
                if attr['k_value'] < overbought - 20:  # Allow some buffer below overbought
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
                ('hours', np.int64), ('days', np.int64),
                ('stoch_k_periods', np.int64), ('stoch_d_periods', np.int64),
                ('stoch_smooths', np.int64), ('adx_periods', np.int64),
                ('k_values', np.float64), ('d_values', np.float64),
                ('adx_values', np.float64), ('plus_di', np.float64),
                ('minus_di', np.float64), ('atr', np.float64), ('atr_pips', np.float64),
            ]}

        arrays = {
            'bars': np.empty(n, dtype=np.int64),
            'prices': np.empty(n, dtype=np.float64),
            'directions': np.empty(n, dtype=np.int64),
            'hours': np.empty(n, dtype=np.int64),
            'days': np.empty(n, dtype=np.int64),
            'stoch_k_periods': np.empty(n, dtype=np.int64),
            'stoch_d_periods': np.empty(n, dtype=np.int64),
            'stoch_smooths': np.empty(n, dtype=np.int64),
            'adx_periods': np.empty(n, dtype=np.int64),
            'k_values': np.empty(n, dtype=np.float64),
            'd_values': np.empty(n, dtype=np.float64),
            'adx_values': np.empty(n, dtype=np.float64),
            'plus_di': np.empty(n, dtype=np.float64),
            'minus_di': np.empty(n, dtype=np.float64),
            'atr': np.empty(n, dtype=np.float64),
            'atr_pips': np.empty(n, dtype=np.float64),
        }

        for i, s in enumerate(signals):
            arrays['bars'][i] = s.bar
            arrays['prices'][i] = s.price
            arrays['directions'][i] = s.direction
            arrays['hours'][i] = s.hour
            arrays['days'][i] = s.day
            a = s.attributes
            arrays['stoch_k_periods'][i] = a['stoch_k_period']
            arrays['stoch_d_periods'][i] = a['stoch_d_period']
            arrays['stoch_smooths'][i] = a['stoch_smooth']
            arrays['adx_periods'][i] = a['adx_period']
            arrays['k_values'][i] = a['k_value']
            arrays['d_values'][i] = a['d_value']
            arrays['adx_values'][i] = a['adx_value']
            arrays['plus_di'][i] = a['plus_di']
            arrays['minus_di'][i] = a['minus_di']
            arrays['atr'][i] = a['atr']
            arrays['atr_pips'][i] = a['atr_pips']

        return arrays

    def _filter_vectorized(self, params: Dict[str, Any]) -> np.ndarray:
        va = self._vec_arrays
        n = len(va['bars'])
        if n == 0:
            return np.array([], dtype=np.bool_)

        mask = np.ones(n, dtype=np.bool_)
        mask &= va['stoch_k_periods'] == params.get('stoch_k', 14)
        mask &= va['stoch_d_periods'] == params.get('stoch_d', 3)
        mask &= va['stoch_smooths'] == params.get('stoch_smooth', 3)
        mask &= va['adx_periods'] == params.get('adx_period', 14)

        # ADX minimum
        mask &= va['adx_values'] >= params.get('adx_min', 20)

        # Stochastic zone filter
        overbought = params.get('overbought', 80)
        oversold = params.get('oversold', 20)
        long_zone_fail = (va['directions'] == 1) & (va['k_values'] > oversold + 20)
        short_zone_fail = (va['directions'] == -1) & (va['k_values'] < overbought - 20)
        mask &= ~(long_zone_fail | short_zone_fail)

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
