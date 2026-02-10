"""
EMA Cross Strategy - VERSION 6.1

Simple EMA crossover entry with ATR-based SL/TP.
Designed for high trade frequency (200-400 trades/3yr) to enable
ML entry filtering via pipeline --ml-exit --ml-mode entry_filter.

Entry: EMA(fast) crosses EMA(slow)
Exit: SL/TP (ATR multiples) — no ML exit, no trade management

Architecture:
- Pre-compute signals for ALL (fast, slow) combos: [8,13,21] x [34,55,89] = 9 combos
- Per trial: filter by EMA combo, compute SL/TP from ATR multiples
- 5 crossover-quality features flow to ML entry filter via get_signal_attributes()

Parameter groups (2 groups, 6 total params):
  1. signal (3 params): ema_fast_period, ema_slow_period, min_atr_pips
  2. risk (3 params): sl_atr_mult, tp_atr_mult, max_hold_bars

Created: 2026-02-07
Updated: 2026-02-09 — V6.1: removed broken ml_exit group, fixed tp_atr_mult default,
                        added 5 crossover-quality signal attributes for ML entry filter
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.fast_strategy import FastStrategy, FastSignal, ParameterGroup, ParameterDef


class EMACrossMLStrategy(FastStrategy):
    """
    EMA Cross Strategy (V6.1).

    Entry: EMA(fast) crosses EMA(slow) — simple, high-frequency
    Exit: SL/TP as ATR multiples

    Parameter groups for staged optimization:
    1. signal (3 params): EMA periods + ATR gate
    2. risk (3 params): SL/TP as ATR multiples + max hold bars

    Total: 6 parameters
    """

    name = "EMA_Cross_ML"
    version = "6.1"

    ATR_PERIOD = 14

    # EMA period combos to pre-compute
    FAST_PERIODS = [8, 13, 21]
    SLOW_PERIODS = [34, 55, 89]

    def get_parameter_groups(self) -> Optional[Dict[str, ParameterGroup]]:
        groups = {}

        # === Group 1: Signal (3 params, uses basic_backtest) ===
        signal = ParameterGroup('signal', 'EMA crossover signal parameters')
        signal.add_param('ema_fast_period', [8, 13, 21], default=13)
        signal.add_param('ema_slow_period', [34, 55, 89], default=55)
        signal.add_param('min_atr_pips', [0, 5, 10, 15], default=5)
        groups['signal'] = signal

        # === Group 2: Risk (3 params, uses full_backtest) ===
        risk = ParameterGroup('risk', 'Stop loss and take profit as ATR multiples')
        risk.add_param('sl_atr_mult', [1.5, 2.0, 2.5, 3.0], default=2.0)
        risk.add_param('tp_atr_mult', [2.0, 3.0, 4.0, 6.0, 8.0], default=4.0)
        risk.add_param('max_hold_bars', [0, 50, 100, 200], default=0)  # 0 = unlimited
        groups['risk'] = risk

        return groups

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        groups = self.get_parameter_groups()
        space = {}
        for group in groups.values():
            space.update(group.get_param_space())
        return space

    def _calc_ema(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        ema = np.zeros_like(close, dtype=np.float64)
        if len(close) < period:
            return ema

        # SMA for first value
        ema[period - 1] = np.mean(close[:period])
        mult = 2.0 / (period + 1)

        for i in range(period, len(close)):
            ema[i] = close[i] * mult + ema[i - 1] * (1.0 - mult)

        return ema

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
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

    def precompute(self, df: pd.DataFrame) -> List[FastSignal]:
        """
        Pre-compute signals for ALL (fast, slow) EMA combinations.
        Also compute 5 crossover-quality features per signal.
        """
        signals = []
        closes = df['close'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        hours = df.index.hour.values
        days = df.index.dayofweek.values
        n = len(closes)

        # Pre-compute ATR
        atr = self._calc_atr(highs, lows, closes, self.ATR_PERIOD)

        # Pre-compute EMAs for all period combos
        emas = {}
        for period in set(self.FAST_PERIODS + self.SLOW_PERIODS):
            emas[period] = self._calc_ema(closes, period)

        # Track last cross bar per (fast, slow) combo for bars_since_last_cross
        last_cross = {}

        # Generate crossover signals for each combo
        for fast_p in self.FAST_PERIODS:
            for slow_p in self.SLOW_PERIODS:
                if fast_p >= slow_p:
                    continue  # Skip invalid combos

                ema_fast = emas[fast_p]
                ema_slow = emas[slow_p]
                combo_key = (fast_p, slow_p)
                last_cross[combo_key] = -1000  # sentinel

                # Need enough bars for slow EMA to be valid
                start_bar = slow_p + 1

                # Gap = fast - slow (for cross_velocity)
                gap = ema_fast - ema_slow

                for i in range(start_bar, n):
                    # Check for valid EMA values
                    if ema_fast[i] == 0 or ema_slow[i] == 0:
                        continue
                    if ema_fast[i - 1] == 0 or ema_slow[i - 1] == 0:
                        continue

                    direction = 0

                    # Buy: EMA_fast crosses above EMA_slow
                    if ema_fast[i - 1] <= ema_slow[i - 1] and ema_fast[i] > ema_slow[i]:
                        direction = 1

                    # Sell: EMA_fast crosses below EMA_slow
                    elif ema_fast[i - 1] >= ema_slow[i - 1] and ema_fast[i] < ema_slow[i]:
                        direction = -1

                    if direction != 0:
                        atr_val = atr[i]
                        atr_pips = atr_val / self._pip_size if self._pip_size > 0 else 0

                        # === 5 crossover-quality features ===
                        # 1. EMA separation: how far apart the EMAs are at crossover (ATR-normalized)
                        ema_separation = abs(ema_fast[i] - ema_slow[i]) / atr_val if atr_val > 0 else 0.0

                        # 2. Fast EMA slope: momentum of fast EMA (3-bar lookback, ATR-normalized)
                        if i >= 3 and atr_val > 0:
                            ema_fast_slope = (ema_fast[i] - ema_fast[i - 3]) / atr_val
                        else:
                            ema_fast_slope = 0.0

                        # 3. Slow EMA slope: direction context of slow EMA
                        if i >= 3 and atr_val > 0:
                            ema_slow_slope = (ema_slow[i] - ema_slow[i - 3]) / atr_val
                        else:
                            ema_slow_slope = 0.0

                        # 4. Cross velocity: convergence speed at crossover
                        if i >= 1 and atr_val > 0:
                            cross_velocity = (gap[i] - gap[i - 1]) / atr_val
                        else:
                            cross_velocity = 0.0

                        # 5. Bars since last cross of same EMA pair
                        bars_since_last = float(i - last_cross[combo_key])

                        last_cross[combo_key] = i

                        signals.append(FastSignal(
                            bar=i,
                            direction=direction,
                            price=closes[i],  # Enter at close of crossover bar
                            hour=int(hours[i]),
                            day=int(days[i]),
                            attributes={
                                'ema_fast_period': fast_p,
                                'ema_slow_period': slow_p,
                                'atr': atr_val,
                                'atr_pips': atr_pips,
                                'ema_separation': ema_separation,
                                'ema_fast_slope': ema_fast_slope,
                                'ema_slow_slope': ema_slow_slope,
                                'cross_velocity': cross_velocity,
                                'bars_since_last_cross': bars_since_last,
                            }
                        ))

        # Remove duplicates (same bar, same direction - from overlapping combos)
        seen = set()
        unique = []
        for s in signals:
            key = (s.bar, s.direction, s.attributes['ema_fast_period'], s.attributes['ema_slow_period'])
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
        """Filter signals by matching EMA combo + ATR gate."""
        result = []
        ema_fast = params.get('ema_fast_period', 13)
        ema_slow = params.get('ema_slow_period', 55)
        min_atr = params.get('min_atr_pips', 5)

        for s in signals:
            attr = s.attributes

            # Must match selected EMA combo
            if attr['ema_fast_period'] != ema_fast:
                continue
            if attr['ema_slow_period'] != ema_slow:
                continue

            # ATR gate (0 = disabled)
            if min_atr > 0 and attr['atr_pips'] < min_atr:
                continue

            result.append(s)

        return result

    def compute_sl_tp(
        self,
        signal: FastSignal,
        params: Dict[str, Any],
        pip_size: float
    ) -> Tuple[float, float]:
        """Compute SL/TP based on ATR multiples."""
        attr = signal.attributes
        atr = attr['atr']

        sl_mult = params.get('sl_atr_mult', 2.0)
        tp_mult = params.get('tp_atr_mult', 4.0)

        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr

        if signal.direction == 1:  # Long
            sl = signal.price - sl_dist
            tp = signal.price + tp_dist
        else:  # Short
            sl = signal.price + sl_dist
            tp = signal.price - tp_dist

        return sl, tp

    def get_management_arrays(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Return management arrays — all management OFF."""
        n = len(signals)
        if n == 0:
            return None

        mgmt = {
            'use_trailing': np.zeros(n, dtype=np.bool_),
            'trail_start_pips': np.zeros(n, dtype=np.float64),
            'trail_step_pips': np.zeros(n, dtype=np.float64),
            'use_breakeven': np.zeros(n, dtype=np.bool_),
            'be_trigger_pips': np.zeros(n, dtype=np.float64),
            'be_offset_pips': np.zeros(n, dtype=np.float64),
            'use_partial': np.zeros(n, dtype=np.bool_),
            'partial_pct': np.zeros(n, dtype=np.float64),
            'partial_target_pips': np.zeros(n, dtype=np.float64),
            'max_bars': np.full(n, params.get('max_hold_bars', 0), dtype=np.int64),
            # V5 compat arrays
            'trail_mode': np.zeros(n, dtype=np.int64),
            'chandelier_atr_mult': np.full(n, 3.0, dtype=np.float64),
            'atr_pips': np.array([s.attributes.get('atr_pips', 35.0) for s in signals], dtype=np.float64),
            'stale_exit_bars': np.zeros(n, dtype=np.int64),
            # ML exit OFF — hardcoded to never fire
            'use_ml_exit': np.zeros(n, dtype=np.bool_),
            'ml_min_hold': np.zeros(n, dtype=np.int64),
            'ml_threshold': np.ones(n, dtype=np.float64),  # 1.0 = never fires
        }

        return mgmt

    # ===================================================================
    # VECTORIZED PATH - 10-50x faster per trial
    # ===================================================================

    def _build_signal_arrays(self, signals: list) -> Dict[str, np.ndarray]:
        """Convert list of FastSignal into parallel numpy arrays."""
        n = len(signals)
        if n == 0:
            return {
                'bars': np.array([], dtype=np.int64),
                'prices': np.array([], dtype=np.float64),
                'directions': np.array([], dtype=np.int64),
                'hours': np.array([], dtype=np.int64),
                'days': np.array([], dtype=np.int64),
                'ema_fast_period': np.array([], dtype=np.int64),
                'ema_slow_period': np.array([], dtype=np.int64),
                'atr': np.array([], dtype=np.float64),
                'atr_pips': np.array([], dtype=np.float64),
                'ema_separation': np.array([], dtype=np.float64),
                'ema_fast_slope': np.array([], dtype=np.float64),
                'ema_slow_slope': np.array([], dtype=np.float64),
                'cross_velocity': np.array([], dtype=np.float64),
                'bars_since_last_cross': np.array([], dtype=np.float64),
            }

        bars = np.empty(n, dtype=np.int64)
        prices = np.empty(n, dtype=np.float64)
        directions = np.empty(n, dtype=np.int64)
        hours = np.empty(n, dtype=np.int64)
        days_arr = np.empty(n, dtype=np.int64)
        ema_fast = np.empty(n, dtype=np.int64)
        ema_slow = np.empty(n, dtype=np.int64)
        atr = np.empty(n, dtype=np.float64)
        atr_pips = np.empty(n, dtype=np.float64)
        ema_separation = np.empty(n, dtype=np.float64)
        ema_fast_slope = np.empty(n, dtype=np.float64)
        ema_slow_slope = np.empty(n, dtype=np.float64)
        cross_velocity = np.empty(n, dtype=np.float64)
        bars_since_last_cross = np.empty(n, dtype=np.float64)

        for i, s in enumerate(signals):
            bars[i] = s.bar
            prices[i] = s.price
            directions[i] = s.direction
            hours[i] = s.hour
            days_arr[i] = s.day
            attr = s.attributes
            ema_fast[i] = attr['ema_fast_period']
            ema_slow[i] = attr['ema_slow_period']
            atr[i] = attr['atr']
            atr_pips[i] = attr['atr_pips']
            ema_separation[i] = attr['ema_separation']
            ema_fast_slope[i] = attr['ema_fast_slope']
            ema_slow_slope[i] = attr['ema_slow_slope']
            cross_velocity[i] = attr['cross_velocity']
            bars_since_last_cross[i] = attr['bars_since_last_cross']

        return {
            'bars': bars,
            'prices': prices,
            'directions': directions,
            'hours': hours,
            'days': days_arr,
            'ema_fast_period': ema_fast,
            'ema_slow_period': ema_slow,
            'atr': atr,
            'atr_pips': atr_pips,
            'ema_separation': ema_separation,
            'ema_fast_slope': ema_fast_slope,
            'ema_slow_slope': ema_slow_slope,
            'cross_velocity': cross_velocity,
            'bars_since_last_cross': bars_since_last_cross,
        }

    def _filter_vectorized(self, params: Dict[str, Any]) -> np.ndarray:
        """Vectorized signal filtering using numpy boolean masks."""
        va = self._vec_arrays
        n = len(va['bars'])
        if n == 0:
            return np.array([], dtype=np.bool_)

        mask = np.ones(n, dtype=np.bool_)

        # Must match selected EMA combo
        mask &= va['ema_fast_period'] == params.get('ema_fast_period', 13)
        mask &= va['ema_slow_period'] == params.get('ema_slow_period', 55)

        # ATR gate
        min_atr = params.get('min_atr_pips', 5)
        if min_atr > 0:
            mask &= va['atr_pips'] >= min_atr

        return mask

    def _compute_sl_tp_vectorized(
        self,
        mask: np.ndarray,
        params: Dict[str, Any],
        pip_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized SL/TP computation."""
        va = self._vec_arrays
        prices = va['prices'][mask]
        directions = va['directions'][mask]
        atr = va['atr'][mask]

        sl_mult = params.get('sl_atr_mult', 2.0)
        tp_mult = params.get('tp_atr_mult', 4.0)

        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr

        is_long = directions == 1
        sl_prices = np.where(is_long, prices - sl_dist, prices + sl_dist)
        tp_prices = np.where(is_long, prices + tp_dist, prices - tp_dist)

        return sl_prices, tp_prices

    def _get_management_arrays_vectorized(
        self,
        mask: np.ndarray,
        params: Dict[str, Any],
        sl_prices: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Vectorized management array computation."""
        va = self._vec_arrays
        n = int(np.sum(mask))
        if n == 0:
            return None

        atr_pips = va['atr_pips'][mask]

        mgmt = {
            'use_trailing': np.zeros(n, dtype=np.bool_),
            'trail_start_pips': np.zeros(n, dtype=np.float64),
            'trail_step_pips': np.zeros(n, dtype=np.float64),
            'use_breakeven': np.zeros(n, dtype=np.bool_),
            'be_trigger_pips': np.zeros(n, dtype=np.float64),
            'be_offset_pips': np.zeros(n, dtype=np.float64),
            'use_partial': np.zeros(n, dtype=np.bool_),
            'partial_pct': np.zeros(n, dtype=np.float64),
            'partial_target_pips': np.zeros(n, dtype=np.float64),
            'max_bars': np.full(n, params.get('max_hold_bars', 0), dtype=np.int64),
            'trail_mode': np.zeros(n, dtype=np.int64),
            'chandelier_atr_mult': np.full(n, 3.0, dtype=np.float64),
            'atr_pips': atr_pips.copy(),
            'stale_exit_bars': np.zeros(n, dtype=np.int64),
            # ML exit OFF — hardcoded to never fire
            'use_ml_exit': np.zeros(n, dtype=np.bool_),
            'ml_min_hold': np.zeros(n, dtype=np.int64),
            'ml_threshold': np.ones(n, dtype=np.float64),  # 1.0 = never fires
        }

        return mgmt


# Register in strategy registry
from strategies.rsi_fast import FAST_STRATEGIES
FAST_STRATEGIES['ema_cross_ml'] = EMACrossMLStrategy
