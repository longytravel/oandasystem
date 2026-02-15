"""
Fair Price MA Strategy - VERSION 1.0

Converted from: fairPriceMP v4.0 (MT5 Expert Advisor)
Original author: Unknown
Conversion: 2026-02-14

Core Logic (faithfully converted):
- Indicators: Fast EMA + Slow EMA on close price
- Trend detection: Fast EMA > Slow EMA = uptrend, vice versa
- Entry: BUY when price is trigger_pips below fast EMA in uptrend (mean reversion)
         SELL when price is trigger_pips above fast EMA in downtrend
- Grid: After initial entry, N orders evenly spaced across grid_range_pips
         All grid signals share same group_id (for group-aware MC shuffling)
- TP: At fast EMA value (mean reversion target) or R:R mode
- SL: Fixed pips or ATR multiple (approximates original equity stop)

Simplified/omitted vs original:
- Session-based correlation filter (N/A for single-pair pipeline)
- Per-pair equity stop replaced with fixed SL per trade
- Global equity stop not applicable in backtest
- Limit order fill modeling simplified to "price reaches level"

Parameter groups (4 groups, ~18 params):
1. signal (5): EMA periods, trigger_pips, grid_orders, grid_range_pips
2. risk (5): SL/TP mode and values
3. filters (3): Session time filter
4. management (5): Trailing, breakeven, max hold bars

Created: 2026-02-14
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.fast_strategy import FastStrategy, FastSignal, ParameterGroup, ParameterDef


class FairPriceMAStrategy(FastStrategy):
    """
    Fair Price MA Strategy (converted from fairPriceMP v4.0 EA).

    Mean-reversion grid strategy:
    - Fast/Slow EMA trend detection
    - Entry on pullback to trigger distance from fast EMA
    - Optional grid of additional entries across a price range
    - TP at fast EMA (mean reversion) or fixed R:R

    Supports grid trading via supports_grid() = True.
    The optimizer uses grid_backtest_numba for this strategy.
    """

    name = "Fair_Price_MA"
    version = "1.0"

    ATR_PERIOD = 14

    # EMA period combos to pre-compute
    FAST_PERIODS = [50, 100, 150, 200, 300]
    SLOW_PERIODS = [400, 600, 800, 1000]

    def supports_grid(self) -> bool:
        return True

    def get_parameter_groups(self) -> Optional[Dict[str, ParameterGroup]]:
        groups = {}

        # === Group 1: Signal (5 params) ===
        signal = ParameterGroup('signal', 'EMA trend + pullback entry + grid config')
        signal.add_param('fast_ema_period', [50, 100, 150, 200, 300], default=200)
        signal.add_param('slow_ema_period', [400, 600, 800, 1000], default=800)
        signal.add_param('trigger_pips', [30, 50, 75, 100, 150, 200], default=100)
        signal.add_param('grid_orders', [0, 3, 5, 10], default=0)  # 0 = no grid
        signal.add_param('grid_range_pips', [20, 30, 50, 75, 100], default=50)
        groups['signal'] = signal

        # === Group 2: Risk (5 params) ===
        risk = ParameterGroup('risk', 'Stop loss and take profit')
        risk.add_param('sl_mode', ['fixed', 'atr'], default='fixed')
        risk.add_param('sl_fixed_pips', [50, 75, 100, 150, 200], default=100)
        risk.add_param('sl_atr_mult', [2.0, 3.0, 4.0, 5.0], default=3.0)
        risk.add_param('tp_mode', ['ma', 'rr'], default='rr')
        risk.add_param('tp_rr_ratio', [1.5, 2.0, 3.0, 5.0], default=3.0)
        groups['risk'] = risk

        # === Group 3: Filters (3 params) ===
        filters = ParameterGroup('filters', 'Session time filter')
        filters.add_param('use_time_filter', [True, False], default=False)
        filters.add_param('trade_start_hour', [0, 2, 4, 6, 8], default=0)
        filters.add_param('trade_end_hour', [18, 20, 22, 23], default=23)
        groups['filters'] = filters

        # === Group 4: Management (5 params) ===
        management = ParameterGroup('management', 'Trade management features')
        management.add_param('use_trailing', [True, False], default=False)
        management.add_param('trail_start_pips', [15, 20, 30, 50], default=30)
        management.add_param('trail_step_pips', [8, 10, 15, 20], default=10)
        management.add_param('use_break_even', [True, False], default=False)
        management.add_param('max_hold_bars', [0, 50, 100, 200], default=0)
        groups['management'] = management

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
        Pre-compute ALL possible signals for all EMA combinations.

        For each bar where price pulls back trigger_pips from fast EMA:
        1. Create initial entry signal
        2. Scan forward to find bars where grid levels would fill
        3. Create grid-level signals at those bars (same group_id)
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

        # Pre-compute EMAs
        emas = {}
        for period in set(self.FAST_PERIODS + self.SLOW_PERIODS):
            emas[period] = self._calc_ema(closes, period)

        group_counter = 0

        # Generate signals for each EMA combo
        for fast_p in self.FAST_PERIODS:
            for slow_p in self.SLOW_PERIODS:
                if fast_p >= slow_p:
                    continue

                ema_fast = emas[fast_p]
                ema_slow = emas[slow_p]

                # Need enough bars for slow EMA to be valid
                start_bar = slow_p + 1

                # Track last signal bar to avoid duplicate entries too close together
                last_signal_bar = -100

                for i in range(start_bar, n):
                    if ema_fast[i] == 0 or ema_slow[i] == 0:
                        continue

                    # Minimum spacing: 5 bars between initial signals (same combo)
                    if i - last_signal_bar < 5:
                        continue

                    atr_val = atr[i]
                    if atr_val <= 0:
                        continue
                    atr_pips = atr_val / self._pip_size if self._pip_size > 0 else 0

                    fast_val = ema_fast[i]
                    slow_val = ema_slow[i]

                    # Determine trend
                    is_uptrend = fast_val > slow_val
                    is_downtrend = fast_val < slow_val

                    # Calculate distance from price to fast EMA in pips
                    distance = (fast_val - closes[i]) / self._pip_size if self._pip_size > 0 else 0

                    direction = 0

                    # BUY: price is below fast EMA in uptrend (pullback buy)
                    if is_uptrend and distance > 0:
                        # distance > 0 means fast_val > close (price below EMA)
                        direction = 1

                    # SELL: price is above fast EMA in downtrend (pullback sell)
                    elif is_downtrend and distance < 0:
                        # distance < 0 means fast_val < close (price above EMA)
                        direction = -1

                    if direction == 0:
                        continue

                    abs_distance_pips = abs(distance)

                    # Store initial signal with all trigger distances we might filter on
                    # We store signals for ALL trigger distances and filter per trial
                    group_id = group_counter
                    group_counter += 1
                    last_signal_bar = i

                    signals.append(FastSignal(
                        bar=i,
                        direction=direction,
                        price=closes[i],
                        hour=int(hours[i]),
                        day=int(days[i]),
                        attributes={
                            'fast_ema_period': fast_p,
                            'slow_ema_period': slow_p,
                            'atr': atr_val,
                            'atr_pips': atr_pips,
                            'distance_pips': abs_distance_pips,
                            'fast_ema_value': fast_val,
                            'group_id': group_id,
                            'grid_level': 0,  # 0 = initial entry
                        }
                    ))

                    # Generate grid expansion signals
                    # For each possible grid configuration, scan forward for fills
                    for n_orders in [3, 5, 10]:
                        for grid_range in [20, 30, 50, 75, 100]:
                            step = grid_range / n_orders if n_orders > 0 else 0
                            grid_group = group_counter
                            group_counter += 1

                            for level in range(1, n_orders + 1):
                                grid_offset_pips = step * level
                                grid_offset = grid_offset_pips * self._pip_size

                                if direction == 1:
                                    grid_price = closes[i] - grid_offset
                                else:
                                    grid_price = closes[i] + grid_offset

                                # Scan forward to find fill bar
                                fill_bar = -1
                                for j in range(i + 1, min(i + 500, n)):
                                    if direction == 1:
                                        if lows[j] <= grid_price:
                                            fill_bar = j
                                            break
                                    else:
                                        if highs[j] >= grid_price:
                                            fill_bar = j
                                            break

                                if fill_bar > 0:
                                    signals.append(FastSignal(
                                        bar=fill_bar,
                                        direction=direction,
                                        price=grid_price,
                                        hour=int(hours[fill_bar]),
                                        day=int(days[fill_bar]),
                                        attributes={
                                            'fast_ema_period': fast_p,
                                            'slow_ema_period': slow_p,
                                            'atr': atr[fill_bar],
                                            'atr_pips': atr[fill_bar] / self._pip_size if self._pip_size > 0 else 0,
                                            'distance_pips': abs_distance_pips,
                                            'fast_ema_value': ema_fast[fill_bar],
                                            'group_id': grid_group,
                                            'grid_level': level,
                                            'grid_orders': n_orders,
                                            'grid_range_pips': grid_range,
                                            'parent_bar': i,
                                        }
                                    ))

        # Sort by bar
        signals.sort(key=lambda x: (x.bar, x.attributes.get('grid_level', 0)))
        return signals

    def filter_signals(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any]
    ) -> List[FastSignal]:
        """Filter signals by EMA combo, trigger distance, grid config, and time."""
        fast_p = params.get('fast_ema_period', 200)
        slow_p = params.get('slow_ema_period', 800)
        trigger = params.get('trigger_pips', 100)
        grid_orders = params.get('grid_orders', 0)
        grid_range = params.get('grid_range_pips', 50)
        use_time = params.get('use_time_filter', False)
        start_hour = params.get('trade_start_hour', 0)
        end_hour = params.get('trade_end_hour', 23)

        result = []
        seen_groups = set()  # Track which grid groups we've included

        for s in signals:
            attr = s.attributes

            # Must match EMA combo
            if attr['fast_ema_period'] != fast_p or attr['slow_ema_period'] != slow_p:
                continue

            grid_level = attr.get('grid_level', 0)

            if grid_level == 0:
                # Initial entry: check trigger distance
                if attr['distance_pips'] < trigger:
                    continue

                # Time filter (only check initial signal)
                if use_time and not (start_hour <= s.hour <= end_hour):
                    continue

                result.append(s)

            else:
                # Grid signal: only include if grid_orders > 0 and matches config
                if grid_orders <= 0:
                    continue
                if attr.get('grid_orders') != grid_orders:
                    continue
                if attr.get('grid_range_pips') != grid_range:
                    continue
                if attr.get('grid_level', 0) > grid_orders:
                    continue

                # Check parent initial signal's distance (parent must have triggered)
                if attr['distance_pips'] < trigger:
                    continue

                result.append(s)

        return result

    def compute_sl_tp(
        self,
        signal: FastSignal,
        params: Dict[str, Any],
        pip_size: float
    ) -> Tuple[float, float]:
        """Compute SL/TP for a signal."""
        attr = signal.attributes
        atr_val = attr['atr']

        # SL
        sl_mode = params.get('sl_mode', 'fixed')
        if sl_mode == 'atr':
            sl_dist = params.get('sl_atr_mult', 3.0) * atr_val
        else:
            sl_dist = params.get('sl_fixed_pips', 100) * pip_size

        # TP
        tp_mode = params.get('tp_mode', 'rr')
        if tp_mode == 'ma':
            # TP at fast EMA value (mean reversion target)
            fast_val = attr.get('fast_ema_value', signal.price)
            tp_dist = abs(fast_val - signal.price)
            # Ensure minimum TP distance (at least 0.5x SL)
            if tp_dist < sl_dist * 0.5:
                tp_dist = sl_dist * 0.5
        else:
            # R:R mode
            tp_dist = sl_dist * params.get('tp_rr_ratio', 3.0)

        if signal.direction == 1:
            sl = signal.price - sl_dist
            tp = signal.price + tp_dist
        else:
            sl = signal.price + sl_dist
            tp = signal.price - tp_dist

        return sl, tp

    def get_management_arrays(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Return management arrays for non-grid mode (full_backtest_numba)."""
        n = len(signals)
        if n == 0:
            return None

        use_trail = params.get('use_trailing', False)
        use_be = params.get('use_break_even', False)

        mgmt = {
            'use_trailing': np.full(n, use_trail, dtype=np.bool_),
            'trail_start_pips': np.full(n, params.get('trail_start_pips', 30), dtype=np.float64),
            'trail_step_pips': np.full(n, params.get('trail_step_pips', 10), dtype=np.float64),
            'use_breakeven': np.full(n, use_be, dtype=np.bool_),
            'be_trigger_pips': np.full(n, 20.0, dtype=np.float64),
            'be_offset_pips': np.full(n, 5.0, dtype=np.float64),
            'use_partial': np.zeros(n, dtype=np.bool_),
            'partial_pct': np.zeros(n, dtype=np.float64),
            'partial_target_pips': np.zeros(n, dtype=np.float64),
            'max_bars': np.full(n, params.get('max_hold_bars', 0), dtype=np.int64),
            'trail_mode': np.zeros(n, dtype=np.int64),
            'chandelier_atr_mult': np.full(n, 3.0, dtype=np.float64),
            'atr_pips': np.array([s.attributes.get('atr_pips', 35.0) for s in signals], dtype=np.float64),
            'stale_exit_bars': np.zeros(n, dtype=np.int64),
            'use_ml_exit': np.zeros(n, dtype=np.bool_),
            'ml_min_hold': np.zeros(n, dtype=np.int64),
            'ml_threshold': np.ones(n, dtype=np.float64),
        }

        return mgmt

    def get_grid_arrays(
        self,
        params: Dict[str, Any],
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get signal arrays with group_ids for grid backtesting.

        Uses vectorized path when available (10-50x faster on 1.5M signals).
        """
        # Fast vectorized path
        if self._vec_arrays is not None and hasattr(self, '_filter_vectorized'):
            mask = self._filter_vectorized(params)
            if not np.any(mask):
                return None

            va = self._vec_arrays
            entry_bars = va['bars'][mask]
            entry_prices = va['prices'][mask]
            directions = va['directions'][mask]
            group_ids = va['group_id'][mask]
            sl_prices, tp_prices = self._compute_sl_tp_vectorized(mask, params, self._pip_size)

            return entry_bars, entry_prices, directions, sl_prices, tp_prices, group_ids

        # Fallback: Python loop path
        signals = self.filter_signals(self._precomputed_signals, params)

        if not signals:
            return None

        n = len(signals)
        entry_bars = np.array([s.bar for s in signals], dtype=np.int64)
        entry_prices = np.array([s.price for s in signals], dtype=np.float64)
        directions = np.array([s.direction for s in signals], dtype=np.int64)
        group_ids = np.array([s.attributes.get('group_id', 0) for s in signals], dtype=np.int64)

        sl_prices = np.zeros(n, dtype=np.float64)
        tp_prices = np.zeros(n, dtype=np.float64)

        for i, signal in enumerate(signals):
            sl, tp = self.compute_sl_tp(signal, params, self._pip_size)
            sl_prices[i] = sl
            tp_prices[i] = tp

        return entry_bars, entry_prices, directions, sl_prices, tp_prices, group_ids

    # ===================================================================
    # VECTORIZED PATH (for fast per-trial filtering)
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
                'fast_ema_period': np.array([], dtype=np.int64),
                'slow_ema_period': np.array([], dtype=np.int64),
                'atr': np.array([], dtype=np.float64),
                'atr_pips': np.array([], dtype=np.float64),
                'distance_pips': np.array([], dtype=np.float64),
                'fast_ema_value': np.array([], dtype=np.float64),
                'group_id': np.array([], dtype=np.int64),
                'grid_level': np.array([], dtype=np.int64),
                'grid_orders': np.array([], dtype=np.int64),
                'grid_range_pips': np.array([], dtype=np.int64),
            }

        bars = np.empty(n, dtype=np.int64)
        prices = np.empty(n, dtype=np.float64)
        directions = np.empty(n, dtype=np.int64)
        hours = np.empty(n, dtype=np.int64)
        days = np.empty(n, dtype=np.int64)
        fast_ema = np.empty(n, dtype=np.int64)
        slow_ema = np.empty(n, dtype=np.int64)
        atr_arr = np.empty(n, dtype=np.float64)
        atr_pips = np.empty(n, dtype=np.float64)
        distance = np.empty(n, dtype=np.float64)
        fast_val = np.empty(n, dtype=np.float64)
        group_ids = np.empty(n, dtype=np.int64)
        grid_level = np.empty(n, dtype=np.int64)
        grid_orders_arr = np.empty(n, dtype=np.int64)
        grid_range_arr = np.empty(n, dtype=np.int64)

        for i, s in enumerate(signals):
            bars[i] = s.bar
            prices[i] = s.price
            directions[i] = s.direction
            hours[i] = s.hour
            days[i] = s.day
            attr = s.attributes
            fast_ema[i] = attr['fast_ema_period']
            slow_ema[i] = attr['slow_ema_period']
            atr_arr[i] = attr['atr']
            atr_pips[i] = attr['atr_pips']
            distance[i] = attr['distance_pips']
            fast_val[i] = attr.get('fast_ema_value', s.price)
            group_ids[i] = attr.get('group_id', 0)
            grid_level[i] = attr.get('grid_level', 0)
            grid_orders_arr[i] = attr.get('grid_orders', 0)
            grid_range_arr[i] = attr.get('grid_range_pips', 0)

        return {
            'bars': bars,
            'prices': prices,
            'directions': directions,
            'hours': hours,
            'days': days,
            'fast_ema_period': fast_ema,
            'slow_ema_period': slow_ema,
            'atr': atr_arr,
            'atr_pips': atr_pips,
            'distance_pips': distance,
            'fast_ema_value': fast_val,
            'group_id': group_ids,
            'grid_level': grid_level,
            'grid_orders': grid_orders_arr,
            'grid_range_pips': grid_range_arr,
        }

    def _filter_vectorized(self, params: Dict[str, Any]) -> np.ndarray:
        """Vectorized filter: return boolean mask over _vec_arrays."""
        va = self._vec_arrays
        if va is None or len(va['bars']) == 0:
            return np.array([], dtype=np.bool_)

        fast_p = params.get('fast_ema_period', 200)
        slow_p = params.get('slow_ema_period', 800)
        trigger = params.get('trigger_pips', 100)
        grid_orders = params.get('grid_orders', 0)
        grid_range = params.get('grid_range_pips', 50)
        use_time = params.get('use_time_filter', False)
        start_hour = params.get('trade_start_hour', 0)
        end_hour = params.get('trade_end_hour', 23)

        # EMA combo match
        mask = (va['fast_ema_period'] == fast_p) & (va['slow_ema_period'] == slow_p)

        # Trigger distance (applies to all signals in a group)
        mask &= va['distance_pips'] >= trigger

        # Grid filter
        is_initial = va['grid_level'] == 0
        if grid_orders <= 0:
            # No grid: only keep initial entries
            mask &= is_initial
        else:
            # Keep initial entries + matching grid config
            grid_match = (
                (va['grid_orders'] == grid_orders) &
                (va['grid_range_pips'] == grid_range) &
                (va['grid_level'] <= grid_orders)
            )
            mask &= is_initial | grid_match

        # Time filter (only on initial signals; grid inherits from parent)
        if use_time:
            time_ok = (va['hours'] >= start_hour) & (va['hours'] <= end_hour)
            # Apply time filter to initial entries, pass grid signals through
            mask &= time_ok | (~is_initial)

        return mask

    def _compute_sl_tp_vectorized(
        self,
        mask: np.ndarray,
        params: Dict[str, Any],
        pip_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized SL/TP computation."""
        va = self._vec_arrays
        n = int(np.sum(mask))

        prices = va['prices'][mask]
        directions = va['directions'][mask]
        atr_arr = va['atr'][mask]
        fast_vals = va['fast_ema_value'][mask]

        sl_mode = params.get('sl_mode', 'fixed')
        tp_mode = params.get('tp_mode', 'rr')

        # SL distance
        if sl_mode == 'atr':
            sl_dist = params.get('sl_atr_mult', 3.0) * atr_arr
        else:
            sl_dist = np.full(n, params.get('sl_fixed_pips', 100) * pip_size, dtype=np.float64)

        # TP distance
        if tp_mode == 'ma':
            tp_dist = np.abs(fast_vals - prices)
            # Minimum: 0.5x SL
            min_tp = sl_dist * 0.5
            tp_dist = np.maximum(tp_dist, min_tp)
        else:
            tp_dist = sl_dist * params.get('tp_rr_ratio', 3.0)

        sl_prices = np.where(directions == 1, prices - sl_dist, prices + sl_dist)
        tp_prices = np.where(directions == 1, prices + tp_dist, prices - tp_dist)

        return sl_prices, tp_prices
