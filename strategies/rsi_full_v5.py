"""
RSI Hidden Divergence Strategy - VERSION 5 (Chandelier Exit + Stale Exit)

Changes from V4 targeting the catastrophic exit problem:
V4 analysis: 91% of wins are 0-5 pips (265/291). BE trigger at 0.3x ATR (~4 pips)
locks entry+5pip, price hits SL on next bar. TP target (6x ATR ~99-144 pips) unreachable.
Optimizer turned OFF chain_be_to_trail because tiny BE trigger = tight trail.

V3 comparison: V3 disabled BE entirely — avg win was $52.70 (25x more per trade).

V5 fixes:
1. Chandelier Exit: ATR-adaptive trailing from highest-high/lowest-low, active from bar 1.
   Replaces fixed-pip trail which needed a start threshold.
2. Wider BE triggers: minimum 0.8x ATR (~25 pips) prevents premature exit.
3. Stale trade exit: Exit if no significant move (< 0.5R) in N bars.
4. 1R partial close: Partial at SL-multiple distance (e.g. 1R), independent of BE.
5. Removed chain_be_to_trail (redundant with chandelier) and partial_at_be (replaced by 1R).

Created: 2026-02-07
Based on analysis of: GBP_USD_H1 V4 results (81.5/100 GREEN, 91% of wins 0-5 pips)
Goal: Higher avg win (>10 pips), fewer 0-5 pip exits, better per-trade quality.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.fast_strategy import FastStrategy, FastSignal, ParameterGroup, ParameterDef


class RSIDivergenceFullFastV5(FastStrategy):
    """
    RSI Hidden Divergence - VERSION 5 (Chandelier Exit + Stale Exit).

    Inherits V3/V4's stability-hardened signal generation:
    1. Multi-period RSI consensus (eliminates rsi_period fragility)
    2. Adaptive swing detection (eliminates swing_strength fragility)
    3. Dual-MA trend filter (eliminates trend_ma_period fragility)
    4. ATR-band SL (eliminates sl_atr_mult fragility)
    5. ATR-scaled breakeven (wider triggers to prevent premature exit)

    V5 additions:
    6. Chandelier Exit (ATR-adaptive trailing from highest high/lowest low)
    7. Stale trade exit (exit if no move in N bars)
    8. 1R partial close (SL-multiple based, independent of BE)

    Parameter groups for staged optimization:
    1. signal (6 params): RSI divergence settings
    2. filters (5 params): Slope, trend, spread
    3. risk (5 params): SL/TP modes
    4. management (12 params): Trailing, breakeven, partial closes + V5 additions
    5. time (6 params): Trading hours, days

    Total: ~34 parameters
    """

    name = "RSI_Divergence_v5"
    version = "5.0"

    ATR_PERIOD = 14

    # V3: Fixed consensus periods - NOT optimizable
    RSI_CONSENSUS_PERIODS = [7, 14, 21]

    # V3: Adaptive swing - try all strengths, take union
    SWING_STRENGTHS_UNION = [3, 5, 7]

    def get_parameter_groups(self) -> Optional[Dict[str, ParameterGroup]]:
        groups = {}

        # === Group 1: Signal Generation (6 params) - same as V3/V4 ===
        signal = ParameterGroup('signal', 'Signal generation parameters')
        signal.add_param('rsi_overbought', [70, 75, 80], default=70)
        signal.add_param('rsi_oversold', [20, 25, 30], default=30)
        signal.add_param('min_rsi_diff', [2.0, 3.0, 5.0, 8.0], default=3.0)
        signal.add_param('min_bars_between', [5, 10, 15, 20], default=5)
        signal.add_param('max_bars_between', [40, 60, 80, 100], default=80)
        signal.add_param('require_pullback', [True, False], default=False)
        groups['signal'] = signal

        # === Group 2: Entry Filters (5 params) - same as V3/V4 ===
        filters = ParameterGroup('filters', 'Entry filter parameters')
        filters.add_param('use_slope_filter', [True, False], default=False)
        filters.add_param('min_price_slope', [5.0, 15.0, 25.0], default=5.0)
        filters.add_param('max_price_slope', [50.0, 65.0, 80.0], default=80.0)
        filters.add_param('use_trend_filter', [True, False], default=False)
        filters.add_param('max_spread_pips', [2.0, 3.0, 5.0], default=3.0)
        groups['filters'] = filters

        # === Group 3: Risk (SL/TP) (5 params) - same as V4 ===
        risk = ParameterGroup('risk', 'Stop loss and take profit')
        risk.add_param('sl_mode', ['fixed', 'atr'], default='fixed')
        risk.add_param('sl_fixed_pips', [25, 35, 50, 75], default=50)
        risk.add_param('sl_atr_pct', [80, 100, 150, 200, 300], default=300)
        risk.add_param('tp_mode', ['rr', 'atr', 'fixed'], default='rr')
        risk.add_param('tp_rr_ratio', [1.0, 1.5, 2.0, 3.0, 5.0], default=2.0)
        risk.add_param('tp_atr_mult', [1.5, 2.0, 3.0, 4.0, 6.0], default=3.0)
        risk.add_param('tp_fixed_pips', [20, 30, 50, 75], default=30)
        groups['risk'] = risk

        # === Group 4: Trade Management (12 params) - V5: chandelier + stale + 1R partial ===
        management = ParameterGroup('management', 'Trailing, breakeven, partial closes')
        management.add_param('use_trailing', [True, False], default=False)
        # V5: trail_mode - fixed (V4 behavior) or chandelier (ATR-adaptive)
        management.add_param('trail_mode', ['fixed', 'chandelier'], default='chandelier')
        # V5: Chandelier ATR multiplier (distance from highest high/lowest low)
        management.add_param('chandelier_atr_mult', [2.0, 2.5, 3.0, 3.5], default=3.0)
        management.add_param('trail_start_pips', [15, 20, 30, 50], default=20)
        management.add_param('trail_step_pips', [5, 8, 10, 15], default=10)
        management.add_param('use_break_even', [True, False], default=False)
        # V5: Wider BE triggers - minimum 0.8x ATR (~25 pips) prevents premature exit
        management.add_param('be_atr_mult', [0.8, 1.0, 1.2, 1.5], default=1.0)
        # V5: Wider BE offset - more room to breathe
        management.add_param('be_offset_pips', [2, 5, 10], default=5)
        management.add_param('use_partial_close', [True, False], default=False)
        management.add_param('partial_close_pct', [0.3, 0.5, 0.7], default=0.3)
        # V5: Partial target mode - SL-multiple based (1R) or fixed pip
        management.add_param('partial_target_mode', ['sl_multiple', 'fixed'], default='sl_multiple')
        # V5: Partial at 1R (1x SL distance), adjustable
        management.add_param('partial_target_sl_mult', [0.8, 1.0, 1.5, 2.0], default=1.0)
        # V5: Stale trade exit (0=disabled)
        management.add_param('stale_exit_bars', [0, 30, 50, 80], default=0)
        groups['management'] = management

        # === Group 5: Time Filters (6 params) - same as V3/V4 ===
        time_group = ParameterGroup('time', 'Trading time filters')
        time_group.add_param('use_time_filter', [True, False], default=False)
        time_group.add_param('trade_start_hour', [0, 2, 4, 6, 8], default=4)
        time_group.add_param('trade_end_hour', [18, 20, 22, 23], default=22)
        time_group.add_param('trade_monday', [True, False], default=False)
        time_group.add_param('trade_friday', [True, False], default=False)
        time_group.add_param('friday_close_hour', [18, 20], default=18)
        groups['time'] = time_group

        return groups

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        groups = self.get_parameter_groups()
        space = {}
        for group in groups.values():
            space.update(group.get_param_space())
        return space

    def get_management_arrays(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any],
    ) -> Optional[Dict[str, np.ndarray]]:
        n = len(signals)
        if n == 0:
            return None

        # Trailing stop arrays
        use_trailing = np.full(n, params.get('use_trailing', False), dtype=np.bool_)
        trail_start = np.full(n, params.get('trail_start_pips', 20), dtype=np.float64)
        trail_step = np.full(n, params.get('trail_step_pips', 10), dtype=np.float64)

        # V5: Trail mode (0=fixed, 1=chandelier)
        trail_mode_str = params.get('trail_mode', 'chandelier')
        trail_mode_val = 1 if trail_mode_str == 'chandelier' else 0
        trail_mode = np.full(n, trail_mode_val, dtype=np.int64)

        # V5: Chandelier ATR multiplier
        chandelier_atr_mult = np.full(n, params.get('chandelier_atr_mult', 3.0), dtype=np.float64)

        # V5: ATR pips per signal (from signal attributes)
        atr_pips_arr = np.zeros(n, dtype=np.float64)
        for i, sig in enumerate(signals):
            atr_pips_arr[i] = sig.attributes.get('atr_pips', 35.0)

        # Breakeven arrays - V5: wider ATR-scaled trigger
        use_be = np.full(n, params.get('use_break_even', False), dtype=np.bool_)
        be_atr_mult = params.get('be_atr_mult', 1.0)
        be_trigger = np.zeros(n, dtype=np.float64)
        for i, sig in enumerate(signals):
            atr_pips = sig.attributes.get('atr_pips', 35.0)
            be_trigger[i] = atr_pips * be_atr_mult
        be_offset = np.full(n, params.get('be_offset_pips', 5), dtype=np.float64)

        # Partial close arrays
        use_partial = np.full(n, params.get('use_partial_close', False), dtype=np.bool_)
        partial_pct = np.full(n, params.get('partial_close_pct', 0.3), dtype=np.float64)

        # V5: Partial target - SL-multiple based or fixed
        partial_target_mode = params.get('partial_target_mode', 'sl_multiple')
        partial_target = np.zeros(n, dtype=np.float64)
        for i, sig in enumerate(signals):
            if partial_target_mode == 'sl_multiple':
                # V5: Partial at N x SL distance (e.g. 1R)
                sl_pips = self._get_sl_pips(sig, params)
                partial_target[i] = sl_pips * params.get('partial_target_sl_mult', 1.0)
            else:
                # Fixed: 50% of TP distance
                sl_pips = self._get_sl_pips(sig, params)
                atr_pips = sig.attributes.get('atr_pips', 35.0)
                tp_pips = self._get_tp_pips(sl_pips, params, atr_pips)
                partial_target[i] = tp_pips * 0.5

        max_bars = np.zeros(n, dtype=np.int64)

        # V5: Stale exit bars
        stale_exit_bars = np.full(n, params.get('stale_exit_bars', 0), dtype=np.int64)

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
            # V5 new arrays
            'trail_mode': trail_mode,
            'chandelier_atr_mult': chandelier_atr_mult,
            'atr_pips': atr_pips_arr,
            'stale_exit_bars': stale_exit_bars,
        }

    def _get_sl_pips(self, signal: FastSignal, params: Dict[str, Any]) -> float:
        sl_mode = params.get('sl_mode', 'atr')
        attr = signal.attributes

        if sl_mode == 'fixed':
            return params.get('sl_fixed_pips', 50)
        elif sl_mode == 'atr':
            atr_pips = attr.get('atr_pips', attr.get('atr', 0.0035) / self._pip_size)
            sl_atr_pct = params.get('sl_atr_pct', 100) / 100.0
            return atr_pips * sl_atr_pct
        return params.get('sl_fixed_pips', 50)

    def _get_tp_pips(self, sl_pips: float, params: Dict[str, Any], atr_pips: float = 35.0) -> float:
        tp_mode = params.get('tp_mode', 'fixed')

        if tp_mode == 'rr':
            return sl_pips * params.get('tp_rr_ratio', 2.0)
        elif tp_mode == 'atr':
            return atr_pips * params.get('tp_atr_mult', 3.0)
        elif tp_mode == 'fixed':
            return params.get('tp_fixed_pips', 30)
        return sl_pips * params.get('tp_rr_ratio', 2.0)

    def _calc_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
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
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return atr

    def _calc_ma(self, close: np.ndarray, period: int) -> np.ndarray:
        ma = np.zeros_like(close)
        for i in range(period - 1, len(close)):
            ma[i] = np.mean(close[i-period+1:i+1])
        return ma

    def _find_swings(self, highs: np.ndarray, lows: np.ndarray, strength: int) -> Tuple[List, List]:
        """Find swing points with soft threshold (1 bar tolerance)."""
        swing_highs = []
        swing_lows = []
        n = len(highs)
        max_breaks = max(1, strength // 4)

        for i in range(strength, n - strength):
            # Swing high
            breaks = 0
            is_high = True
            for j in range(1, strength + 1):
                if highs[i - j] >= highs[i] or highs[i + j] >= highs[i]:
                    breaks += 1
                    if breaks > max_breaks:
                        is_high = False
                        break
            if is_high:
                swing_highs.append((i, highs[i]))

            # Swing low
            breaks = 0
            is_low = True
            for j in range(1, strength + 1):
                if lows[i - j] <= lows[i] or lows[i + j] <= lows[i]:
                    breaks += 1
                    if breaks > max_breaks:
                        is_low = False
                        break
            if is_low:
                swing_lows.append((i, lows[i]))

        return swing_highs, swing_lows

    def precompute(self, df: pd.DataFrame) -> List[FastSignal]:
        """
        V3/V4/V5: Pre-compute signals using multi-period RSI consensus
        and adaptive swing detection. (Identical to V3/V4)
        """
        signals = []
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        hours = df.index.hour.values
        days = df.index.dayofweek.values
        n = len(closes)

        # Pre-compute ATR
        atr = self._calc_atr(highs, lows, closes, self.ATR_PERIOD)

        # Pre-compute MAs for dual-MA trend filter
        ma50 = self._calc_ma(closes, 50)
        ma200 = self._calc_ma(closes, 200)

        # Multi-period RSI consensus
        rsi_arrays = []
        for period in self.RSI_CONSENSUS_PERIODS:
            rsi_arrays.append(self._calc_rsi(closes, period))

        # Average RSI across all periods
        rsi_consensus = np.zeros(n, dtype=np.float64)
        for rsi_arr in rsi_arrays:
            rsi_consensus += rsi_arr
        rsi_consensus /= len(rsi_arrays)

        # Adaptive swing detection - union of multiple strengths
        all_swing_lows = {}
        all_swing_highs = {}

        for strength in self.SWING_STRENGTHS_UNION:
            sh, sl = self._find_swings(highs, lows, strength)
            for bar_idx, price in sh:
                if bar_idx not in all_swing_highs:
                    all_swing_highs[bar_idx] = (bar_idx, price)
            for bar_idx, price in sl:
                if bar_idx not in all_swing_lows:
                    all_swing_lows[bar_idx] = (bar_idx, price)

        # Sort by bar index
        swing_highs = sorted(all_swing_highs.values(), key=lambda x: x[0])
        swing_lows = sorted(all_swing_lows.values(), key=lambda x: x[0])

        max_strength = max(self.SWING_STRENGTHS_UNION)

        # Generate bullish signals (from swing lows)
        for i in range(1, len(swing_lows)):
            prev_bar, prev_price = swing_lows[i - 1]
            curr_bar, curr_price = swing_lows[i]
            bars_between = curr_bar - prev_bar

            if bars_between < 2 or bars_between > 120:
                continue

            prev_rsi = rsi_consensus[prev_bar]
            curr_rsi = rsi_consensus[curr_bar]

            # Hidden bullish: Price HL, RSI LL
            if curr_price > prev_price and curr_rsi < prev_rsi:
                rsi_diff = abs(prev_rsi - curr_rsi)
                slope = (curr_price - prev_price) / bars_between * 10000
                slope_angle = abs(np.arctan(slope) * 180 / np.pi)

                signal_bar = curr_bar + max_strength + 1
                if signal_bar < n:
                    hour = hours[signal_bar]
                    day = days[signal_bar]

                    signals.append(FastSignal(
                        bar=signal_bar,
                        direction=1,
                        price=closes[signal_bar],
                        hour=hour,
                        day=day,
                        attributes={
                            'rsi_value': curr_rsi,
                            'prev_rsi': prev_rsi,
                            'rsi_diff': rsi_diff,
                            'bars_between': bars_between,
                            'swing_price': curr_price,
                            'price_slope': slope_angle,
                            'atr': atr[signal_bar],
                            'atr_pips': atr[signal_bar] / self._pip_size,
                            'ma50': ma50[signal_bar],
                            'ma200': ma200[signal_bar],
                            'is_london': 8 <= hour < 16,
                            'is_ny': 13 <= hour < 21,
                            'is_asian': hour < 8,
                            'is_overlap': 13 <= hour < 16,
                        }
                    ))

        # Generate bearish signals (from swing highs)
        for i in range(1, len(swing_highs)):
            prev_bar, prev_price = swing_highs[i - 1]
            curr_bar, curr_price = swing_highs[i]
            bars_between = curr_bar - prev_bar

            if bars_between < 2 or bars_between > 120:
                continue

            prev_rsi = rsi_consensus[prev_bar]
            curr_rsi = rsi_consensus[curr_bar]

            # Hidden bearish: Price LH, RSI HH
            if curr_price < prev_price and curr_rsi > prev_rsi:
                rsi_diff = abs(curr_rsi - prev_rsi)
                slope = abs(prev_price - curr_price) / bars_between * 10000
                slope_angle = abs(np.arctan(slope) * 180 / np.pi)

                signal_bar = curr_bar + max_strength + 1
                if signal_bar < n:
                    hour = hours[signal_bar]
                    day = days[signal_bar]

                    signals.append(FastSignal(
                        bar=signal_bar,
                        direction=-1,
                        price=closes[signal_bar],
                        hour=hour,
                        day=day,
                        attributes={
                            'rsi_value': curr_rsi,
                            'prev_rsi': prev_rsi,
                            'rsi_diff': rsi_diff,
                            'bars_between': bars_between,
                            'swing_price': curr_price,
                            'price_slope': slope_angle,
                            'atr': atr[signal_bar],
                            'atr_pips': atr[signal_bar] / self._pip_size,
                            'ma50': ma50[signal_bar],
                            'ma200': ma200[signal_bar],
                            'is_london': 8 <= hour < 16,
                            'is_ny': 13 <= hour < 21,
                            'is_asian': hour < 8,
                            'is_overlap': 13 <= hour < 16,
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

        unique.sort(key=lambda x: x.bar)
        return unique

    def filter_signals(
        self,
        signals: List[FastSignal],
        params: Dict[str, Any]
    ) -> List[FastSignal]:
        """Filter signals. Identical to V3/V4."""
        result = []

        rsi_overbought = params.get('rsi_overbought', 70)
        rsi_oversold = params.get('rsi_oversold', 30)

        for s in signals:
            attr = s.attributes

            if attr['rsi_diff'] < params['min_rsi_diff']:
                continue
            if attr['bars_between'] < params['min_bars_between']:
                continue
            if attr['bars_between'] > params['max_bars_between']:
                continue

            if params.get('require_pullback', False):
                prev_rsi = attr.get('prev_rsi', 50)
                curr_rsi = attr.get('rsi_value', 50)
                if s.direction == 1:
                    if prev_rsi >= rsi_oversold and curr_rsi >= rsi_oversold:
                        continue
                else:
                    if prev_rsi <= rsi_overbought and curr_rsi <= rsi_overbought:
                        continue

            if params.get('use_slope_filter', True):
                if attr['price_slope'] < params['min_price_slope']:
                    continue
                if attr['price_slope'] > params['max_price_slope']:
                    continue

            if params.get('use_trend_filter', True):
                ma50 = attr.get('ma50', 0)
                ma200 = attr.get('ma200', 0)
                if ma50 > 0 and ma200 > 0:
                    if s.direction == 1:
                        if s.price < ma50 or s.price < ma200:
                            continue
                    else:
                        if s.price > ma50 or s.price > ma200:
                            continue

            if params.get('use_time_filter', False):
                if s.day == 0 and not params.get('trade_monday', False):
                    continue
                if s.day == 4 and not params.get('trade_friday', False):
                    continue
                if s.day == 4 and s.hour >= params.get('friday_close_hour', 18):
                    continue
                if s.day == 6:
                    continue

                start = params.get('trade_start_hour', 4)
                end = params.get('trade_end_hour', 22)
                if start < end:
                    if s.hour < start or s.hour >= end:
                        continue
                else:
                    if s.hour < start and s.hour >= end:
                        continue

            result.append(s)

        return result

    def compute_sl_tp(
        self,
        signal: FastSignal,
        params: Dict[str, Any],
        pip_size: float
    ) -> Tuple[float, float]:
        attr = signal.attributes

        # === STOP LOSS ===
        sl_mode = params.get('sl_mode', 'atr')

        if sl_mode == 'fixed':
            sl_pips = params.get('sl_fixed_pips', 50)
        elif sl_mode == 'atr':
            atr_pips = attr['atr'] / pip_size
            sl_atr_pct = params.get('sl_atr_pct', 100) / 100.0
            sl_pips = atr_pips * sl_atr_pct
        else:
            sl_pips = params.get('sl_fixed_pips', 50)

        sl_pips = max(sl_pips, 10)

        # === TAKE PROFIT ===
        tp_mode = params.get('tp_mode', 'fixed')

        if tp_mode == 'rr':
            tp_pips = sl_pips * params.get('tp_rr_ratio', 2.0)
        elif tp_mode == 'atr':
            atr_pips = attr['atr'] / pip_size
            tp_pips = atr_pips * params.get('tp_atr_mult', 3.0)
        elif tp_mode == 'fixed':
            tp_pips = params.get('tp_fixed_pips', 30)
        else:
            tp_pips = sl_pips * params.get('tp_rr_ratio', 2.0)

        if signal.direction == 1:
            sl = signal.price - sl_pips * pip_size
            tp = signal.price + tp_pips * pip_size
        else:
            sl = signal.price + sl_pips * pip_size
            tp = signal.price - tp_pips * pip_size

        return sl, tp

    # ===================================================================
    # VECTORIZED PATH - 10-50x faster per trial (OPT-1, OPT-2)
    # ===================================================================

    def _build_signal_arrays(self, signals: list) -> Dict[str, np.ndarray]:
        """Convert list of FastSignal into parallel numpy arrays for vectorized filtering."""
        n = len(signals)
        if n == 0:
            return {
                'bars': np.array([], dtype=np.int64),
                'prices': np.array([], dtype=np.float64),
                'directions': np.array([], dtype=np.int64),
                'hours': np.array([], dtype=np.int64),
                'days': np.array([], dtype=np.int64),
                'rsi_diffs': np.array([], dtype=np.float64),
                'bars_between': np.array([], dtype=np.float64),
                'prev_rsi': np.array([], dtype=np.float64),
                'rsi_values': np.array([], dtype=np.float64),
                'price_slopes': np.array([], dtype=np.float64),
                'atr': np.array([], dtype=np.float64),
                'atr_pips': np.array([], dtype=np.float64),
                'ma50': np.array([], dtype=np.float64),
                'ma200': np.array([], dtype=np.float64),
            }

        bars = np.empty(n, dtype=np.int64)
        prices = np.empty(n, dtype=np.float64)
        directions = np.empty(n, dtype=np.int64)
        hours = np.empty(n, dtype=np.int64)
        days_arr = np.empty(n, dtype=np.int64)
        rsi_diffs = np.empty(n, dtype=np.float64)
        bars_between = np.empty(n, dtype=np.float64)
        prev_rsi = np.empty(n, dtype=np.float64)
        rsi_values = np.empty(n, dtype=np.float64)
        price_slopes = np.empty(n, dtype=np.float64)
        atr = np.empty(n, dtype=np.float64)
        atr_pips = np.empty(n, dtype=np.float64)
        ma50 = np.empty(n, dtype=np.float64)
        ma200 = np.empty(n, dtype=np.float64)

        for i, s in enumerate(signals):
            bars[i] = s.bar
            prices[i] = s.price
            directions[i] = s.direction
            hours[i] = s.hour
            days_arr[i] = s.day
            attr = s.attributes
            rsi_diffs[i] = attr['rsi_diff']
            bars_between[i] = attr['bars_between']
            prev_rsi[i] = attr['prev_rsi']
            rsi_values[i] = attr['rsi_value']
            price_slopes[i] = attr['price_slope']
            atr[i] = attr['atr']
            atr_pips[i] = attr['atr_pips']
            ma50[i] = attr['ma50']
            ma200[i] = attr['ma200']

        return {
            'bars': bars,
            'prices': prices,
            'directions': directions,
            'hours': hours,
            'days': days_arr,
            'rsi_diffs': rsi_diffs,
            'bars_between': bars_between,
            'prev_rsi': prev_rsi,
            'rsi_values': rsi_values,
            'price_slopes': price_slopes,
            'atr': atr,
            'atr_pips': atr_pips,
            'ma50': ma50,
            'ma200': ma200,
        }

    def _filter_vectorized(self, params: Dict[str, Any]) -> np.ndarray:
        """Vectorized signal filtering using numpy boolean masks."""
        va = self._vec_arrays
        n = len(va['bars'])
        if n == 0:
            return np.array([], dtype=np.bool_)

        mask = np.ones(n, dtype=np.bool_)

        mask &= va['rsi_diffs'] >= params['min_rsi_diff']
        mask &= va['bars_between'] >= params['min_bars_between']
        mask &= va['bars_between'] <= params['max_bars_between']

        if params.get('require_pullback', False):
            rsi_overbought = params.get('rsi_overbought', 70)
            rsi_oversold = params.get('rsi_oversold', 30)
            long_mask = va['directions'] == 1
            long_pullback_fail = long_mask & (va['prev_rsi'] >= rsi_oversold) & (va['rsi_values'] >= rsi_oversold)
            short_mask = va['directions'] == -1
            short_pullback_fail = short_mask & (va['prev_rsi'] <= rsi_overbought) & (va['rsi_values'] <= rsi_overbought)
            mask &= ~(long_pullback_fail | short_pullback_fail)

        if params.get('use_slope_filter', True):
            mask &= va['price_slopes'] >= params['min_price_slope']
            mask &= va['price_slopes'] <= params['max_price_slope']

        if params.get('use_trend_filter', True):
            ma_valid = (va['ma50'] > 0) & (va['ma200'] > 0)
            long_trend_fail = (va['directions'] == 1) & ma_valid & ((va['prices'] < va['ma50']) | (va['prices'] < va['ma200']))
            short_trend_fail = (va['directions'] == -1) & ma_valid & ((va['prices'] > va['ma50']) | (va['prices'] > va['ma200']))
            mask &= ~(long_trend_fail | short_trend_fail)

        if params.get('use_time_filter', False):
            if not params.get('trade_monday', False):
                mask &= va['days'] != 0
            if not params.get('trade_friday', False):
                mask &= va['days'] != 4
            friday_close = params.get('friday_close_hour', 18)
            mask &= ~((va['days'] == 4) & (va['hours'] >= friday_close))
            mask &= va['days'] != 6
            start = params.get('trade_start_hour', 4)
            end = params.get('trade_end_hour', 22)
            if start < end:
                mask &= (va['hours'] >= start) & (va['hours'] < end)
            else:
                mask &= (va['hours'] >= start) | (va['hours'] < end)

        return mask

    def _compute_sl_tp_vectorized(
        self,
        mask: np.ndarray,
        params: Dict[str, Any],
        pip_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized SL/TP computation on filtered signals."""
        va = self._vec_arrays
        prices = va['prices'][mask]
        directions = va['directions'][mask]
        atr_raw = va['atr'][mask]
        atr_pips = va['atr_pips'][mask]
        n = len(prices)

        # === STOP LOSS (vectorized) ===
        sl_mode = params.get('sl_mode', 'atr')
        if sl_mode == 'fixed':
            sl_pips = np.full(n, params.get('sl_fixed_pips', 50), dtype=np.float64)
        elif sl_mode == 'atr':
            sl_atr_pct = params.get('sl_atr_pct', 100) / 100.0
            sl_pips = atr_pips * sl_atr_pct
        else:
            sl_pips = np.full(n, params.get('sl_fixed_pips', 50), dtype=np.float64)

        sl_pips = np.maximum(sl_pips, 10.0)

        # === TAKE PROFIT (vectorized) ===
        tp_mode = params.get('tp_mode', 'fixed')
        if tp_mode == 'rr':
            tp_pips = sl_pips * params.get('tp_rr_ratio', 2.0)
        elif tp_mode == 'atr':
            tp_pips = atr_pips * params.get('tp_atr_mult', 3.0)
        elif tp_mode == 'fixed':
            tp_pips = np.full(n, params.get('tp_fixed_pips', 30), dtype=np.float64)
        else:
            tp_pips = sl_pips * params.get('tp_rr_ratio', 2.0)

        is_long = directions == 1
        sl_prices = np.where(is_long, prices - sl_pips * pip_size, prices + sl_pips * pip_size)
        tp_prices = np.where(is_long, prices + tp_pips * pip_size, prices - tp_pips * pip_size)

        return sl_prices, tp_prices

    def _get_management_arrays_vectorized(
        self,
        mask: np.ndarray,
        params: Dict[str, Any],
        sl_prices: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Vectorized management array computation on filtered signals."""
        va = self._vec_arrays
        n = int(np.sum(mask))
        if n == 0:
            return None

        atr_pips = va['atr_pips'][mask]
        prices = va['prices'][mask]
        directions = va['directions'][mask]

        # Trailing stop arrays
        use_trailing = np.full(n, params.get('use_trailing', False), dtype=np.bool_)
        trail_start = np.full(n, params.get('trail_start_pips', 20), dtype=np.float64)
        trail_step = np.full(n, params.get('trail_step_pips', 10), dtype=np.float64)

        # V5: Trail mode (0=fixed, 1=chandelier)
        trail_mode_str = params.get('trail_mode', 'chandelier')
        trail_mode_val = 1 if trail_mode_str == 'chandelier' else 0
        trail_mode = np.full(n, trail_mode_val, dtype=np.int64)

        # V5: Chandelier ATR multiplier
        chandelier_atr_mult = np.full(n, params.get('chandelier_atr_mult', 3.0), dtype=np.float64)

        # Breakeven arrays - V5: wider ATR-scaled trigger
        use_be = np.full(n, params.get('use_break_even', False), dtype=np.bool_)
        be_atr_mult = params.get('be_atr_mult', 1.0)
        be_trigger = atr_pips * be_atr_mult
        be_offset = np.full(n, params.get('be_offset_pips', 5), dtype=np.float64)

        # Partial close arrays
        use_partial = np.full(n, params.get('use_partial_close', False), dtype=np.bool_)
        partial_pct = np.full(n, params.get('partial_close_pct', 0.3), dtype=np.float64)

        # V5: Partial target - SL-multiple based or fixed
        partial_target_mode = params.get('partial_target_mode', 'sl_multiple')
        if partial_target_mode == 'sl_multiple':
            # V5: Partial at N x SL distance
            pip_size = self._pip_size
            sl_dist_pips = np.abs(prices - sl_prices) / pip_size
            partial_target = sl_dist_pips * params.get('partial_target_sl_mult', 1.0)
        else:
            # Fixed: 50% of TP distance
            pip_size = self._pip_size
            sl_dist_pips = np.abs(prices - sl_prices) / pip_size

            tp_mode = params.get('tp_mode', 'fixed')
            if tp_mode == 'rr':
                tp_pips = sl_dist_pips * params.get('tp_rr_ratio', 2.0)
            elif tp_mode == 'atr':
                tp_pips = atr_pips * params.get('tp_atr_mult', 3.0)
            elif tp_mode == 'fixed':
                tp_pips = np.full(n, params.get('tp_fixed_pips', 30), dtype=np.float64)
            else:
                tp_pips = sl_dist_pips * params.get('tp_rr_ratio', 2.0)

            partial_target = tp_pips * 0.5

        max_bars = np.zeros(n, dtype=np.int64)

        # V5: Stale exit bars
        stale_exit_bars = np.full(n, params.get('stale_exit_bars', 0), dtype=np.int64)

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
            # V5 new arrays
            'trail_mode': trail_mode,
            'chandelier_atr_mult': chandelier_atr_mult,
            'atr_pips': atr_pips.copy(),
            'stale_exit_bars': stale_exit_bars,
        }


# Register in strategy registry
from strategies.rsi_fast import FAST_STRATEGIES
FAST_STRATEGIES['rsi_v5'] = RSIDivergenceFullFastV5
