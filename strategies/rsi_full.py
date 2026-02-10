"""
RSI Hidden Divergence Strategy - FULL VERSION with ALL parameters.

This is the comprehensive version with 35+ parameters like a real MT5 EA:
- RSI settings
- Swing detection
- Entry filters (slope, RSI levels, trend)
- Stop loss modes (fixed, ATR, swing-based)
- Take profit modes (fixed multiplier, ATR, R:R)
- Break even
- Trailing stop
- Partial closes
- Time filters (hours, days, sessions)
- Risk management

Supports full-featured optimization with staged group optimization.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.fast_strategy import FastStrategy, FastSignal, ParameterGroup, ParameterDef


@dataclass
class FullSignal:
    """Signal with ALL computed attributes for filtering."""
    bar: int
    direction: int  # 1=buy, -1=sell
    price: float
    hour: int
    day: int  # 0=Mon, 6=Sun

    # RSI attributes
    rsi_value: float
    rsi_diff: float

    # Swing attributes
    bars_between: int
    swing_price: float  # The swing low/high price for SL

    # Slope
    price_slope: float  # degrees

    # Volatility
    atr: float

    # Session info
    is_london: bool  # 8-16 UTC
    is_ny: bool      # 13-21 UTC
    is_asian: bool   # 0-8 UTC
    is_overlap: bool # London-NY overlap 13-16 UTC


class RSIDivergenceFullFast(FastStrategy):
    """
    RSI Hidden Divergence - FULL VERSION with 35+ parameters.

    Parameter groups for staged optimization:
    1. signal (8 params): RSI period, swing strength, divergence settings
    2. filters (7 params): Slope, trend, time filters
    3. risk (6 params): SL/TP modes, position sizing
    4. management (8 params): Trailing, breakeven, partial closes
    5. time (6 params): Trading hours, days

    Total: 35 parameters
    """

    name = "RSI_Divergence_Full"
    version = "4.0"

    # Pre-compute ranges
    RSI_PERIODS = [5, 7, 9, 11, 14, 18, 21]
    SWING_STRENGTHS = [3, 5, 7, 10]
    ATR_PERIOD = 14

    def get_parameter_groups(self) -> Optional[Dict[str, ParameterGroup]]:
        """
        Return parameters organized into groups for staged optimization.
        """
        groups = {}

        # === Group 1: Signal Generation ===
        signal = ParameterGroup('signal', 'Signal generation parameters')
        signal.add_param('rsi_period', [7, 14, 21], default=14)
        signal.add_param('rsi_overbought', [70, 75, 80], default=70)
        signal.add_param('rsi_oversold', [20, 25, 30], default=30)
        signal.add_param('min_rsi_diff', [3.0, 5.0, 8.0, 12.0], default=5.0)
        signal.add_param('swing_strength', [3, 5, 7], default=5)
        signal.add_param('min_bars_between', [5, 10, 15, 20], default=10)
        signal.add_param('max_bars_between', [40, 60, 80, 100], default=80)
        signal.add_param('require_pullback', [True, False], default=False)
        groups['signal'] = signal

        # === Group 2: Entry Filters ===
        filters = ParameterGroup('filters', 'Entry filter parameters')
        filters.add_param('use_slope_filter', [True, False], default=True)
        filters.add_param('min_price_slope', [5.0, 15.0, 25.0], default=5.0)
        filters.add_param('max_price_slope', [50.0, 65.0, 80.0], default=65.0)
        filters.add_param('use_rsi_extreme_filter', [True, False], default=False)
        filters.add_param('use_trend_filter', [True, False], default=False)
        filters.add_param('trend_ma_period', [50, 100, 200], default=100)
        filters.add_param('max_spread_pips', [2.0, 3.0, 5.0], default=3.0)
        groups['filters'] = filters

        # === Group 3: Risk (SL/TP) ===
        risk = ParameterGroup('risk', 'Stop loss and take profit')
        risk.add_param('sl_mode', ['fixed', 'atr', 'swing'], default='fixed')
        risk.add_param('sl_fixed_pips', [25, 35, 50, 75], default=35)
        risk.add_param('sl_atr_mult', [1.5, 2.0, 2.5, 3.0], default=2.0)
        risk.add_param('sl_swing_buffer', [5, 10, 15], default=10)
        risk.add_param('tp_mode', ['rr', 'atr', 'fixed'], default='rr')
        risk.add_param('tp_rr_ratio', [1.0, 1.5, 2.0, 2.5, 3.0], default=1.5)
        risk.add_param('tp_atr_mult', [2.0, 3.0, 4.0], default=3.0)
        risk.add_param('tp_fixed_pips', [30, 50, 75, 100], default=50)  # FIX: Added for fixed TP mode
        groups['risk'] = risk

        # === Group 4: Trade Management ===
        management = ParameterGroup('management', 'Trailing, breakeven, partial closes')
        management.add_param('use_trailing', [True, False], default=False)
        management.add_param('trail_start_pips', [20, 30, 50], default=30)
        management.add_param('trail_step_pips', [10, 15, 20], default=15)
        management.add_param('use_break_even', [True, False], default=False)
        management.add_param('be_trigger_pips', [15, 20, 30, 40], default=20)
        management.add_param('be_offset_pips', [0, 2, 5], default=2)
        management.add_param('use_partial_close', [True, False], default=False)
        management.add_param('partial_close_pct', [0.3, 0.5, 0.7], default=0.5)
        groups['management'] = management

        # === Group 5: Time Filters ===
        time_group = ParameterGroup('time', 'Trading time filters')
        time_group.add_param('use_time_filter', [True, False], default=True)
        time_group.add_param('trade_start_hour', [0, 2, 4, 6, 8], default=6)
        time_group.add_param('trade_end_hour', [18, 20, 22, 23], default=20)
        time_group.add_param('trade_monday', [True, False], default=True)
        time_group.add_param('trade_friday', [True, False], default=True)
        time_group.add_param('friday_close_hour', [18, 20], default=20)
        groups['time'] = time_group

        return groups

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Return FULL parameter grid - 35 parameters.
        Built from parameter groups.
        """
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
        """
        Get trade management arrays for full-featured backtest.
        """
        n = len(signals)
        if n == 0:
            return None

        # Trailing stop arrays
        use_trailing = np.full(n, params.get('use_trailing', False), dtype=np.bool_)
        trail_start = np.full(n, params.get('trail_start_pips', 30), dtype=np.float64)
        trail_step = np.full(n, params.get('trail_step_pips', 15), dtype=np.float64)

        # Breakeven arrays
        use_be = np.full(n, params.get('use_break_even', False), dtype=np.bool_)
        be_trigger = np.full(n, params.get('be_trigger_pips', 20), dtype=np.float64)
        be_offset = np.full(n, params.get('be_offset_pips', 2), dtype=np.float64)

        # Partial close arrays
        use_partial = np.full(n, params.get('use_partial_close', False), dtype=np.bool_)
        partial_pct = np.full(n, params.get('partial_close_pct', 0.5), dtype=np.float64)

        # Partial target based on SL distance
        partial_target = np.zeros(n, dtype=np.float64)
        for i, sig in enumerate(signals):
            # Partial close at 50% of TP distance
            sl_pips = self._get_sl_pips(sig, params)
            # FIX: Pass actual signal ATR instead of using default 35 pips
            atr_pips = sig.attributes.get('atr_pips', 35.0)
            tp_pips = self._get_tp_pips(sl_pips, params, atr_pips)
            partial_target[i] = tp_pips * 0.5

        # Max bars (0 = unlimited)
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
        """Calculate SL pips for a signal."""
        sl_mode = params.get('sl_mode', 'fixed')
        attr = signal.attributes

        if sl_mode == 'fixed':
            return params.get('sl_fixed_pips', 35)
        elif sl_mode == 'atr':
            # FIX: Use pre-computed atr_pips from signal attributes
            atr_pips = attr.get('atr_pips', attr.get('atr', 0.0035) / self._pip_size)
            return atr_pips * params.get('sl_atr_mult', 2.0)
        elif sl_mode == 'swing':
            swing_dist = abs(signal.price - attr.get('swing_price', signal.price)) / self._pip_size
            return max(swing_dist + params.get('sl_swing_buffer', 10), 10)  # Minimum 10 pips
        return params.get('sl_fixed_pips', 35)

    def _get_tp_pips(self, sl_pips: float, params: Dict[str, Any], atr_pips: float = 35.0) -> float:
        """Calculate TP pips based on SL."""
        tp_mode = params.get('tp_mode', 'rr')

        if tp_mode == 'rr':
            return sl_pips * params.get('tp_rr_ratio', 1.5)
        elif tp_mode == 'atr':
            return atr_pips * params.get('tp_atr_mult', 3.0)
        elif tp_mode == 'fixed':
            # FIX: Properly handle fixed TP mode with dedicated parameter
            return params.get('tp_fixed_pips', 50)
        return sl_pips * params.get('tp_rr_ratio', 1.5)  # Fallback to RR

    def _calc_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Fast RSI calculation."""
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
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return atr

    def _calc_ma(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        ma = np.zeros_like(close)
        for i in range(period - 1, len(close)):
            ma[i] = np.mean(close[i-period+1:i+1])
        return ma

    def _find_swings(self, highs: np.ndarray, lows: np.ndarray, strength: int) -> Tuple[List, List]:
        """Find swing high and low points."""
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
                swing_highs.append((i, highs[i]))

            # Swing low
            is_low = True
            for j in range(1, strength + 1):
                if lows[i - j] <= lows[i] or lows[i + j] <= lows[i]:
                    is_low = False
                    break
            if is_low:
                swing_lows.append((i, lows[i]))

        return swing_highs, swing_lows

    def precompute(self, df: pd.DataFrame) -> List[FastSignal]:
        """
        Pre-compute ALL possible signals with ALL attributes.
        """
        signals = []
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        hours = df.index.hour.values
        days = df.index.dayofweek.values

        # Pre-compute ATR (fixed period for now)
        atr = self._calc_atr(highs, lows, closes, self.ATR_PERIOD)

        # Pre-compute MAs for trend filter
        ma50 = self._calc_ma(closes, 50)
        ma100 = self._calc_ma(closes, 100)
        ma200 = self._calc_ma(closes, 200)

        for rsi_period in self.RSI_PERIODS:
            rsi = self._calc_rsi(closes, rsi_period)

            for strength in self.SWING_STRENGTHS:
                swing_highs, swing_lows = self._find_swings(highs, lows, strength)

                # Bullish divergences (from swing lows)
                for i in range(1, len(swing_lows)):
                    prev_bar, prev_price = swing_lows[i - 1]
                    curr_bar, curr_price = swing_lows[i]
                    bars_between = curr_bar - prev_bar

                    if bars_between < 2 or bars_between > 120:
                        continue

                    prev_rsi = rsi[prev_bar]
                    curr_rsi = rsi[curr_bar]

                    # Hidden bullish: Price HL, RSI LL
                    if curr_price > prev_price and curr_rsi < prev_rsi:
                        rsi_diff = abs(prev_rsi - curr_rsi)
                        slope = (curr_price - prev_price) / bars_between * 10000
                        slope_angle = abs(np.arctan(slope) * 180 / np.pi)

                        # FIX: Lookahead bias - swing at curr_bar can only be confirmed
                        # after seeing 'strength' more bars. Signal emits one bar after confirmation.
                        signal_bar = curr_bar + strength + 1
                        if signal_bar < len(closes):
                            hour = hours[signal_bar]
                            day = days[signal_bar]

                            signals.append(FastSignal(
                                bar=signal_bar,
                                direction=1,
                                price=closes[signal_bar],
                                hour=hour,
                                day=day,
                                attributes={
                                    'rsi_period': rsi_period,
                                    'swing_strength': strength,
                                    'rsi_value': curr_rsi,
                                    'prev_rsi': prev_rsi,  # Store for OB/OS filter
                                    'rsi_diff': rsi_diff,
                                    'bars_between': bars_between,
                                    'swing_price': curr_price,  # Swing low for SL
                                    'price_slope': slope_angle,
                                    'atr': atr[signal_bar],
                                    'atr_pips': atr[signal_bar] / self._pip_size,  # Store for ATR TP calc
                                    'ma50': ma50[signal_bar],
                                    'ma100': ma100[signal_bar],
                                    'ma200': ma200[signal_bar],
                                    'is_london': 8 <= hour < 16,
                                    'is_ny': 13 <= hour < 21,
                                    'is_asian': hour < 8,
                                    'is_overlap': 13 <= hour < 16,
                                }
                            ))

                # Bearish divergences (from swing highs)
                for i in range(1, len(swing_highs)):
                    prev_bar, prev_price = swing_highs[i - 1]
                    curr_bar, curr_price = swing_highs[i]
                    bars_between = curr_bar - prev_bar

                    if bars_between < 2 or bars_between > 120:
                        continue

                    prev_rsi = rsi[prev_bar]
                    curr_rsi = rsi[curr_bar]

                    # Hidden bearish: Price LH, RSI HH
                    if curr_price < prev_price and curr_rsi > prev_rsi:
                        rsi_diff = abs(curr_rsi - prev_rsi)
                        slope = abs(prev_price - curr_price) / bars_between * 10000
                        slope_angle = abs(np.arctan(slope) * 180 / np.pi)

                        # FIX: Lookahead bias - swing at curr_bar can only be confirmed
                        # after seeing 'strength' more bars. Signal emits one bar after confirmation.
                        signal_bar = curr_bar + strength + 1
                        if signal_bar < len(closes):
                            hour = hours[signal_bar]
                            day = days[signal_bar]

                            signals.append(FastSignal(
                                bar=signal_bar,
                                direction=-1,
                                price=closes[signal_bar],
                                hour=hour,
                                day=day,
                                attributes={
                                    'rsi_period': rsi_period,
                                    'swing_strength': strength,
                                    'rsi_value': curr_rsi,
                                    'prev_rsi': prev_rsi,  # Store for OB/OS filter
                                    'rsi_diff': rsi_diff,
                                    'bars_between': bars_between,
                                    'swing_price': curr_price,  # Swing high for SL
                                    'price_slope': slope_angle,
                                    'atr': atr[signal_bar],
                                    'atr_pips': atr[signal_bar] / self._pip_size,  # Store for ATR TP calc
                                    'ma50': ma50[signal_bar],
                                    'ma100': ma100[signal_bar],
                                    'ma200': ma200[signal_bar],
                                    'is_london': 8 <= hour < 16,
                                    'is_ny': 13 <= hour < 21,
                                    'is_asian': hour < 8,
                                    'is_overlap': 13 <= hour < 16,
                                }
                            ))

        # Remove duplicates - but KEEP different RSI period/swing strength variants
        # FIX: Previously dropped all variants at same bar, making RSI period optimization meaningless
        seen = set()
        unique = []
        for s in signals:
            # Include rsi_period and swing_strength in key so different params = different signals
            key = (s.bar, s.direction, s.attributes['rsi_period'], s.attributes['swing_strength'])
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
        """
        Filter pre-computed signals based on ALL parameters.

        FIX: Now properly enforces ALL parameters including rsi_period, swing_strength,
        rsi_overbought/oversold, and require_pullback.
        """
        result = []

        # Get threshold params once (with defaults)
        rsi_overbought = params.get('rsi_overbought', 70)
        rsi_oversold = params.get('rsi_oversold', 30)
        target_rsi_period = params.get('rsi_period', 14)
        target_swing_strength = params.get('swing_strength', 5)

        for s in signals:
            attr = s.attributes

            # === RSI Period Match ===
            # FIX: Now enforced - only accept signals from the requested RSI period
            if attr['rsi_period'] != target_rsi_period:
                continue

            # === Swing Strength Match ===
            # FIX: Now enforced - only accept signals from the requested swing strength
            if attr['swing_strength'] != target_swing_strength:
                continue

            # === RSI Diff Filter ===
            if attr['rsi_diff'] < params['min_rsi_diff']:
                continue

            # === Swing Detection Filters ===
            if attr['bars_between'] < params['min_bars_between']:
                continue
            if attr['bars_between'] > params['max_bars_between']:
                continue

            # === Require Pullback Filter ===
            # FIX: Now implemented - require RSI to have been in extreme zone
            if params.get('require_pullback', False):
                prev_rsi = attr.get('prev_rsi', 50)
                curr_rsi = attr.get('rsi_value', 50)
                if s.direction == 1:  # Buy - require prior oversold condition
                    if prev_rsi >= rsi_oversold and curr_rsi >= rsi_oversold:
                        continue  # Neither swing was in oversold - skip
                else:  # Sell - require prior overbought condition
                    if prev_rsi <= rsi_overbought and curr_rsi <= rsi_overbought:
                        continue  # Neither swing was in overbought - skip

            # === Slope Filter ===
            if params['use_slope_filter']:
                if attr['price_slope'] < params['min_price_slope']:
                    continue
                if attr['price_slope'] > params['max_price_slope']:
                    continue

            # === RSI Extreme Filter ===
            # FIX: Now uses actual parameters instead of hardcoded 70/30
            if params.get('use_rsi_extreme_filter', False):
                prev_rsi = attr.get('prev_rsi', 50)
                curr_rsi = attr.get('rsi_value', 50)
                if s.direction == 1:  # Buy - check for oversold
                    if prev_rsi >= rsi_oversold and curr_rsi >= rsi_oversold:
                        continue  # Neither was oversold
                else:  # Sell - check for overbought
                    if prev_rsi <= rsi_overbought and curr_rsi <= rsi_overbought:
                        continue  # Neither was overbought

            # === Trend Filter ===
            if params['use_trend_filter']:
                ma_period = params['trend_ma_period']
                ma_key = f'ma{ma_period}'
                if ma_key in attr:
                    ma_val = attr[ma_key]
                    if ma_val > 0:  # MA is valid
                        if s.direction == 1 and s.price < ma_val:  # Buy below MA
                            continue
                        if s.direction == -1 and s.price > ma_val:  # Sell above MA
                            continue

            # === Time Filters ===
            if params['use_time_filter']:
                # Day filters
                if s.day == 0 and not params['trade_monday']:  # Monday
                    continue
                if s.day == 4 and not params['trade_friday']:  # Friday
                    continue
                if s.day == 4 and s.hour >= params['friday_close_hour']:
                    continue
                if s.day == 6:  # Sunday
                    continue

                # Hour filters
                start = params['trade_start_hour']
                end = params['trade_end_hour']
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
        """
        Compute stop loss and take profit based on mode.
        """
        attr = signal.attributes

        # === STOP LOSS ===
        sl_mode = params['sl_mode']

        if sl_mode == 'fixed':
            sl_pips = params['sl_fixed_pips']
        elif sl_mode == 'atr':
            atr_pips = attr['atr'] / pip_size
            sl_pips = atr_pips * params['sl_atr_mult']
        elif sl_mode == 'swing':
            swing_dist = abs(signal.price - attr['swing_price']) / pip_size
            sl_pips = swing_dist + params['sl_swing_buffer']
        else:
            sl_pips = params['sl_fixed_pips']

        # Ensure minimum SL
        sl_pips = max(sl_pips, 10)

        # === TAKE PROFIT ===
        tp_mode = params['tp_mode']

        if tp_mode == 'rr':
            tp_pips = sl_pips * params['tp_rr_ratio']
        elif tp_mode == 'atr':
            atr_pips = attr['atr'] / pip_size
            tp_pips = atr_pips * params['tp_atr_mult']
        elif tp_mode == 'fixed':
            tp_pips = params.get('tp_fixed_pips', 50)
        else:  # fallback to RR
            tp_pips = sl_pips * params['tp_rr_ratio']

        # Calculate prices
        if signal.direction == 1:  # Buy
            sl = signal.price - sl_pips * pip_size
            tp = signal.price + tp_pips * pip_size
        else:  # Sell
            sl = signal.price + sl_pips * pip_size
            tp = signal.price - tp_pips * pip_size

        return sl, tp


# Add to registry
from strategies.rsi_fast import FAST_STRATEGIES
FAST_STRATEGIES['rsi_full'] = RSIDivergenceFullFast
