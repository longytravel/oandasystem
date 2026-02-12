"""
Adapter to bridge pipeline FastStrategy to the live Strategy interface.

The pipeline optimizer uses FastStrategy (vectorized, numba-based) while
the LiveTrader expects Strategy with generate_signals(df) -> List[Signal].

This adapter:
1. Wraps a FastStrategy + optimized params
2. On generate_signals(df): runs precompute + get_all_arrays
3. Returns Signal objects for the latest bar
4. Provides trade management (trailing stop, breakeven) via manage_positions()
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from loguru import logger

from strategies.base import Strategy, Signal, SignalType


class PipelineStrategyAdapter(Strategy):
    """
    Wraps a FastStrategy for use with LiveTrader.

    Usage:
        from strategies.rsi_full_v3 import RSIDivergenceFullFastV3
        fast = RSIDivergenceFullFastV3()
        params = {...}  # from pipeline best candidate
        adapter = PipelineStrategyAdapter(fast, params, pair='GBP_USD')
        trader = LiveTrader(adapter, instrument='GBP_USD', timeframe='M15')
    """

    name = "PipelineAdapter"
    version = "1.0"

    def __init__(
        self,
        fast_strategy,
        optimized_params: Dict[str, Any],
        pair: str = 'GBP_USD',
    ):
        self.fast_strategy = fast_strategy
        self.optimized_params = optimized_params
        self.pair = pair
        self._pip_size = 0.01 if 'JPY' in pair else 0.0001

        # Set adapter name from underlying strategy
        self.name = f"Pipeline_{fast_strategy.name}"
        self.version = fast_strategy.version

        # Trade management params (extracted from optimized_params)
        self.use_trailing = optimized_params.get('use_trailing', False)
        self.trail_start_pips = optimized_params.get('trail_start_pips', 50)
        self.trail_step_pips = optimized_params.get('trail_step_pips', 10)
        self.use_break_even = optimized_params.get('use_break_even', False)
        self.be_atr_mult = optimized_params.get('be_atr_mult', 0.5)
        self.be_offset_pips = optimized_params.get('be_offset_pips', 5)

        # Initialize base Strategy (needed for LiveTrader compatibility)
        self.params = optimized_params

        logger.info(f"PipelineStrategyAdapter: {fast_strategy.name} v{fast_strategy.version}")
        logger.info(f"  Pair: {pair}, Pip size: {self._pip_size}")
        logger.info(f"  Trailing: {self.use_trailing} (start={self.trail_start_pips}, step={self.trail_step_pips})")
        logger.info(f"  Breakeven: {self.use_break_even} (mult={self.be_atr_mult}, offset={self.be_offset_pips})")

    def get_default_parameters(self) -> Dict[str, Any]:
        return self.optimized_params

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """
        Run FastStrategy on candle data and extract signals for the latest bar.

        This is called by LiveTrader on each candle close.
        """
        if df is None or df.empty:
            return []

        # Ensure we have the columns expected
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
                return []

        try:
            # Set pip size on the fast strategy
            self.fast_strategy._pip_size = self._pip_size

            # Precompute all signals on the dataset
            n_signals = self.fast_strategy.precompute_for_dataset(df)

            if n_signals == 0:
                return []

            # Get filtered signal arrays with the optimized params
            highs = df['high'].values.astype(np.float64)
            lows = df['low'].values.astype(np.float64)
            closes = df['close'].values.astype(np.float64)
            days = df.index.dayofweek.values.astype(np.int64) if hasattr(df.index, 'dayofweek') else np.zeros(len(df), dtype=np.int64)

            signal_arrays, mgmt_arrays = self.fast_strategy.get_all_arrays(
                self.optimized_params, highs, lows, closes, days
            )

            entry_bars = signal_arrays['entry_bars']
            if len(entry_bars) == 0:
                return []

            # Find signals on the LAST bar (most recent candle close)
            last_bar = len(df) - 1
            signals = []

            for i in range(len(entry_bars)):
                if entry_bars[i] == last_bar:
                    direction = int(signal_arrays['directions'][i])
                    entry_price = float(signal_arrays['entry_prices'][i])
                    sl_price = float(signal_arrays['sl_prices'][i])
                    tp_price = float(signal_arrays['tp_prices'][i])

                    signal_type = SignalType.BUY if direction == 1 else SignalType.SELL
                    timestamp = df.index[last_bar]

                    sl_pips = abs(entry_price - sl_price) / self._pip_size
                    tp_pips = abs(tp_price - entry_price) / self._pip_size
                    rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0

                    signal = Signal(
                        type=signal_type,
                        price=entry_price,
                        timestamp=timestamp,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        metadata={
                            'strategy': self.fast_strategy.name,
                            'sl_pips': round(sl_pips, 1),
                            'tp_pips': round(tp_pips, 1),
                            'rr_ratio': round(rr_ratio, 1),
                            'params': self.optimized_params,
                        }
                    )
                    signals.append(signal)

                    logger.info(
                        f"Signal: {'BUY' if direction == 1 else 'SELL'} @ {entry_price:.5f} "
                        f"SL={sl_price:.5f} ({sl_pips:.0f}pip) "
                        f"TP={tp_price:.5f} ({tp_pips:.0f}pip) "
                        f"R:R={rr_ratio:.1f}"
                    )

            return signals

        except Exception as e:
            logger.exception(f"Signal generation failed: {e}")
            return []

    def manage_positions(
        self,
        positions: list,
        current_price: float,
        atr_pips: float,
        client=None,
        bar_high: float = None,
        bar_low: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Manage open positions: trailing stop and breakeven.

        Called by the trading loop on each candle close for each open position.
        Uses bar_high/bar_low for BE/trailing triggers to match backtest behavior
        (backtest checks intra-bar excursions, not just close price).

        Args:
            positions: List of LivePosition objects
            current_price: Current market price (candle close)
            atr_pips: Current ATR in pips (for breakeven calculation)
            client: OandaClient for modifying trades (None for dry run)
            bar_high: High of current candle (for BE/trailing trigger check)
            bar_low: Low of current candle (for BE/trailing trigger check)

        Returns:
            List of management actions taken
        """
        actions = []

        # Fall back to current_price if high/low not provided
        if bar_high is None:
            bar_high = current_price
        if bar_low is None:
            bar_low = current_price

        for pos in positions:
            if pos.instrument != self.pair:
                continue

            is_long = pos.direction == "BUY"
            entry_price = pos.entry_price
            current_sl = pos.stop_loss

            # Use bar_high for longs, bar_low for shorts to check max favorable excursion
            # This matches the backtest engine which uses intra-bar highs/lows
            if is_long:
                best_price = bar_high
                profit_pips = (best_price - entry_price) / self._pip_size
            else:
                best_price = bar_low
                profit_pips = (entry_price - best_price) / self._pip_size

            new_sl = None

            # --- Breakeven ---
            if self.use_break_even and profit_pips > 0:
                be_trigger_pips = self.be_atr_mult * atr_pips
                be_sl = entry_price + (self.be_offset_pips * self._pip_size if is_long
                                       else -self.be_offset_pips * self._pip_size)

                if profit_pips >= be_trigger_pips:
                    # Check if SL hasn't already been moved to breakeven or better
                    if is_long and current_sl < be_sl:
                        new_sl = be_sl
                        actions.append({
                            'trade_id': pos.trade_id,
                            'action': 'breakeven',
                            'old_sl': current_sl,
                            'new_sl': new_sl,
                            'profit_pips': profit_pips,
                            'trigger': f'bar_high={bar_high:.5f}',
                        })
                    elif not is_long and current_sl > be_sl:
                        new_sl = be_sl
                        actions.append({
                            'trade_id': pos.trade_id,
                            'action': 'breakeven',
                            'old_sl': current_sl,
                            'new_sl': new_sl,
                            'profit_pips': profit_pips,
                            'trigger': f'bar_low={bar_low:.5f}',
                        })

            # --- Trailing Stop ---
            if self.use_trailing and profit_pips >= self.trail_start_pips:
                # For trailing, use current_price (close) to set the new SL level
                # but use bar_high/low for the activation check (matching backtest)
                if is_long:
                    trail_sl = current_price - self.trail_step_pips * self._pip_size
                    if trail_sl > current_sl and (new_sl is None or trail_sl > new_sl):
                        new_sl = trail_sl
                        actions.append({
                            'trade_id': pos.trade_id,
                            'action': 'trailing',
                            'old_sl': current_sl,
                            'new_sl': new_sl,
                            'profit_pips': profit_pips,
                        })
                else:
                    trail_sl = current_price + self.trail_step_pips * self._pip_size
                    if trail_sl < current_sl and (new_sl is None or trail_sl < new_sl):
                        new_sl = trail_sl
                        actions.append({
                            'trade_id': pos.trade_id,
                            'action': 'trailing',
                            'old_sl': current_sl,
                            'new_sl': new_sl,
                            'profit_pips': profit_pips,
                        })

            # Execute SL modification
            if new_sl is not None and client is not None:
                try:
                    client.modify_trade(pos.trade_id, stop_loss_price=new_sl)
                    pos.stop_loss = new_sl
                    logger.info(
                        f"Trade {pos.trade_id}: SL moved {current_sl:.5f} -> {new_sl:.5f} "
                        f"({actions[-1]['action']}, profit={profit_pips:.0f}pip)"
                    )
                except Exception as e:
                    logger.error(f"Failed to modify trade {pos.trade_id}: {e}")

        return actions
