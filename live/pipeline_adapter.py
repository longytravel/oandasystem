"""
Adapter to bridge pipeline FastStrategy to the live Strategy interface.

The pipeline optimizer uses FastStrategy (vectorized, numba-based) while
the LiveTrader expects Strategy with generate_signals(df) -> List[Signal].

This adapter:
1. Wraps a FastStrategy + optimized params
2. On generate_signals(df): runs precompute + get_all_arrays
3. Returns Signal objects for the latest bar
4. Provides trade management (trailing stop, breakeven, chandelier,
   partial close, stale exit, max bars) via manage_positions()
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
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

        # === Trade management params (extracted from optimized_params) ===

        # Trailing stop
        self.use_trailing = optimized_params.get('use_trailing', False)
        self.trail_mode = optimized_params.get('trail_mode', 0)  # 0=fixed, 1=chandelier
        self.trail_start_pips = optimized_params.get('trail_start_pips', 50)
        self.trail_step_pips = optimized_params.get('trail_step_pips', 10)
        # Cap step at start distance (matches backtester)
        self.trail_step_pips = min(self.trail_step_pips, self.trail_start_pips)
        self.chandelier_atr_mult = optimized_params.get('chandelier_atr_mult', 3.0)

        # Breakeven
        self.use_break_even = optimized_params.get('use_break_even', False)
        self.be_trigger_pips = optimized_params.get('be_trigger_pips', None)  # Fixed pips (backtester)
        self.be_atr_mult = optimized_params.get('be_atr_mult', 0.5)  # Dynamic (legacy)
        self.be_offset_pips = optimized_params.get('be_offset_pips', 5)

        # Partial close
        self.use_partial_close = optimized_params.get('use_partial_close', False)
        self.partial_close_pct = optimized_params.get('partial_close_pct', 0.5)
        self.partial_target_pips = optimized_params.get('partial_target_pips', 20)

        # Stale & max bars exit
        self.stale_exit_bars = optimized_params.get('stale_exit_bars', 0)
        self.max_bars_in_trade = optimized_params.get('max_bars_in_trade', 0)

        # Fixed ATR from params (for chandelier, stale exit — matches backtester per-signal atr)
        self._param_atr_pips = optimized_params.get('atr_pips', 35.0)

        # === Per-trade state tracking ===
        self._trail_highs: Dict[str, float] = {}   # pos_trail_high equivalent
        self._trail_active: Dict[str, bool] = {}    # pos_trail_active equivalent
        self._be_triggered: Dict[str, bool] = {}    # pos_be_triggered equivalent
        self._partial_closed: Dict[str, bool] = {}  # pos_partial_done equivalent
        self._bars_in_trade: Dict[str, int] = {}    # bar counter per trade

        # === Action logging & persistence ===
        self._state_dir: Optional[Path] = None

        # Initialize base Strategy (needed for LiveTrader compatibility)
        self.params = optimized_params

        logger.info(f"PipelineStrategyAdapter: {fast_strategy.name} v{fast_strategy.version}")
        logger.info(f"  Pair: {pair}, Pip size: {self._pip_size}")
        logger.info(f"  Trailing: {self.use_trailing} mode={self.trail_mode} "
                    f"(start={self.trail_start_pips}, step={self.trail_step_pips})")
        if self.trail_mode == 1:
            logger.info(f"  Chandelier: mult={self.chandelier_atr_mult}, "
                       f"atr_pips={self._param_atr_pips}")
        logger.info(f"  Breakeven: {self.use_break_even} "
                    f"(trigger={self.be_trigger_pips or f'{self.be_atr_mult}*ATR'}, "
                    f"offset={self.be_offset_pips})")
        logger.info(f"  Partial close: {self.use_partial_close} "
                    f"(pct={self.partial_close_pct}, target={self.partial_target_pips}pip)")
        logger.info(f"  Stale exit: {self.stale_exit_bars} bars, "
                    f"Max bars: {self.max_bars_in_trade}")

    # ═══════════════════════════════════════════════════════════════
    # State persistence & logging
    # ═══════════════════════════════════════════════════════════════

    def set_state_dir(self, state_dir):
        """Set state directory for persistence and action logging."""
        self._state_dir = Path(state_dir)
        self._load_mgmt_state()

    def get_mgmt_state(self) -> dict:
        """Get management state dict for persistence (restart recovery)."""
        return {
            'trail_highs': dict(self._trail_highs),
            'trail_active': {k: bool(v) for k, v in self._trail_active.items()},
            'be_triggered': {k: bool(v) for k, v in self._be_triggered.items()},
            'partial_closed': {k: bool(v) for k, v in self._partial_closed.items()},
            'bars_in_trade': {k: int(v) for k, v in self._bars_in_trade.items()},
        }

    def restore_mgmt_state(self, state: dict):
        """Restore management state from dict (restart recovery)."""
        self._trail_highs = {k: float(v) for k, v in state.get('trail_highs', {}).items()}
        self._trail_active = {k: bool(v) for k, v in state.get('trail_active', {}).items()}
        self._be_triggered = {k: bool(v) for k, v in state.get('be_triggered', {}).items()}
        self._partial_closed = {k: bool(v) for k, v in state.get('partial_closed', {}).items()}
        self._bars_in_trade = {k: int(v) for k, v in state.get('bars_in_trade', {}).items()}

    def _save_mgmt_state(self):
        """Persist management state to disk."""
        if not self._state_dir:
            return
        state_file = self._state_dir / 'mgmt_state.json'
        try:
            tmp = state_file.with_suffix('.json.tmp')
            with open(tmp, 'w') as f:
                json.dump(self.get_mgmt_state(), f, indent=2)
            os.replace(str(tmp), str(state_file))
        except Exception as e:
            logger.warning(f"Failed to save mgmt state: {e}")

    def _load_mgmt_state(self):
        """Load management state from disk."""
        if not self._state_dir:
            return
        state_file = self._state_dir / 'mgmt_state.json'
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.restore_mgmt_state(state)
                logger.info(f"Restored mgmt state: {len(self._bars_in_trade)} tracked trades")
            except Exception as e:
                logger.warning(f"Failed to load mgmt state: {e}")

    def _log_mgmt_action(self, action: dict):
        """Append a management action to mgmt_actions.jsonl (line-delimited JSON)."""
        if not self._state_dir:
            return
        log_file = self._state_dir / 'mgmt_actions.jsonl'
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(action) + '\n')
        except Exception as e:
            logger.warning(f"Failed to log mgmt action: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Signal generation (unchanged)
    # ═══════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════
    # Position management
    # ═══════════════════════════════════════════════════════════════

    def manage_positions(
        self,
        positions: list,
        current_price: float,
        atr_pips: float,
        client=None,
        bar_high: float = None,
        bar_low: float = None,
        bar_time=None,
    ) -> List[Dict[str, Any]]:
        """
        Manage open positions: trailing stop, breakeven, chandelier,
        partial close, stale exit, max bars exit.

        Called by the trading loop on each candle close for each open position.
        Matches backtester logic order (numba_backtest.py):
          1. Max bars exit
          2. Stale exit
          3. Breakeven (one-time trigger)
          4. Trailing stop (mode 0: fixed pip, mode 1: chandelier)
          5. Partial close

        Args:
            positions: List of LivePosition objects
            current_price: Current market price (candle close)
            atr_pips: Current ATR in pips (for legacy BE calculation)
            client: OandaClient for modifying/closing trades (None for dry run)
            bar_high: High of current candle
            bar_low: Low of current candle
            bar_time: Timestamp of current bar (for action logging)

        Returns:
            List of management actions taken
        """
        actions = []
        bar_time_str = str(bar_time) if bar_time else datetime.now(timezone.utc).isoformat()

        # Fall back to current_price if high/low not provided
        if bar_high is None:
            bar_high = current_price
        if bar_low is None:
            bar_low = current_price

        for pos in positions:
            if pos.instrument != self.pair:
                continue

            tid = pos.trade_id
            is_long = pos.direction == "BUY"
            entry_price = pos.entry_price
            current_sl = pos.stop_loss

            # --- Initialize state for new trades (entry bar) ---
            # Backtester does `continue` on entry bar, skipping management
            if tid not in self._bars_in_trade:
                self._bars_in_trade[tid] = 0
                self._trail_highs[tid] = 0.0
                self._trail_active[tid] = False
                self._be_triggered[tid] = False
                self._partial_closed[tid] = False
                continue

            # Increment bar counter
            self._bars_in_trade[tid] += 1
            bars = self._bars_in_trade[tid]

            # ── 1. Max bars exit ──────────────────────────────────
            # Backtester: if bars_in_trade >= max_bars → exit at close
            if self.max_bars_in_trade > 0 and bars >= self.max_bars_in_trade:
                action = {
                    'timestamp': bar_time_str,
                    'trade_id': tid,
                    'bar_number': bars,
                    'action': 'max_bars_exit',
                    'trigger': f'bars={bars} >= max={self.max_bars_in_trade}',
                }
                actions.append(action)
                self._log_mgmt_action(action)
                logger.info(f"Trade {tid}: MAX BARS EXIT after {bars} bars")
                if client is not None:
                    try:
                        client.close_trade(tid)
                    except Exception as e:
                        logger.error(f"Failed to close trade {tid} (max bars): {e}")
                continue  # Trade is closing, skip management

            # ── 2. Stale exit ─────────────────────────────────────
            # Backtester: if bars >= stale_exit_bars AND move < 0.5R → exit
            if self.stale_exit_bars > 0 and bars >= self.stale_exit_bars:
                half_r = self._param_atr_pips * self._pip_size * 0.5
                if is_long:
                    move = current_price - entry_price
                else:
                    move = entry_price - current_price

                if move < half_r:
                    action = {
                        'timestamp': bar_time_str,
                        'trade_id': tid,
                        'bar_number': bars,
                        'action': 'stale_exit',
                        'trigger': (
                            f'move={move / self._pip_size:.1f}pip '
                            f'< 0.5R={half_r / self._pip_size:.1f}pip '
                            f'after {bars} bars'
                        ),
                    }
                    actions.append(action)
                    self._log_mgmt_action(action)
                    logger.info(
                        f"Trade {tid}: STALE EXIT after {bars} bars "
                        f"(move={move / self._pip_size:.1f}pip "
                        f"< {half_r / self._pip_size:.1f}pip)"
                    )
                    if client is not None:
                        try:
                            client.close_trade(tid)
                        except Exception as e:
                            logger.error(f"Failed to close trade {tid} (stale): {e}")
                    continue  # Trade is closing, skip management

            new_sl = None

            # ── 3. Breakeven (one-time trigger) ───────────────────
            # Backtester: once pos_be_triggered is set, never re-checks
            if self.use_break_even and not self._be_triggered.get(tid, False):
                # Use fixed be_trigger_pips if in params, else compute from ATR
                if self.be_trigger_pips is not None:
                    trigger_pips = self.be_trigger_pips
                else:
                    trigger_pips = self.be_atr_mult * atr_pips

                be_trigger_dist = trigger_pips * self._pip_size
                effective_offset = min(self.be_offset_pips, trigger_pips)

                if is_long:
                    if bar_high - entry_price >= be_trigger_dist:
                        self._be_triggered[tid] = True
                        be_sl = entry_price + effective_offset * self._pip_size
                        if be_sl > current_sl:
                            new_sl = be_sl
                            action = {
                                'timestamp': bar_time_str,
                                'trade_id': tid,
                                'bar_number': bars,
                                'action': 'breakeven',
                                'old_sl': current_sl,
                                'new_sl': new_sl,
                                'trigger': (
                                    f'bar_high={bar_high:.5f} '
                                    f'profit={( bar_high - entry_price) / self._pip_size:.1f}pip '
                                    f'>= {trigger_pips:.1f}pip'
                                ),
                            }
                            actions.append(action)
                            self._log_mgmt_action(action)
                else:
                    if entry_price - bar_low >= be_trigger_dist:
                        self._be_triggered[tid] = True
                        be_sl = entry_price - effective_offset * self._pip_size
                        if be_sl < current_sl:
                            new_sl = be_sl
                            action = {
                                'timestamp': bar_time_str,
                                'trade_id': tid,
                                'bar_number': bars,
                                'action': 'breakeven',
                                'old_sl': current_sl,
                                'new_sl': new_sl,
                                'trigger': (
                                    f'bar_low={bar_low:.5f} '
                                    f'profit={(entry_price - bar_low) / self._pip_size:.1f}pip '
                                    f'>= {trigger_pips:.1f}pip'
                                ),
                            }
                            actions.append(action)
                            self._log_mgmt_action(action)

            # ── 4. Trailing stop ──────────────────────────────────
            if self.use_trailing:

                if self.trail_mode == 0:
                    # ═══ Fixed pip trailing (backtester mode 0) ═══
                    trail_start = self.trail_start_pips * self._pip_size
                    trail_step = self.trail_step_pips * self._pip_size

                    if is_long:
                        current_profit = bar_high - entry_price
                        if not self._trail_active.get(tid, False) and current_profit >= trail_start:
                            self._trail_active[tid] = True
                            self._trail_highs[tid] = bar_high
                        if self._trail_active.get(tid, False):
                            if bar_high > self._trail_highs[tid]:
                                self._trail_highs[tid] = bar_high
                            trail_sl = self._trail_highs[tid] - trail_step
                            if trail_sl > current_sl and (new_sl is None or trail_sl > new_sl):
                                new_sl = trail_sl
                                action = {
                                    'timestamp': bar_time_str,
                                    'trade_id': tid,
                                    'bar_number': bars,
                                    'action': 'trailing',
                                    'old_sl': current_sl,
                                    'new_sl': new_sl,
                                    'trigger': f'trail_high={self._trail_highs[tid]:.5f}',
                                }
                                actions.append(action)
                                self._log_mgmt_action(action)
                    else:
                        current_profit = entry_price - bar_low
                        if not self._trail_active.get(tid, False) and current_profit >= trail_start:
                            self._trail_active[tid] = True
                            self._trail_highs[tid] = bar_low
                        if self._trail_active.get(tid, False):
                            if self._trail_highs[tid] == 0.0 or bar_low < self._trail_highs[tid]:
                                self._trail_highs[tid] = bar_low
                            trail_sl = self._trail_highs[tid] + trail_step
                            if trail_sl < current_sl and (new_sl is None or trail_sl < new_sl):
                                new_sl = trail_sl
                                action = {
                                    'timestamp': bar_time_str,
                                    'trade_id': tid,
                                    'bar_number': bars,
                                    'action': 'trailing',
                                    'old_sl': current_sl,
                                    'new_sl': new_sl,
                                    'trigger': f'trail_low={self._trail_highs[tid]:.5f}',
                                }
                                actions.append(action)
                                self._log_mgmt_action(action)

                elif self.trail_mode == 1:
                    # ═══ Chandelier Exit (ATR-adaptive, active from bar 1) ═══
                    ch_dist = self.chandelier_atr_mult * self._param_atr_pips * self._pip_size

                    if is_long:
                        if bar_high > self._trail_highs[tid] or self._trail_highs[tid] == 0.0:
                            self._trail_highs[tid] = bar_high
                        trail_sl = self._trail_highs[tid] - ch_dist
                        if trail_sl > current_sl and (new_sl is None or trail_sl > new_sl):
                            new_sl = trail_sl
                            self._trail_active[tid] = True
                            action = {
                                'timestamp': bar_time_str,
                                'trade_id': tid,
                                'bar_number': bars,
                                'action': 'chandelier',
                                'old_sl': current_sl,
                                'new_sl': new_sl,
                                'trigger': (
                                    f'trail_high={self._trail_highs[tid]:.5f} '
                                    f'ch_dist={ch_dist / self._pip_size:.1f}pip'
                                ),
                            }
                            actions.append(action)
                            self._log_mgmt_action(action)
                    else:
                        if self._trail_highs[tid] == 0.0 or bar_low < self._trail_highs[tid]:
                            self._trail_highs[tid] = bar_low
                        trail_sl = self._trail_highs[tid] + ch_dist
                        if trail_sl < current_sl and (new_sl is None or trail_sl < new_sl):
                            new_sl = trail_sl
                            self._trail_active[tid] = True
                            action = {
                                'timestamp': bar_time_str,
                                'trade_id': tid,
                                'bar_number': bars,
                                'action': 'chandelier',
                                'old_sl': current_sl,
                                'new_sl': new_sl,
                                'trigger': (
                                    f'trail_low={self._trail_highs[tid]:.5f} '
                                    f'ch_dist={ch_dist / self._pip_size:.1f}pip'
                                ),
                            }
                            actions.append(action)
                            self._log_mgmt_action(action)

            # ── Execute SL modification ───────────────────────────
            if new_sl is not None and client is not None:
                try:
                    client.modify_trade(tid, stop_loss_price=new_sl)
                    pos.stop_loss = new_sl
                    logger.info(
                        f"Trade {tid}: SL {current_sl:.5f} -> {new_sl:.5f} "
                        f"({actions[-1]['action']}, bar={bars})"
                    )
                except Exception as e:
                    logger.error(f"Failed to modify trade {tid}: {e}")

            # ── 5. Partial close ──────────────────────────────────
            # Backtester: close pct of size when price reaches partial_target
            if self.use_partial_close and not self._partial_closed.get(tid, False):
                partial_target = self.partial_target_pips * self._pip_size
                triggered = False
                if is_long:
                    if bar_high - entry_price >= partial_target:
                        triggered = True
                else:
                    if entry_price - bar_low >= partial_target:
                        triggered = True

                if triggered:
                    self._partial_closed[tid] = True
                    partial_units = int(pos.units * self.partial_close_pct)
                    if partial_units > 0 and client is not None:
                        try:
                            client.close_trade(tid, units=str(partial_units))
                            action = {
                                'timestamp': bar_time_str,
                                'trade_id': tid,
                                'bar_number': bars,
                                'action': 'partial_close',
                                'units_closed': partial_units,
                                'pct': self.partial_close_pct,
                                'trigger': f'target={self.partial_target_pips}pip reached',
                            }
                            actions.append(action)
                            self._log_mgmt_action(action)
                            logger.info(
                                f"Trade {tid}: PARTIAL CLOSE {partial_units} units "
                                f"({self.partial_close_pct * 100:.0f}%)"
                            )
                        except Exception as e:
                            logger.error(f"Failed partial close trade {tid}: {e}")
                            self._partial_closed[tid] = False

        # Clean up state for closed positions
        active_ids = {pos.trade_id for pos in positions if pos.instrument == self.pair}
        for tid in list(self._bars_in_trade.keys()):
            if tid not in active_ids:
                self._bars_in_trade.pop(tid, None)
                self._trail_highs.pop(tid, None)
                self._trail_active.pop(tid, None)
                self._be_triggered.pop(tid, None)
                self._partial_closed.pop(tid, None)

        # Persist state for restart recovery
        self._save_mgmt_state()

        return actions
