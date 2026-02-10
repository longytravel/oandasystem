"""
Live trading engine.

Main trading loop that monitors candles and executes trades
based on strategy signals.
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from loguru import logger

from strategies.base import Strategy, Signal, SignalType
from live.oanda_client import OandaClient
from live.position_manager import PositionManager, LivePosition
from live.risk_manager import RiskManager
from config.settings import settings


class LiveTrader:
    """
    Live trading engine.

    Main Loop:
    1. Wait for candle close (e.g., H1 = top of hour)
    2. Fetch recent candles from OANDA
    3. Run strategy.generate_signals()
    4. If signal on current candle -> risk check -> execute
    5. Sync positions with OANDA (detect SL/TP hits)
    6. Repeat
    """

    # Timeframe to minutes mapping
    TIMEFRAME_MINUTES = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H2": 120, "H4": 240,
        "D": 1440
    }

    def __init__(
        self,
        strategy: Strategy,
        instrument: str,
        timeframe: str = "H1",
        dry_run: bool = True,
        candles_needed: int = 200,
        risk_per_trade: float = None,
        state_dir: Optional[str] = None,
        instance_id: Optional[str] = None,
    ):
        """
        Initialize live trader.

        Args:
            strategy: Strategy instance to use
            instrument: Currency pair (e.g., "GBP_USD")
            timeframe: Timeframe (e.g., "H1")
            dry_run: If True, log signals but don't execute trades
            candles_needed: Number of candles to fetch for signal generation
            risk_per_trade: Risk per trade as % of equity (default from settings)
            state_dir: Directory for position state (for instance isolation)
            instance_id: Unique instance identifier (for multi-strategy deployment)
        """
        self.strategy = strategy
        self.instrument = instrument
        self.timeframe = timeframe
        self.dry_run = dry_run
        self.candles_needed = candles_needed
        self.risk_per_trade = risk_per_trade or settings.MAX_RISK_PER_TRADE
        self.instance_id = instance_id

        # Initialize components
        self.client = OandaClient()
        self.position_manager = PositionManager(
            state_dir=Path(state_dir) if state_dir else None
        )
        self.risk_manager = RiskManager(self.position_manager)

        # State
        self.running = False
        self.last_signal_time: Optional[datetime] = None
        self.last_candle_time: Optional[datetime] = None
        self.error_count: int = 0

        # Validate timeframe
        if timeframe not in self.TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        logger.info(f"LiveTrader initialized:")
        logger.info(f"  Strategy: {strategy.name}")
        logger.info(f"  Instrument: {instrument}")
        logger.info(f"  Timeframe: {timeframe}")
        logger.info(f"  Dry Run: {dry_run}")
        logger.info(f"  Risk per trade: {self.risk_per_trade}%")

    def _write_heartbeat(self, status: str = "running", last_candle: str = ""):
        """Write health.json heartbeat if instance_id is set."""
        if not self.instance_id:
            return
        try:
            from live.health import write_heartbeat
            # Write to instance dir (same parent as state_dir)
            instance_dir = self.position_manager.state_dir.parent
            write_heartbeat(
                instance_id=self.instance_id,
                status=status,
                positions=self.position_manager.open_position_count,
                last_candle=last_candle,
                errors=self.error_count,
                instance_dir=instance_dir,
            )
        except Exception as e:
            logger.debug(f"Heartbeat write failed: {e}")

    def _setup_client(self):
        """Setup OANDA client with account ID."""
        if not self.client.account_id:
            accounts = self.client.get_accounts()
            if not accounts:
                raise ValueError("No OANDA accounts found")
            self.client.account_id = accounts[0]['id']
            logger.info(f"Using account: {self.client.account_id}")

    def _get_next_candle_time(self) -> datetime:
        """Calculate when the next candle closes."""
        now = datetime.utcnow()
        minutes = self.TIMEFRAME_MINUTES[self.timeframe]

        if minutes < 60:
            # Sub-hourly: align to minute boundary
            next_minute = ((now.minute // minutes) + 1) * minutes
            if next_minute >= 60:
                next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_time = now.replace(minute=next_minute, second=0, microsecond=0)
        elif minutes == 60:
            # Hourly: top of next hour
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif minutes < 1440:
            # Multi-hour: align to timeframe
            hours = minutes // 60
            next_hour = ((now.hour // hours) + 1) * hours
            if next_hour >= 24:
                next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        else:
            # Daily: midnight
            next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        return next_time

    def _wait_for_candle_close(self):
        """Wait until the next candle closes."""
        next_candle = self._get_next_candle_time()
        wait_seconds = (next_candle - datetime.utcnow()).total_seconds()

        if wait_seconds > 0:
            logger.info(f"Waiting {wait_seconds:.0f}s for next candle close at {next_candle}")

            # Wait in chunks to allow interruption
            while wait_seconds > 0 and self.running:
                sleep_time = min(30, wait_seconds)
                time.sleep(sleep_time)
                wait_seconds = (next_candle - datetime.utcnow()).total_seconds()

    def _get_account_info(self) -> Dict[str, float]:
        """Get current account information."""
        account = self.client.get_account_summary()
        return {
            'balance': float(account.get('balance', 0)),
            'nav': float(account.get('NAV', 0)),
            'unrealized_pnl': float(account.get('unrealizedPL', 0)),
            'margin_used': float(account.get('marginUsed', 0)),
            'margin_available': float(account.get('marginAvailable', 0)),
        }

    def _get_current_spread(self) -> float:
        """Get current spread in pips."""
        price = self.client.get_price(self.instrument)
        if not price:
            return 999.0  # High value to prevent trading

        bid = float(price.get('bids', [{}])[0].get('price', 0))
        ask = float(price.get('asks', [{}])[0].get('price', 0))

        pip_size = 0.0001 if 'JPY' not in self.instrument else 0.01
        spread_pips = (ask - bid) / pip_size

        return spread_pips

    def _fetch_candles(self):
        """Fetch recent candles for signal generation."""
        df = self.client.get_candles(
            instrument=self.instrument,
            granularity=self.timeframe,
            count=self.candles_needed
        )

        if df.empty:
            logger.warning("No candles received")
            return None

        logger.debug(f"Fetched {len(df)} candles, latest: {df.index[-1]}")
        return df

    def _check_for_signal(self, df) -> Optional[Signal]:
        """
        Check if strategy generates a signal on the latest candle.

        Returns:
            Signal if one is generated on the latest candle, None otherwise
        """
        signals = self.strategy.generate_signals(df)

        if not signals:
            return None

        # Check if the latest signal is on the most recent candle
        latest_candle_time = df.index[-1]
        latest_signal = signals[-1]

        if latest_signal.timestamp == latest_candle_time:
            # Avoid duplicate signals
            if self.last_signal_time == latest_signal.timestamp:
                logger.debug("Signal already processed")
                return None

            self.last_signal_time = latest_signal.timestamp
            return latest_signal

        return None

    def _execute_signal(self, signal: Signal, account_info: Dict[str, float]) -> bool:
        """
        Execute a trade based on signal.

        Args:
            signal: Trading signal
            account_info: Current account info

        Returns:
            True if trade was executed successfully
        """
        direction = "BUY" if signal.type == SignalType.BUY else "SELL"

        logger.info(f"Signal: {direction} {self.instrument} @ {signal.price:.5f}")
        logger.info(f"  SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}")

        if self.dry_run:
            logger.info("DRY RUN - Trade not executed")
            return False

        # Calculate position size
        pip_size = 0.0001 if 'JPY' not in self.instrument else 0.01
        sl_pips = abs(signal.price - signal.stop_loss) / pip_size

        units = self.risk_manager.calculate_position_size(
            balance=account_info['balance'],
            risk_pct=self.risk_per_trade,
            stop_loss_pips=sl_pips
        )

        # Adjust sign for direction
        if signal.type == SignalType.SELL:
            units = -units

        # Place order
        try:
            result = self.client.market_order(
                instrument=self.instrument,
                units=units,
                stop_loss_price=signal.stop_loss,
                take_profit_price=signal.take_profit
            )

            if 'orderFillTransaction' in result:
                fill = result['orderFillTransaction']
                trade_id = fill.get('tradeOpened', {}).get('tradeID')
                fill_price = float(fill.get('price', 0))

                # Track position
                position = LivePosition(
                    trade_id=trade_id,
                    instrument=self.instrument,
                    direction=direction,
                    units=abs(units),
                    entry_price=fill_price,
                    entry_time=datetime.utcnow(),
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    metadata=signal.metadata
                )
                self.position_manager.add_position(position)

                logger.info(f"Trade executed: #{trade_id} {direction} {abs(units)} @ {fill_price:.5f}")
                return True

            elif 'orderCancelTransaction' in result:
                reason = result['orderCancelTransaction'].get('reason', 'Unknown')
                logger.error(f"Order cancelled: {reason}")
                return False

            else:
                logger.error(f"Unexpected order result: {result}")
                return False

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return False

    def _sync_positions(self):
        """Sync local positions with broker."""
        try:
            broker_trades = self.client.get_open_trades()
            self.position_manager.sync_with_broker(broker_trades)
        except Exception as e:
            logger.error(f"Position sync failed: {e}")

    def run_once(self) -> Dict[str, Any]:
        """
        Run a single iteration of the trading loop.

        Returns:
            Dict with iteration results
        """
        result = {
            'timestamp': datetime.utcnow(),
            'signal': None,
            'trade_executed': False,
            'error': None
        }

        try:
            # Get account info
            account_info = self._get_account_info()
            self.position_manager.initialize_daily_stats(account_info['balance'])
            self.position_manager.update_balance(account_info['nav'])

            # Get current spread
            spread_pips = self._get_current_spread()

            # Risk check
            can_trade, reason, details = self.risk_manager.can_trade(
                instrument=self.instrument,
                current_balance=account_info['nav'],
                peak_balance=self.position_manager.daily_stats.peak_balance,
                current_spread_pips=spread_pips
            )

            if not can_trade:
                logger.warning(f"Risk check failed: {reason}")
                result['error'] = reason
                return result

            # Fetch candles
            df = self._fetch_candles()
            if df is None or df.empty:
                result['error'] = "No candle data"
                return result

            # Check for signal
            signal = self._check_for_signal(df)
            result['signal'] = signal

            if signal:
                # Execute trade
                executed = self._execute_signal(signal, account_info)
                result['trade_executed'] = executed

            # Manage positions (trailing stop, breakeven)
            if hasattr(self.strategy, 'manage_positions'):
                positions = self.position_manager.get_all_positions()
                if positions and df is not None and not df.empty:
                    # Compute ATR for breakeven calculation
                    pip_size = 0.01 if 'JPY' in self.instrument else 0.0001
                    if len(df) >= 14:
                        highs = df['high'].values[-14:]
                        lows = df['low'].values[-14:]
                        closes = df['close'].values[-14:]
                        tr = []
                        for i in range(1, len(highs)):
                            tr.append(max(highs[i] - lows[i],
                                         abs(highs[i] - closes[i-1]),
                                         abs(lows[i] - closes[i-1])))
                        atr_pips = (sum(tr) / len(tr)) / pip_size if tr else 30.0
                    else:
                        atr_pips = 30.0

                    current_price = float(df['close'].iloc[-1])
                    mgmt_client = self.client if not self.dry_run else None
                    actions = self.strategy.manage_positions(
                        positions, current_price, atr_pips, client=mgmt_client
                    )
                    if actions:
                        result['management_actions'] = actions

            # Sync positions
            self._sync_positions()

        except Exception as e:
            logger.error(f"Iteration error: {e}")
            result['error'] = str(e)

        return result

    def run(self):
        """
        Run the main trading loop.

        Runs until stopped via stop() method.
        """
        logger.info("Starting live trading loop...")

        self._setup_client()
        self.running = True

        # Initial sync
        self._sync_positions()

        while self.running:
            try:
                # Wait for candle close
                self._wait_for_candle_close()

                if not self.running:
                    break

                # Small delay to ensure candle is complete
                time.sleep(2)

                # Run iteration
                result = self.run_once()

                if result['signal']:
                    direction = "BUY" if result['signal'].type == SignalType.BUY else "SELL"
                    logger.info(f"Signal detected: {direction}")
                    if result['trade_executed']:
                        logger.info("Trade executed successfully")
                else:
                    logger.debug("No signal on this candle")

                if result.get('error'):
                    self.error_count += 1
                else:
                    self.error_count = 0

                self._write_heartbeat(
                    status="running",
                    last_candle=str(result.get('timestamp', '')),
                )

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt - stopping")
                break
            except Exception as e:
                self.error_count += 1
                logger.error(f"Loop error: {e}")
                self._write_heartbeat(status="error")
                time.sleep(10)  # Wait before retrying

        self._write_heartbeat(status="stopped")
        logger.info("Trading loop stopped")

    def stop(self):
        """Stop the trading loop."""
        logger.info("Stopping trader...")
        self.running = False

    def get_status(self) -> Dict[str, Any]:
        """Get current trader status."""
        account = {}
        try:
            account = self._get_account_info()
        except Exception:
            pass

        return {
            'running': self.running,
            'dry_run': self.dry_run,
            'instrument': self.instrument,
            'timeframe': self.timeframe,
            'strategy': self.strategy.name,
            'last_signal_time': self.last_signal_time,
            'account': account,
            'positions': [
                {
                    'trade_id': p.trade_id,
                    'direction': p.direction,
                    'units': p.units,
                    'entry_price': p.entry_price,
                    'unrealized_pnl': p.unrealized_pnl
                }
                for p in self.position_manager.get_all_positions()
            ],
            'risk': self.risk_manager.get_status(),
            'daily_summary': self.position_manager.get_daily_summary()
        }
