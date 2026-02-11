"""
Position manager for live trading.

Tracks open positions, daily statistics, and persists state
between sessions.
"""
import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from loguru import logger

from config.settings import settings


@dataclass
class LivePosition:
    """Tracks a live position."""
    trade_id: str
    instrument: str
    direction: str  # "BUY" or "SELL"
    units: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            'entry_time': self.entry_time.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LivePosition':
        """Create from dictionary."""
        data = data.copy()
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        return cls(**data)


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    starting_balance: float
    current_balance: float
    trades_opened: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    peak_balance: float = 0.0
    max_drawdown: float = 0.0

    @property
    def net_pnl(self) -> float:
        """Net profit/loss for the day."""
        return self.gross_profit + self.gross_loss

    @property
    def daily_return_pct(self) -> float:
        """Daily return as percentage."""
        if self.starting_balance == 0:
            return 0.0
        return (self.current_balance - self.starting_balance) / self.starting_balance * 100

    @property
    def win_rate(self) -> float:
        """Win rate for closed trades."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'date': self.date.isoformat(),
            'starting_balance': self.starting_balance,
            'current_balance': self.current_balance,
            'trades_opened': self.trades_opened,
            'trades_closed': self.trades_closed,
            'wins': self.wins,
            'losses': self.losses,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'peak_balance': self.peak_balance,
            'max_drawdown': self.max_drawdown,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DailyStats':
        """Create from dictionary."""
        data = data.copy()
        data['date'] = date.fromisoformat(data['date'])
        return cls(**data)


class PositionManager:
    """
    Manages positions and trading state for live trading.

    Features:
    - Track open positions
    - Maintain daily statistics
    - Persist state to disk
    - Sync with broker (OANDA)
    """

    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize position manager.

        Args:
            state_dir: Directory for state files (defaults to project/data/state)
        """
        self.state_dir = state_dir or settings.DATA_DIR / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.positions: Dict[str, LivePosition] = {}  # trade_id -> position
        self.daily_stats: Optional[DailyStats] = None
        self.trade_history: List[dict] = []

        # Load persisted state
        self._load_state()

    def _get_state_file(self) -> Path:
        """Get path to state file."""
        return self.state_dir / "position_state.json"

    def _get_history_file(self) -> Path:
        """Get path to trade history file."""
        return self.state_dir / "trade_history.json"

    def _load_state(self):
        """Load state from disk."""
        state_file = self._get_state_file()

        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)

                # Load positions
                for pos_data in data.get('positions', []):
                    pos = LivePosition.from_dict(pos_data)
                    self.positions[pos.trade_id] = pos

                # Load daily stats
                if data.get('daily_stats'):
                    self.daily_stats = DailyStats.from_dict(data['daily_stats'])

                    # Reset if new day
                    if self.daily_stats.date != date.today():
                        logger.info("New day - resetting daily stats")
                        self.daily_stats = None

                logger.info(f"Loaded state: {len(self.positions)} positions")

            except Exception as e:
                logger.warning(f"Could not load state: {e}")

        # Load trade history
        history_file = self._get_history_file()
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.trade_history = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load history: {e}")

    def save_state(self):
        """Save state to disk."""
        state_file = self._get_state_file()

        data = {
            'positions': [pos.to_dict() for pos in self.positions.values()],
            'daily_stats': self.daily_stats.to_dict() if self.daily_stats else None,
            'last_updated': datetime.now().isoformat()
        }

        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)

        # Save trade history
        history_file = self._get_history_file()
        with open(history_file, 'w') as f:
            json.dump(self.trade_history, f, indent=2)

        logger.debug("State saved")

    def initialize_daily_stats(self, balance: float):
        """Initialize daily stats for a new day."""
        today = date.today()

        if self.daily_stats is None or self.daily_stats.date != today:
            self.daily_stats = DailyStats(
                date=today,
                starting_balance=balance,
                current_balance=balance,
                peak_balance=balance
            )
            logger.info(f"Initialized daily stats - Starting balance: {balance:.2f}")
            self.save_state()

    def update_balance(self, balance: float):
        """Update current balance and track drawdown."""
        if self.daily_stats is None:
            self.initialize_daily_stats(balance)
            return

        self.daily_stats.current_balance = balance

        # Track peak and drawdown
        if balance > self.daily_stats.peak_balance:
            self.daily_stats.peak_balance = balance

        drawdown = self.daily_stats.peak_balance - balance
        if drawdown > self.daily_stats.max_drawdown:
            self.daily_stats.max_drawdown = drawdown

    def add_position(self, position: LivePosition):
        """Add a new position."""
        self.positions[position.trade_id] = position

        if self.daily_stats:
            self.daily_stats.trades_opened += 1

        logger.info(f"Position added: {position.trade_id} - {position.direction} "
                   f"{position.units} {position.instrument} @ {position.entry_price:.5f}")
        self.save_state()

    def remove_position(
        self,
        trade_id: str,
        exit_price: float,
        realized_pnl: float,
        exit_reason: str = "UNKNOWN"
    ) -> Optional[LivePosition]:
        """
        Remove a closed position.

        Args:
            trade_id: OANDA trade ID
            exit_price: Exit price
            realized_pnl: Realized P&L
            exit_reason: Reason for exit (TP, SL, MANUAL, etc.)

        Returns:
            The removed position, or None if not found
        """
        if trade_id not in self.positions:
            logger.warning(f"Position {trade_id} not found")
            return None

        position = self.positions.pop(trade_id)

        # Update daily stats
        if self.daily_stats:
            self.daily_stats.trades_closed += 1
            if realized_pnl > 0:
                self.daily_stats.wins += 1
                self.daily_stats.gross_profit += realized_pnl
            else:
                self.daily_stats.losses += 1
                self.daily_stats.gross_loss += realized_pnl

        # Add to history
        history_entry = {
            'trade_id': trade_id,
            'instrument': position.instrument,
            'direction': position.direction,
            'units': position.units,
            'entry_price': position.entry_price,
            'entry_time': position.entry_time.isoformat(),
            'exit_price': exit_price,
            'exit_time': datetime.now().isoformat(),
            'exit_reason': exit_reason,
            'realized_pnl': realized_pnl,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
        }
        self.trade_history.append(history_entry)

        logger.info(f"Position closed: {trade_id} - {exit_reason} @ {exit_price:.5f} "
                   f"| P&L: {realized_pnl:+.2f}")
        self.save_state()

        return position

    def get_position(self, trade_id: str) -> Optional[LivePosition]:
        """Get a position by trade ID."""
        return self.positions.get(trade_id)

    def get_positions_for_instrument(self, instrument: str) -> List[LivePosition]:
        """Get all positions for an instrument."""
        return [p for p in self.positions.values() if p.instrument == instrument]

    def get_all_positions(self) -> List[LivePosition]:
        """Get all open positions."""
        return list(self.positions.values())

    def sync_with_broker(self, broker_trades: List[dict], oanda_client=None):
        """
        Sync local positions with broker state.

        Detects trades that were closed externally (SL/TP hit).
        If oanda_client is provided, fetches real exit price and P&L from OANDA.

        Args:
            broker_trades: List of open trades from broker
            oanda_client: Optional OandaClient to fetch closed trade details
        """
        broker_trade_ids = {t['id'] for t in broker_trades}
        local_trade_ids = set(self.positions.keys())

        # Find positions closed externally
        closed_externally = local_trade_ids - broker_trade_ids

        for trade_id in closed_externally:
            position = self.positions[trade_id]
            logger.info(f"Trade {trade_id} closed externally (SL/TP hit)")

            # Try to get real exit details from OANDA
            exit_price = position.entry_price
            realized_pnl = 0.0
            exit_reason = "EXTERNAL"

            if oanda_client:
                try:
                    trade_details = oanda_client.get_trade(trade_id)
                    if trade_details:
                        realized_pnl = float(trade_details.get('realizedPL', 0.0))
                        exit_price = float(trade_details.get('averageClosePrice', position.entry_price))

                        # Determine exit reason from trade state/stopLossOrder/takeProfitOrder
                        state = trade_details.get('state', '')
                        if state == 'CLOSED':
                            # Check if SL or TP was filled
                            sl_order = trade_details.get('stopLossOrder', {})
                            tp_order = trade_details.get('takeProfitOrder', {})
                            if sl_order.get('state') == 'FILLED':
                                exit_reason = "SL"
                            elif tp_order.get('state') == 'FILLED':
                                exit_reason = "TP"
                            else:
                                exit_reason = "CLOSED"

                        logger.info(f"Trade {trade_id}: exit_price={exit_price:.5f}, "
                                   f"P&L={realized_pnl:+.2f}, reason={exit_reason}")
                except Exception as e:
                    logger.warning(f"Could not fetch trade details for {trade_id}: {e}")

            self.remove_position(
                trade_id=trade_id,
                exit_price=exit_price,
                realized_pnl=realized_pnl,
                exit_reason=exit_reason,
            )

        # Update unrealized P&L for open positions
        for trade in broker_trades:
            trade_id = trade['id']
            if trade_id in self.positions:
                unrealized_pnl = float(trade.get('unrealizedPL', 0))
                self.positions[trade_id].unrealized_pnl = unrealized_pnl

        if closed_externally:
            logger.info(f"Sync complete: {len(closed_externally)} positions closed externally")

    @property
    def open_position_count(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    @property
    def daily_trade_count(self) -> int:
        """Number of trades opened today."""
        if self.daily_stats:
            return self.daily_stats.trades_opened
        return 0

    def get_daily_summary(self) -> str:
        """Get formatted daily summary."""
        if not self.daily_stats:
            return "No daily stats available"

        stats = self.daily_stats
        return f"""
Daily Summary ({stats.date})
{'=' * 40}
Starting Balance: {stats.starting_balance:,.2f}
Current Balance:  {stats.current_balance:,.2f}
Daily P&L:        {stats.net_pnl:+,.2f} ({stats.daily_return_pct:+.2f}%)
Max Drawdown:     {stats.max_drawdown:,.2f}
Trades Opened:    {stats.trades_opened}
Trades Closed:    {stats.trades_closed}
Wins/Losses:      {stats.wins}/{stats.losses} ({stats.win_rate:.1%})
Gross Profit:     {stats.gross_profit:+,.2f}
Gross Loss:       {stats.gross_loss:+,.2f}
{'=' * 40}
"""

    def clear_state(self):
        """Clear all state (for testing/reset)."""
        self.positions.clear()
        self.daily_stats = None

        state_file = self._get_state_file()
        if state_file.exists():
            state_file.unlink()

        logger.warning("State cleared")
